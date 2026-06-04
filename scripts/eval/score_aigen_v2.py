"""Score the 10 Veo continuous-action pairs through the v2 LoRA 3-seed ensemble.

v2's backbone is LoRA-tuned, so frozen-DINOv2 embeddings can't be reused -- we
have to run each seed's tuned backbone forward on the pair's boundary keyframes
(left_img2 + right_img0). Per-pair score = mean of sigmoid'd logits across the
three best-by-val seeds. Reads Dispatch buckets + prior model scores from
outputs/aigen_eval/results/per_pair_scores.csv to compute Spearman correlations
that match what's reported in earlier per_pair_analysis.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms
from transformers import AutoModel

try:
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
except ImportError:
    raise SystemExit("peft not installed -- run: pip install peft")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_KEYFRAMES = "/mnt/disks/splice-data/outputs/aigen_eval/keyframes"
DEFAULT_PRIOR = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
SEED_DIRS = [
    "/mnt/disks/splice-data/outputs/v2_lora_extended/seed0/r8_a16",
    "/mnt/disks/splice-data/outputs/v2_lora_extended/seed1/r8_a16",
    "/mnt/disks/splice-data/outputs/v2_lora_extended/seed2/r8_a16",
]
BUCKET_OF = {
    "A003": "clean", "A013": "clean",
    "A001": "drift", "A002": "drift", "A004": "drift",
    "A011": "drift", "A012": "drift", "A014": "drift",
    "A005": "major", "A015": "major",
}
BUCKET_RANK = {"clean": 0, "drift": 1, "major": 2}
BUCKET_ORDER = ["clean", "drift", "major"]

_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PairMLP(torch.nn.Module):
    """v1-style MLP head with LayerNorm on the [eL, eR, |eL-eR|, cos] feature."""

    def __init__(self, emb_dim: int = 768, hidden=(512, 128), dropout: float = 0.1) -> None:
        super().__init__()
        in_dim = 2 * emb_dim + emb_dim + 1
        self.norm = torch.nn.LayerNorm(in_dim)
        layers: list[torch.nn.Module] = []
        dim = in_dim
        for w in hidden:
            layers += [torch.nn.Linear(dim, w), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            dim = w
        layers.append(torch.nn.Linear(dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, e_l: torch.Tensor, e_r: torch.Tensor) -> torch.Tensor:
        abs_diff = (e_l - e_r).abs()
        cos = torch.nn.functional.cosine_similarity(e_l, e_r, eps=1e-8).unsqueeze(-1)
        x = torch.cat([e_l, e_r, abs_diff, cos], dim=-1)
        return self.net(self.norm(x)).squeeze(-1)


def load_v2_seed(seed_dir: Path, device: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load LoRA backbone + head from a saved v2 seed directory."""
    base = AutoModel.from_pretrained("facebook/dinov2-base")
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"],
                          lora_dropout=0.1, bias="none")
    backbone = get_peft_model(base, lora_cfg).to(device)
    # peft saved the adapter via backbone.save_pretrained(seed_dir / "backbone_lora")
    adapter_dir = seed_dir / "backbone_lora"
    # peft expects a config.json + adapter_model.safetensors in this dir
    from peft import PeftModel
    backbone = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    head = PairMLP().to(device)
    head_state = torch.load(seed_dir / "head.pt", map_location=device, weights_only=False)
    head.load_state_dict(head_state)
    backbone.eval()
    head.eval()
    return backbone, head


@torch.inference_mode()
def score_pair(backbone, head, left_path: Path, right_path: Path, device: str) -> float:
    pv_l = _TRANSFORM(Image.open(left_path).convert("RGB")).unsqueeze(0).to(device)
    pv_r = _TRANSFORM(Image.open(right_path).convert("RGB")).unsqueeze(0).to(device)
    pv = torch.cat([pv_l, pv_r], dim=0)
    out = backbone(pixel_values=pv)
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state[:, 0]
    emb = pooled.float()
    e_l, e_r = emb[:1], emb[1:]
    logit = head(e_l, e_r)
    return float(torch.sigmoid(logit).cpu().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keyframes_dir", default=DEFAULT_KEYFRAMES)
    ap.add_argument("--prior_scores_csv", default=DEFAULT_PRIOR)
    ap.add_argument("--out_csv", default=str(REPO / "reports/aigen_v2_pilot.csv"))
    ap.add_argument("--out_json", default=str(REPO / "reports/aigen_v2_pilot_metrics.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keyframes_dir = Path(args.keyframes_dir)
    pair_ids = sorted(d.name for d in keyframes_dir.iterdir() if d.is_dir())
    print(f"scoring {len(pair_ids)} Veo pairs across {len(SEED_DIRS)} v2 seeds on {device}")

    seed_scores: dict[int, dict[str, float]] = {}
    for s, sd in enumerate(SEED_DIRS):
        print(f"--- seed {s} : {sd}")
        backbone, head = load_v2_seed(Path(sd), device)
        seed_scores[s] = {}
        for pid in pair_ids:
            left = keyframes_dir / pid / "left_img2.jpg"
            right = keyframes_dir / pid / "right_img0.jpg"
            seed_scores[s][pid] = score_pair(backbone, head, left, right, device)
        del backbone, head
        if device == "cuda":
            torch.cuda.empty_cache()

    prior = pd.read_csv(args.prior_scores_csv)
    prior["pair_id"] = prior["pair_id"].astype(str)
    prior = prior.set_index("pair_id")

    rows = []
    for pid in pair_ids:
        ss = [seed_scores[s][pid] for s in range(3)]
        rows.append({
            "pair_id": pid,
            "bucket": BUCKET_OF[pid],
            "bucket_rank": BUCKET_RANK[BUCKET_OF[pid]],
            "v2_seed0": ss[0],
            "v2_seed1": ss[1],
            "v2_seed2": ss[2],
            "v2_3seed_mean": float(np.mean(ss)),
            "v2_3seed_std": float(np.std(ss, ddof=1)) if len(ss) > 1 else 0.0,
            "v0_logistic": float(prior.loc[pid, "v0_logistic"]),
            "v1_5_MLP": float(prior.loc[pid, "v1.5_MLP"]),
            "clip_cos": float(prior.loc[pid, "clip_cos"]),
        })
    df = pd.DataFrame(rows).sort_values("v2_3seed_mean", ascending=False).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    ranks = df["bucket_rank"].to_numpy()
    metrics = {}
    for model in ("v0_logistic", "v1_5_MLP", "v2_3seed_mean", "clip_cos"):
        rho, p = spearmanr(df[model].to_numpy(), ranks)
        metrics[model] = {"spearman": float(rho), "p_value": float(p)}

    bucket_means = {}
    for b in BUCKET_ORDER:
        g = df[df["bucket"] == b]
        bucket_means[b] = {
            "n": int(len(g)),
            "v0_mean": float(g["v0_logistic"].mean()),
            "v1_5_mean": float(g["v1_5_MLP"].mean()),
            "v2_mean": float(g["v2_3seed_mean"].mean()),
        }

    summary = {
        "spearman_vs_bucket_rank": metrics,
        "bucket_means": bucket_means,
        "v2_top2": df.head(2)["pair_id"].tolist(),
        "v2_bottom2": df.tail(2)["pair_id"].tolist(),
        "agree_top": set(df.head(2)["pair_id"]) == {"A005", "A015"},
        "agree_bottom": set(df.tail(2)["pair_id"]) == {"A003", "A013"},
        "n_pairs": int(len(df)),
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out_json}")

    print("\n=== per-pair (sorted by v2 3-seed mean) ===")
    print(df[["pair_id", "bucket", "v2_3seed_mean", "v2_3seed_std",
              "v1_5_MLP", "v0_logistic", "clip_cos"]].round(4).to_string(index=False))
    print("\n=== Spearman rho vs bucket order (clean<drift<major) ===")
    for m, r in metrics.items():
        print(f"  {m:18s}  rho={r['spearman']:+.4f}  p={r['p_value']:.4f}")
    print(f"\n=== top-2 / bottom-2 by v2 ===")
    print(f"  top:    {summary['v2_top2']}  (matches {{A005,A015}}: {summary['agree_top']})")
    print(f"  bottom: {summary['v2_bottom2']}  (matches {{A003,A013}}: {summary['agree_bottom']})")
    print("\n=== bucket means ===")
    for b, m in bucket_means.items():
        print(f"  {b:<6} n={m['n']}  v0 {m['v0_mean']:.3f}  v1.5 {m['v1_5_mean']:.3f}  v2 {m['v2_mean']:.3f}")


if __name__ == "__main__":
    main()
