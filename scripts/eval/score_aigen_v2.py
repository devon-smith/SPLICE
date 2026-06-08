# score the 10-pair Veo/Dispatch pilot through v2 LoRA
# v2's backbone is LoRA-tuned -- can't reuse the frozen-DINOv2 embedding cache.
# hard-codes the pilot keyframe location, checkpoint dir, and Dispatch buckets.
# usage: python scripts/eval/score_aigen_v2.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms
from transformers import AutoModel
from peft import PeftModel

REPO = Path(__file__).resolve().parents[2]
KF_DIR = "/mnt/disks/splice-data/outputs/aigen_eval/keyframes"
PRIOR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
SEED_DIRS = ["/mnt/disks/splice-data/outputs/v2_lora_sweep/r8_a32_lrbb5e-05"]

BUCKET_OF = {"A003": "clean", "A013": "clean",
             "A001": "drift", "A002": "drift", "A004": "drift",
             "A011": "drift", "A012": "drift", "A014": "drift",
             "A005": "major", "A015": "major"}
BUCKET_RANK = {"clean": 0, "drift": 1, "major": 2}

tfm = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# LayerNorm([eL | eR | |eL-eR| | cos]) -> scalar logit.
# use_norm=False for checkpoints saved before LayerNorm was added.
class PairMLP(nn.Module):
    def __init__(self, emb_dim=768, hidden=(512, 128), dropout=0.1, use_norm=True):
        super().__init__()
        in_dim = 2 * emb_dim + emb_dim + 1
        self.norm = nn.LayerNorm(in_dim) if use_norm else nn.Identity()
        layers = []
        dim = in_dim
        for w in hidden:
            layers += [nn.Linear(dim, w), nn.ReLU(), nn.Dropout(dropout)]
            dim = w
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, e_l, e_r):
        abs_diff = (e_l - e_r).abs()
        cos = F.cosine_similarity(e_l, e_r, eps=1e-8).unsqueeze(-1)
        x = torch.cat([e_l, e_r, abs_diff, cos], dim=-1)
        return self.net(self.norm(x)).squeeze(-1)


def load_seed(seed_dir, device):
    base = AutoModel.from_pretrained("facebook/dinov2-base")
    backbone = PeftModel.from_pretrained(base, str(seed_dir / "backbone_lora")).to(device)
    state = torch.load(seed_dir / "head.pt", map_location=device, weights_only=False)
    head = PairMLP(use_norm="norm.weight" in state).to(device)
    head.load_state_dict(state)
    backbone.eval()
    head.eval()
    return backbone, head


@torch.inference_mode()
def score_pair(backbone, head, left, right, device):
    pv_l = tfm(Image.open(left).convert("RGB")).unsqueeze(0).to(device)
    pv_r = tfm(Image.open(right).convert("RGB")).unsqueeze(0).to(device)
    out = backbone(pixel_values=torch.cat([pv_l, pv_r], dim=0))
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        pooled = out.last_hidden_state[:, 0]
    emb = pooled.float()
    return torch.sigmoid(head(emb[:1], emb[1:])).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kf_dir", default=KF_DIR)
    ap.add_argument("--prior_scores_csv", default=PRIOR_CSV)
    ap.add_argument("--out_csv", default=str(REPO / "reports/aigen_v2_pilot.csv"))
    ap.add_argument("--out_json", default=str(REPO / "reports/aigen_v2_pilot_metrics.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kf_dir = Path(args.kf_dir)
    pair_ids = sorted(d.name for d in kf_dir.iterdir() if d.is_dir())
    n_seeds = len(SEED_DIRS)
    print(f"scoring {len(pair_ids)} Veo pairs across {n_seeds} v2 seed(s) on {device}")

    seed_scores = {}
    for s, sd in enumerate(SEED_DIRS):
        print(f"seed {s}: {sd}")
        backbone, head = load_seed(Path(sd), device)
        seed_scores[s] = {
            pid: score_pair(backbone, head, kf_dir / pid / "left_img2.jpg",
                            kf_dir / pid / "right_img0.jpg", device)
            for pid in pair_ids
        }
        del backbone, head
        if device == "cuda":
            torch.cuda.empty_cache()

    prior = pd.read_csv(args.prior_scores_csv, dtype={"pair_id": str}).set_index("pair_id")

    rows = []
    for pid in pair_ids:
        ss = [seed_scores[s][pid] for s in range(n_seeds)]
        row = {
            "pair_id": pid,
            "bucket": BUCKET_OF[pid],
            "bucket_rank": BUCKET_RANK[BUCKET_OF[pid]],
            "v2_ensemble_mean": np.mean(ss),
            "v2_ensemble_std": np.std(ss, ddof=1) if n_seeds > 1 else 0.0,
            "v0_logistic": prior.loc[pid, "v0_logistic"],
            "v1_5_MLP": prior.loc[pid, "v1.5_MLP"],
            "clip_cos": prior.loc[pid, "clip_cos"],
        }
        for i, s in enumerate(ss):
            row[f"v2_seed{i}"] = s
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("v2_ensemble_mean", ascending=False).reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)

    # Spearman vs Dispatch bucket rank for each model
    ranks = df["bucket_rank"].to_numpy()
    metrics = {}
    for m in ("v0_logistic", "v1_5_MLP", "v2_ensemble_mean", "clip_cos"):
        rho, p = spearmanr(df[m].to_numpy(), ranks)
        metrics[m] = {"spearman": rho, "p_value": p}

    bucket_means = {}
    for b in ("clean", "drift", "major"):
        g = df[df["bucket"] == b]
        bucket_means[b] = {
            "n": len(g),
            "v0_mean": g["v0_logistic"].mean(),
            "v1_5_mean": g["v1_5_MLP"].mean(),
            "v2_mean": g["v2_ensemble_mean"].mean(),
        }

    summary = {
        "spearman_vs_bucket_rank": metrics,
        "bucket_means": bucket_means,
        "v2_top2": df.head(2)["pair_id"].tolist(),
        "v2_bottom2": df.tail(2)["pair_id"].tolist(),
        "agree_top": set(df.head(2)["pair_id"]) == {"A005", "A015"},
        "agree_bottom": set(df.tail(2)["pair_id"]) == {"A003", "A013"},
        "n_pairs": len(df),
        "n_seeds": n_seeds,
        "seed_dirs": [str(p) for p in SEED_DIRS],
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nper-pair (sorted by v2 ensemble mean, {n_seeds} seed(s))")
    print(df[["pair_id", "bucket", "v2_ensemble_mean", "v2_ensemble_std",
              "v1_5_MLP", "v0_logistic", "clip_cos"]].round(4).to_string(index=False))
    print("\nSpearman rho vs bucket order (clean<drift<major)")
    for m, r in metrics.items():
        print(f"  {m:20s}  rho={r['spearman']:+.4f}  p={r['p_value']:.4f}")
    print(f"\ntop-2 by v2: {summary['v2_top2']}  (matches {{A005,A015}}: {summary['agree_top']})")
    print(f"bottom-2:   {summary['v2_bottom2']}  (matches {{A003,A013}}: {summary['agree_bottom']})")
    print("\nbucket means")
    for b, m in bucket_means.items():
        print(f"  {b:<6} n={m['n']}  v0 {m['v0_mean']:.3f}  v1.5 {m['v1_5_mean']:.3f}  "
              f"v2 {m['v2_mean']:.3f}")
    print(f"\nsaved {args.out_csv} + {args.out_json}")


if __name__ == "__main__":
    main()
