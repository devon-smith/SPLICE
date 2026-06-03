"""Visualise the top- and bottom-scoring AI-gen pairs from an eval run.

Loads the pilot index parquet + the scores from aigen_results.json and renders
a grid of (left_img2 | right_img0) sorted by model score, highest first.

Usage:
  python scripts/eval/inspect_aigen_top.py \\
      --aigen_index /mnt/disks/splice-data/outputs/aigen_eval/pilot_index.parquet \\
      --results    /mnt/disks/splice-data/outputs/aigen_eval/results/aigen_results.json \\
      --model      v1.5 \\
      --top_n      12 \\
      --out        /mnt/disks/splice-data/outputs/aigen_eval/top_pairs.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))


def _load_scores_from_results(results_path: Path, model: str, n: int) -> np.ndarray:
    """Re-derive per-row scores from the stored JSON.

    eval_aigen.py does not save per-row scores to JSON (only aggregates), so we
    re-run only the model inference here.  If the scores npz is present we use it
    directly; otherwise we re-embed and re-score inline.
    """
    scores_npz = results_path.parent / "scores.npz"
    if scores_npz.exists():
        d = np.load(scores_npz)
        if model in d:
            return d[model]
    raise FileNotFoundError(
        f"Per-row scores not found at {scores_npz}. "
        "Re-run eval_aigen.py with --save_scores (or use the inline scorer below)."
    )


def _score_inline(cuts: pd.DataFrame, model: str) -> np.ndarray:
    """Lightweight inline scorer — runs only the requested model."""
    import torch
    import yaml
    import joblib
    from src.data.pairs import build_pair_features_batch
    from src.models.dinov2_encoder import DINOv2Encoder

    REPO = HERE.parents[2]
    encoder = DINOv2Encoder()
    all_paths = sorted(set(cuts[["left_img2_path", "right_img0_path"]].to_numpy().ravel()))

    emb: dict[str, np.ndarray] = {}
    bs = 64
    for i in range(0, len(all_paths), bs):
        chunk = all_paths[i:i+bs]
        pv = torch.stack([
            encoder.processor(
                images=Image.open(p).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]
            for p in chunk
        ])
        batch_emb = encoder.encode(pv).numpy()
        for p, e in zip(chunk, batch_emb):
            emb[p] = e

    e_left = np.stack([emb[p] for p in cuts["left_img2_path"]])
    e_right = np.stack([emb[p] for p in cuts["right_img0_path"]])
    feats = build_pair_features_batch(e_left, e_right)

    if model == "v1.5":
        from scripts.train.v1_mlp import MLPHead
        cfg = yaml.safe_load((REPO / "configs" / "v1_sound.yaml").read_text())
        m = MLPHead(in_dim=2305, dropout=cfg["head"]["dropout"])
        v1_path = "/mnt/disks/splice-data/outputs/v1_sound/v1_sound_seed2.pt"
        m.load_state_dict(torch.load(v1_path, map_location="cpu", weights_only=True))
        m.eval()
        scaler = joblib.load("/mnt/disks/splice-data/outputs/v1_sound/scaler.joblib")
        with torch.inference_mode():
            x = torch.from_numpy(scaler.transform(feats).astype(np.float32))
            return torch.sigmoid(m(x)).numpy().ravel()
    elif model == "raw_cosine":
        return 1.0 - feats[:, -1]
    else:
        raise ValueError(f"inline scorer only supports v1.5 and raw_cosine, got {model!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aigen_index", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=None,
                    help="aigen_results.json from eval_aigen.py; "
                         "if omitted, scores are computed inline")
    ap.add_argument("--model", default="v1.5")
    ap.add_argument("--top_n", type=int, default=12)
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/disks/splice-data/outputs/aigen_eval/top_pairs.png"))
    args = ap.parse_args()

    cuts = pd.read_parquet(args.aigen_index)

    if args.results is not None:
        try:
            scores = _load_scores_from_results(args.results, args.model, len(cuts))
        except FileNotFoundError:
            print("scores.npz not found — computing inline (slow first run)")
            scores = _score_inline(cuts, args.model)
    else:
        print("no --results given — computing scores inline")
        scores = _score_inline(cuts, args.model)

    n = min(args.top_n, len(cuts))
    top_idx = np.argsort(scores)[::-1][:n]

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 5, nrows * 2.8))
    axes = axes.reshape(nrows, ncols * 2)

    for k, i in enumerate(top_idx):
        r, c = divmod(k, ncols)
        row = cuts.iloc[i]
        score = scores[i]
        label = "AI-gen" if row["y_inconsistent"] == 1 else "real"

        ax_l = axes[r, c * 2]
        ax_r = axes[r, c * 2 + 1]

        ax_l.imshow(Image.open(row["left_img2_path"]).convert("RGB"))
        ax_r.imshow(Image.open(row["right_img0_path"]).convert("RGB"))

        ax_l.set_title(f"#{k+1} left  [{label}]", fontsize=7)
        ax_r.set_title(f"score={score:.3f}", fontsize=7)
        ax_l.axis("off")
        ax_r.axis("off")

    # blank out unused axes
    for k in range(len(top_idx), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    fig.suptitle(f"Top-{n} scoring pairs — {args.model}", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"saved grid -> {args.out}")


if __name__ == "__main__":
    main()
