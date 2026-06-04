# visualise top-scoring AI-gen pairs from an eval run
# renders a grid of (left_img2 | right_img0) sorted by model score, highest first
# usage: python scripts/eval/inspect_aigen_top.py --aigen_index PATH --model v1.5

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image

from src.data.pairs import build_pair_features_batch
from src.models.dinov2_encoder import DINOv2Encoder

REPO = Path(__file__).resolve().parents[2]
V1_SCALER = "/mnt/disks/splice-data/outputs/v1_sound/scaler.joblib"
V1_CKPT = "/mnt/disks/splice-data/outputs/v1_sound/v1_sound_seed2.pt"


# inlined here so this script doesn't depend on the v1.5 training script
class MLPHead(nn.Module):
    def __init__(self, in_dim=2305, hidden=(512, 128), dropout=0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for w in hidden:
            layers += [nn.Linear(dim, w), nn.ReLU(), nn.Dropout(dropout)]
            dim = w
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# run only the requested model (v1.5 or raw cosine); no need for the full eval harness
def score_inline(cuts, model):
    encoder = DINOv2Encoder()
    paths = sorted(set(cuts[["left_img2_path", "right_img0_path"]].to_numpy().ravel()))
    emb = {}
    bs = 64
    for i in range(0, len(paths), bs):
        chunk = paths[i : i + bs]
        pv = torch.stack([
            encoder.processor(images=Image.open(p).convert("RGB"),
                              return_tensors="pt")["pixel_values"][0]
            for p in chunk
        ])
        batch_emb = encoder.encode(pv).numpy()
        for p, e in zip(chunk, batch_emb):
            emb[p] = e

    e_left = np.stack([emb[p] for p in cuts["left_img2_path"]])
    e_right = np.stack([emb[p] for p in cuts["right_img0_path"]])
    feats = build_pair_features_batch(e_left, e_right)

    if model == "v1.5":
        cfg = yaml.safe_load((REPO / "configs" / "v1_sound.yaml").read_text())
        m = MLPHead(in_dim=2305, dropout=cfg["head"]["dropout"])
        m.load_state_dict(torch.load(V1_CKPT, map_location="cpu", weights_only=True))
        m.eval()
        scaler = joblib.load(V1_SCALER)
        with torch.inference_mode():
            x = torch.from_numpy(scaler.transform(feats).astype(np.float32))
            return torch.sigmoid(m(x)).numpy().ravel()
    elif model == "raw_cosine":
        return 1.0 - feats[:, -1]
    raise ValueError(f"inline scorer supports v1.5 and raw_cosine, got {model!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aigen_index", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=None,
                    help="if scores.npz exists alongside aigen_results.json, use it")
    ap.add_argument("--model", default="v1.5")
    ap.add_argument("--top_n", type=int, default=12)
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/disks/splice-data/outputs/aigen_eval/top_pairs.png"))
    args = ap.parse_args()

    cuts = pd.read_parquet(args.aigen_index)

    scores = None
    if args.results is not None:
        scores_npz = args.results.parent / "scores.npz"
        if scores_npz.exists():
            d = np.load(scores_npz)
            if args.model in d:
                scores = d[args.model]
    if scores is None:
        print("computing scores inline ...")
        scores = score_inline(cuts, args.model)

    n = min(args.top_n, len(cuts))
    top_idx = np.argsort(scores)[::-1][:n]

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 5, nrows * 2.8))
    axes = axes.reshape(nrows, ncols * 2)

    for k, i in enumerate(top_idx):
        r, c = divmod(k, ncols)
        row = cuts.iloc[i]
        label = "AI-gen" if row["y_inconsistent"] == 1 else "real"
        ax_l, ax_r = axes[r, c * 2], axes[r, c * 2 + 1]
        ax_l.imshow(Image.open(row["left_img2_path"]).convert("RGB"))
        ax_r.imshow(Image.open(row["right_img0_path"]).convert("RGB"))
        ax_l.set_title(f"#{k+1} left  [{label}]", fontsize=7)
        ax_r.set_title(f"score={scores[i]:.3f}", fontsize=7)
        ax_l.axis("off")
        ax_r.axis("off")
    for k in range(len(top_idx), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    fig.suptitle(f"Top-{n} scoring pairs -- {args.model}", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"saved grid -> {args.out}")


if __name__ == "__main__":
    main()
