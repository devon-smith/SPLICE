# AI-USE: This script was AI-assisted with Claude via Claude Code.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.movienet import keyframe_key
from src.data.pairs import (PAIR_FEATURE_DIM, PAIR_FEATURE_DIM_HADAMARD,
                             build_pair_features_batch, load_embeddings)

CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
EMB_DIR = "/mnt/disks/splice-data/embeddings/dinov2_base"
OUT_TPL = "/mnt/disks/splice-data/pairs/dino_v0_{mode}"

MODE_COLS = {
    "boundary": (["left_img2_path"], ["right_img0_path"]),
    "mean_pool_3": (["left_img0_path", "left_img1_path", "left_img2_path"],
                    ["right_img0_path", "right_img1_path", "right_img2_path"]),
}


# map path columns to embedding row indices; -1 if a key is missing
def rows_for_cols(df, cols, key2row):
    return np.stack(
        [df[c].map(lambda p: key2row.get(keyframe_key(p), -1)).to_numpy() for c in cols],
        axis=1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut_index", default=CUT_INDEX)
    ap.add_argument("--embeddings", default=EMB_DIR)
    ap.add_argument("--mode", choices=list(MODE_COLS), default="boundary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--include_hadamard", action="store_true",
                    help="append eL*eR -> 3073-d output")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(OUT_TPL.format(mode=args.mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.cut_index)
    print(f"cut index: {len(df)} cuts")
    emb, key2row = load_embeddings(args.embeddings)
    print(f"embeddings: {emb.shape[0]} keyframes x {emb.shape[1]} dims")

    left_cols, right_cols = MODE_COLS[args.mode]
    left_rows = rows_for_cols(df, left_cols, key2row)
    right_rows = rows_for_cols(df, right_cols, key2row)

    valid = (left_rows >= 0).all(axis=1) & (right_rows >= 0).all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f"dropping {n_drop} cuts with missing embeddings ({100 * n_drop / len(df):.2f}%)")
    df = df[valid].reset_index(drop=True)
    left_rows, right_rows = left_rows[valid], right_rows[valid]

    # mean over keyframes (1 frame for boundary, 3 for mean_pool_3)
    e_left = emb[left_rows].mean(axis=1)
    e_right = emb[right_rows].mean(axis=1)
    features = build_pair_features_batch(e_left, e_right, include_hadamard=args.include_hadamard)
    expected_dim = PAIR_FEATURE_DIM_HADAMARD if args.include_hadamard else PAIR_FEATURE_DIM
    assert features.shape == (len(df), expected_dim)
    assert not np.isnan(features).any()

    meta = pd.DataFrame({
        "cut_id": df["movie_id"] + "_" + df["shot_left_idx"].astype(str).str.zfill(4),
        "y_inconsistent": df["y_inconsistent"].to_numpy(),
        "split": df["split"].to_numpy(),
        "movie_id": df["movie_id"].to_numpy(),
    })
    np.save(out_dir / "features.npy", features)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "mode": args.mode,
        "n_cuts": int(len(df)),
        "feature_dim": int(expected_dim),
        "include_hadamard": bool(args.include_hadamard),
        "n_dropped_missing_embedding": n_drop,
    }, indent=2))
    print(f"wrote {features.shape[0]} x {features.shape[1]} features -> {out_dir}")
    print(f"positive rate: {100 * meta['y_inconsistent'].mean():.2f}%")


if __name__ == "__main__":
    main()
