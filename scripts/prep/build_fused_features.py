# build the fused DINOv2+CLIP boundary pair feature: [dino 2305 | clip 2305] = 4610-d
# v2 fusion scaffold -- keeps only cuts whose 4 boundary frames are in both caches
# usage: python scripts/prep/build_fused_features.py --cut_index ... --out ...

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.movienet import keyframe_key
from src.data.pairs import (FUSED_PAIR_FEATURE_DIM, PAIR_FEATURE_DIM,
                             build_fused_pair_features, load_embeddings)

DINO_DIR = "/mnt/disks/splice-data/embeddings/dinov2_base"
CLIP_DIR = "/mnt/disks/splice-data/embeddings/clip_vitl14"
LEFT_COL, RIGHT_COL = "left_img2_path", "right_img0_path"


def rows(df, col, key2row):
    return df[col].map(lambda p: key2row.get(keyframe_key(p), -1)).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut_index", required=True)
    ap.add_argument("--dino_embeddings", default=DINO_DIR)
    ap.add_argument("--clip_embeddings", default=CLIP_DIR)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.cut_index)
    print(f"cut index: {len(df)} cuts")
    dino_emb, dino_k2r = load_embeddings(args.dino_embeddings)
    clip_emb, clip_k2r = load_embeddings(args.clip_embeddings)
    print(f"DINOv2: {dino_emb.shape[0]} x {dino_emb.shape[1]} | "
          f"CLIP: {clip_emb.shape[0]} x {clip_emb.shape[1]}")

    dl, dr = rows(df, LEFT_COL, dino_k2r), rows(df, RIGHT_COL, dino_k2r)
    cl, cr = rows(df, LEFT_COL, clip_k2r), rows(df, RIGHT_COL, clip_k2r)
    valid = (dl >= 0) & (dr >= 0) & (cl >= 0) & (cr >= 0)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f"dropping {n_drop}/{len(df)} cuts missing in one of the caches")
    df = df[valid].reset_index(drop=True)
    dl, dr, cl, cr = dl[valid], dr[valid], cl[valid], cr[valid]
    if len(df) == 0:
        raise SystemExit("no cuts have all 4 keyframes in both caches -- check the CLIP cache")

    features = build_fused_pair_features(dino_emb[dl], dino_emb[dr], clip_emb[cl], clip_emb[cr])
    assert features.shape == (len(df), FUSED_PAIR_FEATURE_DIM)
    assert not np.isnan(features).any()

    # sanity check: mean cosine per half, split by class. y=0 (within-scene) should
    # have higher cos than y=1 (cross-scene) for both DINOv2 and CLIP halves.
    dino_cos = features[:, PAIR_FEATURE_DIM - 1]
    clip_cos = features[:, FUSED_PAIR_FEATURE_DIM - 1]
    y = df["y_inconsistent"].to_numpy().astype(int)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    print(f"\nfused-feature cosine sanity (n_y0={n0}, n_y1={n1})")
    for name, cos in (("dino", dino_cos), ("clip", clip_cos)):
        m0 = float(cos[y == 0].mean()) if n0 else float("nan")
        m1 = float(cos[y == 1].mean()) if n1 else float("nan")
        verdict = "OK" if m0 > m1 else "check"
        print(f"  {name}: within-scene(y0) {m0:.3f}  cross-scene(y1) {m1:.3f}   {verdict}")

    meta = pd.DataFrame({
        "cut_id": df["movie_id"] + "_" + df["shot_left_idx"].astype(str).str.zfill(4),
        "y_inconsistent": y,
        "split": df["split"].to_numpy(),
        "movie_id": df["movie_id"].to_numpy(),
    })
    np.save(out_dir / "features.npy", features)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    (out_dir / "metadata.json").write_text(json.dumps({
        "feature": "fused_dino_clip_boundary",
        "feature_dim": FUSED_PAIR_FEATURE_DIM,
        "layout": "[dino 0:2305 | clip 2305:4610]",
        "n_cuts": int(len(df)),
        "n_dropped_missing_embedding": n_drop,
    }, indent=2))
    print(f"wrote {features.shape[0]} x {features.shape[1]} fused features -> {out_dir}")


if __name__ == "__main__":
    main()
