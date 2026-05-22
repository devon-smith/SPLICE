"""Build the fused DINOv2+CLIP boundary pair feature for a cut index.

v2 fusion scaffold. Reads the cut index, the cached DINOv2 embeddings and the
cached CLIP embeddings, and emits the 4610-d fused boundary feature per cut:

    [ DINOv2 2305-d pair feature | CLIP 2305-d pair feature ]

Output (mirrors build_pair_features.py):
  <out>/features.npy   float32 (n_cuts, 4610)
  <out>/meta.parquet   cut_id, y_inconsistent, split, movie_id
  <out>/metadata.json  provenance + a cosine-similarity sanity summary

This trains nothing. It is the infrastructure a v2 fusion head would consume.
A cut is kept only if all four boundary keyframes are present in *both* caches,
so run it on a cut index whose keyframes the CLIP cache actually covers.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.movienet import keyframe_key  # noqa: E402
from src.data.pairs import (  # noqa: E402
    FUSED_PAIR_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    build_fused_pair_features,
    load_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_fused_features")

DEFAULT_DINO = "/mnt/disks/splice-data/embeddings/dinov2_base"
DEFAULT_CLIP = "/mnt/disks/splice-data/embeddings/clip_vitl14"
LEFT_COL, RIGHT_COL = "left_img2_path", "right_img0_path"  # boundary frames


def _rows(df: pd.DataFrame, col: str, key2row: dict[str, int]) -> np.ndarray:
    """Map a keyframe-path column to embedding row indices; -1 where missing."""
    return df[col].map(lambda p: key2row.get(keyframe_key(p), -1)).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", required=True)
    ap.add_argument("--dino_embeddings", default=DEFAULT_DINO)
    ap.add_argument("--clip_embeddings", default=DEFAULT_CLIP)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.cut_index)
    log.info("cut index: %d cuts", len(df))
    dino_emb, dino_k2r = load_embeddings(args.dino_embeddings)
    clip_emb, clip_k2r = load_embeddings(args.clip_embeddings)
    log.info(
        "DINOv2 cache: %d x %d | CLIP cache: %d x %d",
        *dino_emb.shape,
        *clip_emb.shape,
    )

    dl = _rows(df, LEFT_COL, dino_k2r)
    dr = _rows(df, RIGHT_COL, dino_k2r)
    cl = _rows(df, LEFT_COL, clip_k2r)
    cr = _rows(df, RIGHT_COL, clip_k2r)
    valid = (dl >= 0) & (dr >= 0) & (cl >= 0) & (cr >= 0)
    n_drop = int((~valid).sum())
    if n_drop:
        log.warning(
            "dropping %d/%d cuts missing an embedding in one of the caches", n_drop, len(df)
        )
    df = df[valid].reset_index(drop=True)
    dl, dr, cl, cr = dl[valid], dr[valid], cl[valid], cr[valid]
    if len(df) == 0:
        raise SystemExit("no cuts have all four keyframes in both caches -- check the CLIP cache")

    features = build_fused_pair_features(dino_emb[dl], dino_emb[dr], clip_emb[cl], clip_emb[cr])
    assert features.shape == (len(df), FUSED_PAIR_FEATURE_DIM), features.shape
    assert not np.isnan(features).any(), "NaN in fused features"

    # sanity: the cosine-similarity component of each half, split by class.
    dino_cos = features[:, PAIR_FEATURE_DIM - 1]
    clip_cos = features[:, FUSED_PAIR_FEATURE_DIM - 1]
    y = df["y_inconsistent"].to_numpy().astype(int)
    sanity = _sanity(dino_cos, clip_cos, y)

    meta = pd.DataFrame(
        {
            "cut_id": df["movie_id"] + "_" + df["shot_left_idx"].astype(str).str.zfill(4),
            "y_inconsistent": y,
            "split": df["split"].to_numpy(),
            "movie_id": df["movie_id"].to_numpy(),
        }
    )
    np.save(out_dir / "features.npy", features)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "feature": "fused_dino_clip_boundary",
                "feature_dim": FUSED_PAIR_FEATURE_DIM,
                "layout": "[dino 0:2305 | clip 2305:4610]",
                "n_cuts": int(len(df)),
                "n_dropped_missing_embedding": n_drop,
                "cut_index": str(args.cut_index),
                "dino_embeddings": str(args.dino_embeddings),
                "clip_embeddings": str(args.clip_embeddings),
                "cosine_sanity": sanity,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    log.info("wrote %d x %d fused features -> %s", *features.shape, out_dir)
    _print_sanity(sanity)


def _sanity(dino_cos: np.ndarray, clip_cos: np.ndarray, y: np.ndarray) -> dict:
    """Mean cosine similarity per half, split by class -- y=0 should exceed y=1."""
    out = {"n_y0": int((y == 0).sum()), "n_y1": int((y == 1).sum())}
    for name, cos in (("dino", dino_cos), ("clip", clip_cos)):
        m0 = float(cos[y == 0].mean()) if (y == 0).any() else float("nan")
        m1 = float(cos[y == 1].mean()) if (y == 1).any() else float("nan")
        out[name] = {
            "cos_y0_within_scene": m0,
            "cos_y1_cross_scene": m1,
            "separates_correctly": bool(m0 > m1) if (y == 0).any() and (y == 1).any() else None,
        }
    return out


def _print_sanity(s: dict) -> None:
    print(f"\n=== fused-feature cosine sanity (n_y0={s['n_y0']}, n_y1={s['n_y1']}) ===")
    for name in ("dino", "clip"):
        h = s[name]
        verdict = (
            "OK (within-scene more similar)"
            if h["separates_correctly"]
            else "check" if h["separates_correctly"] is False else "single class"
        )
        print(
            f"  {name:<5} cos  within-scene(y0) {h['cos_y0_within_scene']:.3f}  "
            f"cross-scene(y1) {h['cos_y1_cross_scene']:.3f}   {verdict}"
        )


if __name__ == "__main__":
    main()
