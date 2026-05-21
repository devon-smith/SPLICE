"""Build the 2305-d pair feature for every cut in the index.

Reads the cut index and the cached DINOv2 embeddings, emits per ``--mode``:
  <out>/features.npy     float32  (n_cuts, 2305)
  <out>/meta.parquet     cut_id, y_inconsistent, split, movie_id  (row-aligned)
  <out>/metadata.json    provenance

Example:
  python scripts/prep/build_pair_features.py \\
      --cut_index /mnt/disks/splice-data/outputs/cut_index/cuts.parquet \\
      --embeddings /mnt/disks/splice-data/embeddings/dinov2_base \\
      --mode boundary
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
    PAIR_FEATURE_DIM,
    build_pair_features_batch,
    load_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_pair_features")

DEFAULT_CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
DEFAULT_EMB = "/mnt/disks/splice-data/embeddings/dinov2_base"
DEFAULT_OUT_TPL = "/mnt/disks/splice-data/pairs/dino_v0_{mode}"

MODE_COLS = {
    "boundary": (["left_img2_path"], ["right_img0_path"]),
    "mean_pool_3": (
        ["left_img0_path", "left_img1_path", "left_img2_path"],
        ["right_img0_path", "right_img1_path", "right_img2_path"],
    ),
}


def _rows_for_cols(df: pd.DataFrame, cols: list[str], key2row: dict[str, int]) -> np.ndarray:
    """Map path columns -> embedding row indices, ``(n_cuts, len(cols))``; -1 if missing."""
    return np.stack(
        [df[c].map(lambda p: key2row.get(keyframe_key(p), -1)).to_numpy() for c in cols],
        axis=1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--embeddings", default=DEFAULT_EMB)
    ap.add_argument("--mode", choices=list(MODE_COLS), default="boundary")
    ap.add_argument("--out", default=None, help="default: /mnt/.../pairs/dino_v0_<mode>")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(DEFAULT_OUT_TPL.format(mode=args.mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.cut_index)
    log.info("cut index: %d cuts", len(df))
    emb, key2row = load_embeddings(args.embeddings)
    log.info("embeddings: %d keyframes x %d dims", *emb.shape)

    left_cols, right_cols = MODE_COLS[args.mode]
    left_rows = _rows_for_cols(df, left_cols, key2row)
    right_rows = _rows_for_cols(df, right_cols, key2row)

    valid = (left_rows >= 0).all(axis=1) & (right_rows >= 0).all(axis=1)
    n_drop = int((~valid).sum())
    if n_drop:
        log.warning(
            "dropping %d cuts with missing embeddings (%.2f%%)", n_drop, 100 * n_drop / len(df)
        )
    df = df[valid].reset_index(drop=True)
    left_rows, right_rows = left_rows[valid], right_rows[valid]

    # mean over keyframes (1 frame for boundary mode, 3 for mean_pool_3)
    e_left = emb[left_rows].mean(axis=1)
    e_right = emb[right_rows].mean(axis=1)
    features = build_pair_features_batch(e_left, e_right)
    assert features.shape == (len(df), PAIR_FEATURE_DIM), features.shape
    assert not np.isnan(features).any(), "NaN in pair features"

    meta = pd.DataFrame(
        {
            "cut_id": df["movie_id"] + "_" + df["shot_left_idx"].astype(str).str.zfill(4),
            "y_inconsistent": df["y_inconsistent"].to_numpy(),
            "split": df["split"].to_numpy(),
            "movie_id": df["movie_id"].to_numpy(),
        }
    )
    np.save(out_dir / "features.npy", features)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "n_cuts": int(len(df)),
                "feature_dim": PAIR_FEATURE_DIM,
                "n_dropped_missing_embedding": n_drop,
                "cut_index": str(args.cut_index),
                "embeddings": str(args.embeddings),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    log.info("wrote %d x %d features -> %s", *features.shape, out_dir)
    log.info("positive rate: %.2f%%", 100 * meta["y_inconsistent"].mean())


if __name__ == "__main__":
    main()
