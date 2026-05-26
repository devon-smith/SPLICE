"""Sample balanced (y=0, y=1) cuts from the MovieNet test split as a reference
channel for the AI-gen eval harness.

The sampled cuts are written in the aigen cut-index format so eval_aigen.py
can score them alongside (or instead of) real AI-gen pairs. This gives you:
  - a properly two-class eval set to validate ranking metrics before AI-gen
    y=1 clips are sourced
  - a stable reference: if v2 improves over v1.5 on movienet_test here, it
    should also improve on the actual AI-gen data

Output is a new Parquet file (does NOT modify any existing aigen cuts.parquet).
Pass it directly to eval_aigen.py via --aigen_index.

Example:
  python scripts/eval/build_movienet_aigen_ref.py \\
      --cut_index /mnt/disks/splice-data/outputs/cut_index/cuts.parquet \\
      --out /mnt/disks/splice-data/outputs/aigen_eval/movienet_ref_cuts.parquet \\
      --n_each 150
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.movienet import CUT_INDEX_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_movienet_aigen_ref")

DEFAULT_CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--out", required=True, help="output path for the sampled Parquet file")
    ap.add_argument("--n_each", type=int, default=150,
                    help="cuts to sample per class (y=0 and y=1); capped by availability")
    ap.add_argument("--seed", type=int, default=231)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.cut_index)
    test = df[df["split"] == "test"].reset_index(drop=True)
    log.info("MovieNet test: %d cuts, %.2f%% positive", len(test), 100 * test["y_inconsistent"].mean())

    pos = test[test["y_inconsistent"] == 1].reset_index(drop=True)
    neg = test[test["y_inconsistent"] == 0].reset_index(drop=True)
    n_pos = min(args.n_each, len(pos))
    n_neg = min(args.n_each, len(neg))

    sampled = pd.concat([
        pos.iloc[rng.choice(len(pos), n_pos, replace=False)],
        neg.iloc[rng.choice(len(neg), n_neg, replace=False)],
    ], ignore_index=True).copy()

    sampled["source"] = "movienet_test"
    sampled["notes"] = ""
    sampled["shot_type"] = sampled["y_inconsistent"].map({0: "within-scene", 1: "cross-scene"})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(out_path, index=False)

    pos_count = int((sampled["y_inconsistent"] == 1).sum())
    neg_count = int((sampled["y_inconsistent"] == 0).sum())
    log.info("wrote %d cuts (%d y=1, %d y=0) -> %s", len(sampled), pos_count, neg_count, out_path)
    log.info("positive rate: %.1f%%", 100 * pos_count / len(sampled))
    print(f"MovieNet reference set: {neg_count} within-scene + {pos_count} cross-scene cuts")
    print(f"Pass to eval_aigen.py with:  --aigen_index {out_path}")


if __name__ == "__main__":
    main()
