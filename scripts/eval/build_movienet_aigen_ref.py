# sample balanced (y=0, y=1) cuts from MovieNet test as a reference channel
# for the aigen eval harness -- a stable two-class set for sanity comparisons
# usage: python scripts/eval/build_movienet_aigen_ref.py --out PATH

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut_index", default=CUT_INDEX)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_each", type=int, default=150,
                    help="cuts per class (y=0 and y=1); capped by availability")
    ap.add_argument("--seed", type=int, default=231)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.cut_index)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"MovieNet test: {len(test)} cuts, {100 * test['y_inconsistent'].mean():.2f}% positive")

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

    n_pos_s = int((sampled["y_inconsistent"] == 1).sum())
    n_neg_s = int((sampled["y_inconsistent"] == 0).sum())
    print(f"wrote {len(sampled)} cuts ({n_pos_s} y=1, {n_neg_s} y=0) -> {out_path}")
    print(f"pass to eval_aigen.py with:  --aigen_index {out_path}")


if __name__ == "__main__":
    main()
