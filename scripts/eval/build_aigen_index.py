# build an eval index for a small AI-gen pilot
# each aigen clip dir (with img0/1/2.jpg) becomes the RIGHT shot of a y=1 pair,
# paired with a random MovieNet test y=0 LEFT shot. also adds --n_real real y=0 pairs.
# usage: python scripts/eval/build_aigen_index.py --aigen_frames DIR --out PATH

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MOVIENET_CUTS = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
LEFT_KF = ("left_img0_path", "left_img1_path", "left_img2_path")
RIGHT_KF = ("right_img0_path", "right_img1_path", "right_img2_path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aigen_frames", type=Path, required=True)
    ap.add_argument("--n_real", type=int, default=100, help="real MovieNet y=0 pairs to include")
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/disks/splice-data/outputs/aigen_eval/pilot_index.parquet"))
    ap.add_argument("--source", default="aigen_pilot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cuts = pd.read_parquet(MOVIENET_CUTS)
    test_neg = cuts[(cuts["split"] == "test") & (cuts["y_inconsistent"] == 0)].copy()
    print(f"MovieNet test y=0 pairs: {len(test_neg)}")

    clip_dirs = sorted(d for d in args.aigen_frames.iterdir() if d.is_dir())
    if not clip_dirs:
        raise SystemExit(f"no clip subdirectories in {args.aigen_frames}")
    print(f"AI-gen clips: {len(clip_dirs)}")

    # one random left shot per AI-gen clip (no replacement)
    left_sample = test_neg.sample(n=min(len(clip_dirs), len(test_neg)),
                                  random_state=int(rng.integers(1 << 31)))

    aigen_rows = []
    for clip_dir, (_, left_row) in zip(clip_dirs, left_sample.iterrows()):
        frames = sorted(clip_dir.glob("img*.jpg"))
        if len(frames) < 3:
            print(f"  skipping {clip_dir.name}: only {len(frames)} img*.jpg frames")
            continue
        aigen_rows.append({
            "left_img0_path": left_row["left_img0_path"],
            "left_img1_path": left_row["left_img1_path"],
            "left_img2_path": left_row["left_img2_path"],
            "right_img0_path": str(frames[0]),
            "right_img1_path": str(frames[1]),
            "right_img2_path": str(frames[2]),
            "y_inconsistent": 1,
            "source": args.source,
        })

    # real negative pairs
    real_sample = test_neg.sample(n=min(args.n_real, len(test_neg)),
                                   random_state=int(rng.integers(1 << 31)))
    real_rows = real_sample[list(LEFT_KF + RIGHT_KF)].copy()
    real_rows["y_inconsistent"] = 0
    real_rows["source"] = "movienet_test"

    out_df = pd.concat([pd.DataFrame(aigen_rows), real_rows], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    pos = int(out_df["y_inconsistent"].sum())
    neg = len(out_df) - pos
    print(f"wrote {args.out}  ({pos} AI-gen y=1, {neg} real y=0)")


if __name__ == "__main__":
    main()
