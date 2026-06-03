"""Build an eval index parquet for a small AI-gen qualitative pilot.

For each AI-gen clip in --aigen_frames (one subdir per clip, containing img0.jpg,
img1.jpg, img2.jpg), the clip becomes the RIGHT shot in a y=1 pair. The LEFT shot
is sampled at random from MovieNet test y=0 pairs.

We also include --n_real MovieNet test y=0 pairs so the scorer has a two-class set.

Output parquet columns match eval_aigen.py expectations:
  left_img0_path, left_img1_path, left_img2_path,
  right_img0_path, right_img1_path, right_img2_path,
  y_inconsistent, source

Usage:
  python scripts/eval/build_aigen_index.py \\
      --aigen_frames /mnt/disks/splice-data/aigen_frames/ \\
      --n_real 100 \\
      --out /mnt/disks/splice-data/outputs/aigen_eval/pilot_index.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

MOVIENET_CUTS = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
LEFT_KF = ("left_img0_path", "left_img1_path", "left_img2_path")
RIGHT_KF = ("right_img0_path", "right_img1_path", "right_img2_path")


def _clip_rows(clip_dir: Path, left_paths: tuple[str, str, str], source: str) -> dict:
    frames = sorted(clip_dir.glob("img*.jpg"))
    if len(frames) < 3:
        raise ValueError(f"{clip_dir}: need at least 3 img*.jpg frames, found {len(frames)}")
    return {
        "left_img0_path": left_paths[0],
        "left_img1_path": left_paths[1],
        "left_img2_path": left_paths[2],
        "right_img0_path": str(frames[0]),
        "right_img1_path": str(frames[1]),
        "right_img2_path": str(frames[2]),
        "y_inconsistent": 1,
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aigen_frames", type=Path, required=True,
                    help="directory of clip subdirs, each with img0.jpg img1.jpg img2.jpg")
    ap.add_argument("--n_real", type=int, default=100,
                    help="number of MovieNet test y=0 pairs to include as negatives")
    ap.add_argument("--out", type=Path,
                    default=Path("/mnt/disks/splice-data/outputs/aigen_eval/pilot_index.parquet"))
    ap.add_argument("--source", default="aigen_pilot",
                    help="source label for the AI-gen rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    cuts = pd.read_parquet(MOVIENET_CUTS)
    test_neg = cuts[(cuts["split"] == "test") & (cuts["y_inconsistent"] == 0)].copy()
    print(f"MovieNet test y=0 pairs available: {len(test_neg)}")

    clip_dirs = sorted(d for d in args.aigen_frames.iterdir() if d.is_dir())
    if not clip_dirs:
        raise SystemExit(f"no clip subdirectories found in {args.aigen_frames}")
    print(f"AI-gen clips found: {len(clip_dirs)}")

    # Sample left shots for y=1 pairs (one per AI-gen clip, no replacement)
    left_sample = test_neg.sample(n=min(len(clip_dirs), len(test_neg)),
                                  random_state=int(rng.integers(1 << 31)))

    aigen_rows = []
    for clip_dir, (_, left_row) in zip(clip_dirs, left_sample.iterrows()):
        left_paths = (
            left_row["left_img0_path"],
            left_row["left_img1_path"],
            left_row["left_img2_path"],
        )
        try:
            aigen_rows.append(_clip_rows(clip_dir, left_paths, args.source))
        except ValueError as e:
            print(f"  skipping {clip_dir.name}: {e}")

    # Real negative pairs
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
