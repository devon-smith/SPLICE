# AI-USE: This script was AI-assisted with Claude via Claude Code.
# build the labeled cut index: one row per adjacent shot-pair cut in MovieNet

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.data.movienet import CUT_INDEX_COLUMNS, cut_rows_for_movie, load_shots_by_movie

DATA_ROOT = "/mnt/disks/splice-data/datasets/movienet"
OUT_PATH = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--label_source", choices=["auto", "label318", "json"], default="auto")
    ap.add_argument("--split_file", default=None, help="optional split318.json override")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    anno_dir = data_root / "anno"
    frames_dir = data_root / "240P_frames"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"parsing annotations from {anno_dir}")
    by_movie = load_shots_by_movie(anno_dir)
    print(f"found {len(by_movie)} movies")
    if not by_movie:
        raise SystemExit(f"no annotations found under {anno_dir}")

    # optional override: split318.json maps {split: [movie_id, ...]}
    split_override = {}
    if args.split_file:
        raw = json.loads(Path(args.split_file).read_text())
        split_override = {mid: split for split, ids in raw.items() for mid in ids}

    rows = []
    failed = []
    n_dropped = 0  # cuts dropped because boundary_label == -1 (BaSSL "ignore" marker)
    for movie_id, (split, shots) in tqdm(sorted(by_movie.items()), desc="movies"):
        try:
            split = split_override.get(movie_id, split)
            movie_rows = cut_rows_for_movie(movie_id, split, shots, frames_dir, args.label_source)
            n_dropped += max(len(shots) - 1, 0) - len(movie_rows)
            rows.extend(movie_rows)
        except Exception as exc:
            print(f"failed to parse {movie_id}: {exc}")
            failed.append(movie_id)

    df = pd.DataFrame(rows, columns=CUT_INDEX_COLUMNS)
    df.to_parquet(out_path, index=False)

    if failed:
        fail_path = out_path.parent / "failed_movies.txt"
        fail_path.write_text("\n".join(failed) + "\n")
        print(f"{len(failed)} movies failed -> {fail_path}")

    pos = int(df["y_inconsistent"].sum())
    print(f"wrote {len(df)} cuts to {out_path}")
    print(f"positive (scene-boundary) cuts: {pos} ({100 * pos / max(len(df), 1):.2f}%)")
    print(f"splits: {df.groupby('split').size().to_dict()}")
    print(f"dropped {n_dropped} unlabeled cuts (boundary_label == -1)")


if __name__ == "__main__":
    main()
