"""Build the labeled cut index: one row per adjacent shot-pair cut in MovieNet.

Output is a single Parquet file with the 13-column schema in
``src.data.movienet.CUT_INDEX_COLUMNS``. Every downstream stage (embedding,
pair features, training, calibration) reads this file.

Example:
  python scripts/prep/build_cut_index.py \\
      --data_root /mnt/disks/splice-data/datasets/movienet \\
      --out /mnt/disks/splice-data/outputs/cut_index/cuts.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.movienet import (  # noqa: E402
    CUT_INDEX_COLUMNS,
    cut_rows_for_movie,
    load_shots_by_movie,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_cut_index")

DEFAULT_DATA_ROOT = "/mnt/disks/splice-data/datasets/movienet"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"


def load_split_override(split_file: Path) -> dict[str, str]:
    """Load a split318-style JSON ({split: [movie_id, ...]}) into {movie_id: split}."""
    import json

    raw = json.loads(split_file.read_text())
    return {mid: split for split, ids in raw.items() for mid in ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--anno_dir", default=None, help="default: <data_root>/anno")
    ap.add_argument("--frames_dir", default=None, help="default: <data_root>/240P_frames")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--label_source", choices=["auto", "label318", "json"], default="auto")
    ap.add_argument("--split_file", default=None, help="optional split318.json override")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    anno_dir = Path(args.anno_dir) if args.anno_dir else data_root / "anno"
    frames_dir = Path(args.frames_dir) if args.frames_dir else data_root / "240P_frames"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("parsing annotations from %s", anno_dir)
    by_movie = load_shots_by_movie(anno_dir)
    log.info("found %d movies", len(by_movie))
    if not by_movie:
        raise SystemExit(f"no annotations found under {anno_dir}")

    split_override = load_split_override(Path(args.split_file)) if args.split_file else {}

    rows: list[dict] = []
    failed: list[str] = []
    n_dropped = 0  # cuts dropped because boundary_label == -1 (BaSSL "ignore" marker)
    for movie_id, (split, shots) in tqdm(sorted(by_movie.items()), desc="movies"):
        try:
            split = split_override.get(movie_id, split)
            movie_rows = cut_rows_for_movie(movie_id, split, shots, frames_dir, args.label_source)
            n_dropped += max(len(shots) - 1, 0) - len(movie_rows)
            rows.extend(movie_rows)
        except Exception as exc:  # noqa: BLE001 - keep one bad movie from killing the run
            log.warning("failed to parse %s: %s", movie_id, exc)
            failed.append(movie_id)

    df = pd.DataFrame(rows, columns=CUT_INDEX_COLUMNS)
    df.to_parquet(out_path, index=False)

    if failed:
        fail_path = out_path.parent / "failed_movies.txt"
        fail_path.write_text("\n".join(failed) + "\n")
        log.warning("%d movies failed -> %s", len(failed), fail_path)

    pos = int(df["y_inconsistent"].sum())
    log.info("wrote %d cuts to %s", len(df), out_path)
    log.info("positive (scene-boundary) cuts: %d (%.2f%%)", pos, 100 * pos / max(len(df), 1))
    log.info("splits: %s", df.groupby("split").size().to_dict())
    log.info("dropped %d unlabeled cuts (boundary_label == -1)", n_dropped)


if __name__ == "__main__":
    main()
