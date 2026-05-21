"""Block 3b: build an AI-generated-video cut index from raw clip pairs.

Input layout::

    <clips_root>/
      labels.csv               pair_id,source,y_inconsistent,notes
      <source>/pair_<id>_left.mp4
      <source>/pair_<id>_right.mp4

For each labelled pair this extracts three uniform keyframes from each clip
(img0/1/2 = 0/50/100% of duration, matching MovieNet's 3-keyframes-per-shot
convention; boundary mode then uses left img2 + right img0) and writes a
cut-index Parquet with the MovieNet schema, so the existing embedding /
pair-feature / scoring scripts run on it unchanged.

Example:
  python scripts/eval/build_aigen_eval.py \\
      --clips_root /mnt/disks/splice-data/datasets/aigen/ \\
      --out /mnt/disks/splice-data/outputs/aigen_eval/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.movienet import CUT_INDEX_COLUMNS  # noqa: E402
from src.data.video_frames import extract_n_frames  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_aigen_eval")

MAX_SIDE = (1280, 720)  # 720p cap, aspect preserved
REQUIRED_COLS = {"pair_id", "source", "y_inconsistent"}


def _save_keyframes(clip: Path, dst_dir: Path, side: str) -> list[str]:
    """Extract + save 3 uniform keyframes of one clip; return their paths."""
    paths = [dst_dir / f"{side}_img{i}.jpg" for i in range(3)]
    if all(p.exists() for p in paths):  # idempotent
        return [str(p) for p in paths]
    dst_dir.mkdir(parents=True, exist_ok=True)
    frames = extract_n_frames(clip, n=3)
    for img, p in zip(frames, paths):
        img.thumbnail(MAX_SIDE)
        img.convert("RGB").save(p, quality=95)
    return [str(p) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clips_root = Path(args.clips_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = out_dir / "keyframes"

    labels_path = clips_root / "labels.csv"
    if not labels_path.exists():
        raise SystemExit(f"labels.csv not found at {labels_path}")
    labels = pd.read_csv(labels_path, dtype={"pair_id": str})
    missing_cols = REQUIRED_COLS - set(labels.columns)
    if missing_cols:
        raise SystemExit(f"labels.csv missing required columns: {sorted(missing_cols)}")
    if not labels["y_inconsistent"].isin([0, 1]).all():
        raise SystemExit("labels.csv: y_inconsistent must be 0 or 1")
    log.info(
        "labels.csv: %d pairs across sources %s", len(labels), sorted(labels["source"].unique())
    )

    rows: list[dict] = []
    failed: dict[str, str] = {}
    shot_idx_by_source: dict[str, int] = {}

    for _, lab in labels.iterrows():
        pid, source = lab["pair_id"], lab["source"]
        left = clips_root / source / f"pair_{pid}_left.mp4"
        right = clips_root / source / f"pair_{pid}_right.mp4"
        if not left.exists() or not right.exists():
            which = "left" if not left.exists() else "right"
            failed[pid] = f"{which} clip missing"
            log.warning("pair %s: %s clip missing", pid, which)
            continue
        try:
            pair_dir = kf_dir / str(pid)
            left_paths = _save_keyframes(left, pair_dir, "left")
            right_paths = _save_keyframes(right, pair_dir, "right")
        except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
            failed[pid] = f"frame extraction failed: {exc}"
            log.warning("pair %s: extraction failed: %s", pid, exc)
            continue

        idx = shot_idx_by_source.get(source, 0)
        shot_idx_by_source[source] = idx + 1
        y = int(lab["y_inconsistent"])
        rows.append(
            {
                "movie_id": f"aigen_{source}",
                "shot_left_idx": idx,
                "shot_right_idx": idx + 1,
                "scene_left_id": 0,
                "scene_right_id": 1 if y else 0,
                "y_inconsistent": y,
                "left_img0_path": left_paths[0],
                "left_img1_path": left_paths[1],
                "left_img2_path": left_paths[2],
                "right_img0_path": right_paths[0],
                "right_img1_path": right_paths[1],
                "right_img2_path": right_paths[2],
                "split": "test",
                "source": source,
                "notes": lab.get("notes", ""),
            }
        )

    df = pd.DataFrame(rows, columns=CUT_INDEX_COLUMNS + ["source", "notes"])
    df.to_parquet(out_dir / "cuts.parquet", index=False)

    summary = {
        "total_pairs_in_labels": int(len(labels)),
        "successfully_processed": int(len(df)),
        "failed_pairs": sorted(failed),
        "failure_reasons": failed,
        "counts_by_source": df["source"].value_counts().to_dict(),
        "counts_by_label": {str(k): int(v) for k, v in df["y_inconsistent"].value_counts().items()},
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    if failed:
        (out_dir / "failed_pairs.txt").write_text(
            "\n".join(f"{p}\t{r}" for p, r in sorted(failed.items())) + "\n"
        )

    log.info("processed %d/%d pairs -> %s", len(df), len(labels), out_dir / "cuts.parquet")
    if failed:
        log.warning("%d pairs failed (see failed_pairs.txt)", len(failed))


if __name__ == "__main__":
    main()
