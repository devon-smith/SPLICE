# build an AI-generated-video cut index from raw clip pairs


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.movienet import CUT_INDEX_COLUMNS
from src.data.video_frames import extract_n_frames

MAX_SIDE = (1280, 720)  # 720p cap, aspect preserved
LABEL_COLS = ("intended_label", "y_inconsistent")


# extract + save 3 uniform keyframes of one clip; idempotent
def save_keyframes(clip, dst_dir, side):
    paths = [dst_dir / f"{side}_img{i}.jpg" for i in range(3)]
    if all(p.exists() for p in paths):
        return [str(p) for p in paths]
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img, p in zip(extract_n_frames(clip, n=3), paths):
        img.thumbnail(MAX_SIDE)
        img.convert("RGB").save(p, quality=95)
    return [str(p) for p in paths]


def main():
    ap = argparse.ArgumentParser()
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

    # accept either intended_label or y_inconsistent column
    label_col = next((c for c in LABEL_COLS if c in labels.columns), None)
    if label_col is None:
        raise SystemExit(f"labels.csv needs one of {list(LABEL_COLS)}")
    labels = labels.rename(columns={label_col: "y_inconsistent"})

    # drop pairs the generator flagged unusable
    if "quality_check" in labels.columns:
        discard = labels["quality_check"].astype(str).str.strip().str.lower() == "discard"
        if discard.any():
            print(f"skipping {discard.sum()} pairs marked quality_check=discard")
            labels = labels[~discard].reset_index(drop=True)

    labels["y_inconsistent"] = pd.to_numeric(labels["y_inconsistent"], errors="coerce")
    if not labels["y_inconsistent"].isin([0, 1]).all():
        raise SystemExit("y_inconsistent must be 0 or 1 for every kept row")
    labels["y_inconsistent"] = labels["y_inconsistent"].astype(int)
    print(f"labels.csv: {len(labels)} pairs across sources {sorted(labels['source'].unique())}")

    rows, failed = [], {}
    shot_idx_by_source = {}

    for _, lab in labels.iterrows():
        pid, source = lab["pair_id"], lab["source"]
        left = clips_root / source / f"pair_{pid}_left.mp4"
        right = clips_root / source / f"pair_{pid}_right.mp4"
        if not left.exists() or not right.exists():
            which = "left" if not left.exists() else "right"
            failed[pid] = f"{which} clip missing"
            print(f"pair {pid}: {which} clip missing")
            continue
        try:
            pair_dir = kf_dir / str(pid)
            left_paths = save_keyframes(left, pair_dir, "left")
            right_paths = save_keyframes(right, pair_dir, "right")
        except Exception as exc:
            failed[pid] = f"frame extraction failed: {exc}"
            print(f"pair {pid}: extraction failed: {exc}")
            continue

        idx = shot_idx_by_source.get(source, 0)
        shot_idx_by_source[source] = idx + 1
        y = int(lab["y_inconsistent"])
        rows.append({
            "movie_id": f"aigen_{source}",
            "shot_left_idx": idx, "shot_right_idx": idx + 1,
            "scene_left_id": 0, "scene_right_id": 1 if y else 0,
            "y_inconsistent": y,
            "left_img0_path": left_paths[0], "left_img1_path": left_paths[1],
            "left_img2_path": left_paths[2],
            "right_img0_path": right_paths[0], "right_img1_path": right_paths[1],
            "right_img2_path": right_paths[2],
            "split": "test", "source": source,
            "notes": lab.get("notes", ""), "shot_type": lab.get("shot_type", ""),
        })

    df = pd.DataFrame(rows, columns=CUT_INDEX_COLUMNS + ["source", "notes", "shot_type"])
    df.to_parquet(out_dir / "cuts.parquet", index=False)

    summary = {
        "total_pairs_in_labels": int(len(labels)),
        "successfully_processed": int(len(df)),
        "failed_pairs": sorted(failed),
        "failure_reasons": failed,
        "counts_by_source": df["source"].value_counts().to_dict(),
        "counts_by_label": {str(k): int(v) for k, v in df["y_inconsistent"].value_counts().items()},
        "counts_by_shot_type": {str(k): int(v) for k, v in df["shot_type"].value_counts().items()},
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    if failed:
        (out_dir / "failed_pairs.txt").write_text(
            "\n".join(f"{p}\t{r}" for p, r in sorted(failed.items())) + "\n"
        )

    print(f"processed {len(df)}/{len(labels)} pairs -> {out_dir / 'cuts.parquet'}")
    if failed:
        print(f"{len(failed)} pairs failed (see failed_pairs.txt)")


if __name__ == "__main__":
    main()
