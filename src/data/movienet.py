# MovieNet (BaSSL distribution) annotation parsing.
# the BaSSL annotation set is three NDJSON files (anno.{train,val,test}.ndjson)
# where each line is one shot record. boundary_label is on the LEFT shot of a cut:
# 1 iff the cut crosses a scene boundary. a movie's split = which file it appears in.

import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")

# downstream code depends on this column order
CUT_INDEX_COLUMNS = [
    "movie_id", "shot_left_idx", "shot_right_idx",
    "scene_left_id", "scene_right_id", "y_inconsistent",
    "left_img0_path", "left_img1_path", "left_img2_path",
    "right_img0_path", "right_img1_path", "right_img2_path",
    "split",
]


@dataclass
class Shot:
    shot_idx: int
    scene_id: int
    # 1 iff the cut after this shot is a scene boundary; None if absent
    boundary_label: int | None


def keyframe_path(frames_dir, movie_id, shot_idx, img):
    return frames_dir / movie_id / f"shot_{shot_idx:04d}_img_{img}.jpg"


# stable <movie_id>/<filename> key used by the embedding cache
def keyframe_key(path):
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


# parse the 3 NDJSON files into {movie_id: (split, shots sorted by idx)}
def load_shots_by_movie(anno_dir):
    by_movie = {}
    for split in SPLITS:
        path = anno_dir / f"anno.{split}.ndjson"
        if not path.exists():
            continue
        per_movie = {}
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                bl = rec.get("boundary_label")
                shot = Shot(
                    shot_idx=int(rec["shot_id"]),
                    scene_id=int(rec["invideo_scene_id"]),
                    boundary_label=None if bl is None else int(bl),
                )
                per_movie.setdefault(rec["video_id"], []).append(shot)
        for movie_id, shots in per_movie.items():
            shots.sort(key=lambda s: s.shot_idx)
            by_movie[movie_id] = (split, shots)
    return by_movie


# per-cut label: from BaSSL boundary_label if present (auto/label318),
# else from scene-id disagreement (json)
def derive_label(left, right, label_source):
    y_scene = int(left.scene_id != right.scene_id)
    if label_source == "json":
        return y_scene
    if label_source == "label318":
        if left.boundary_label is None:
            raise ValueError("label_source=label318 but boundary_label is missing")
        return left.boundary_label
    if label_source == "auto":
        return y_scene if left.boundary_label is None else left.boundary_label
    raise ValueError(f"unknown label_source: {label_source}")


# build one cut-index row per adjacent shot pair in a single movie.
# cuts with boundary_label == -1 are dropped (BaSSL's "ignore" marker).
def cut_rows_for_movie(movie_id, split, shots, frames_dir, label_source="auto"):
    rows = []
    for left, right in zip(shots, shots[1:]):
        if left.boundary_label == -1:
            continue
        rows.append({
            "movie_id": movie_id,
            "shot_left_idx": left.shot_idx,
            "shot_right_idx": right.shot_idx,
            "scene_left_id": left.scene_id,
            "scene_right_id": right.scene_id,
            "y_inconsistent": derive_label(left, right, label_source),
            "left_img0_path": str(keyframe_path(frames_dir, movie_id, left.shot_idx, 0)),
            "left_img1_path": str(keyframe_path(frames_dir, movie_id, left.shot_idx, 1)),
            "left_img2_path": str(keyframe_path(frames_dir, movie_id, left.shot_idx, 2)),
            "right_img0_path": str(keyframe_path(frames_dir, movie_id, right.shot_idx, 0)),
            "right_img1_path": str(keyframe_path(frames_dir, movie_id, right.shot_idx, 1)),
            "right_img2_path": str(keyframe_path(frames_dir, movie_id, right.shot_idx, 2)),
            "split": split,
        })
    return rows
