"""MovieNet (BaSSL distribution) annotation parsing utilities.

The BaSSL annotation set is three NDJSON files -- ``anno.{train,val,test}.ndjson``
-- where every line is one *shot*::

    {"video_id": "tt0048545", "shot_id": "0042", "num_shot": 836,
     "boundary_label": 0, "invideo_scene_id": 7, ...}

``boundary_label`` is indexed on the LEFT shot of a cut: it is 1 iff the cut
between this shot and the next crosses a scene boundary (equivalently, this shot
is the last shot of its scene). It equals ``invideo_scene_id`` disagreement and
matches BaSSL's separate ``label318`` artifact. A movie's split is simply which
of the three files it appears in (train/val/test = 190/64/64 movies).
"""

import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "val", "test")

#: Column order of the cut index (downstream code depends on this schema).
CUT_INDEX_COLUMNS = [
    "movie_id",
    "shot_left_idx",
    "shot_right_idx",
    "scene_left_id",
    "scene_right_id",
    "y_inconsistent",
    "left_img0_path",
    "left_img1_path",
    "left_img2_path",
    "right_img0_path",
    "right_img1_path",
    "right_img2_path",
    "split",
]


@dataclass
class Shot:
    """One shot of a movie."""

    shot_idx: int
    scene_id: int
    #: 1 iff the cut after this shot is a scene boundary; None if field absent.
    boundary_label: int | None


def keyframe_path(frames_dir: Path, movie_id: str, shot_idx: int, img: int) -> Path:
    """Absolute path to a single keyframe (``img`` in {0, 1, 2})."""
    return frames_dir / movie_id / f"shot_{shot_idx:04d}_img_{img}.jpg"


def keyframe_key(path: str | Path) -> str:
    """Stable ``<movie_id>/<filename>`` key for a keyframe (used by the embedding cache)."""
    p = Path(path)
    return f"{p.parent.name}/{p.name}"


def load_shots_by_movie(anno_dir: Path) -> dict[str, tuple[str, list[Shot]]]:
    """Parse the three NDJSON files into ``{movie_id: (split, shots sorted by idx)}``."""
    by_movie: dict[str, tuple[str, list[Shot]]] = {}
    for split in SPLITS:
        path = anno_dir / f"anno.{split}.ndjson"
        if not path.exists():
            continue
        per_movie: dict[str, list[Shot]] = {}
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


def derive_label(left: Shot, right: Shot, label_source: str) -> int:
    """Per-cut inconsistency label.

    ``json``     -> derived from scene-id disagreement.
    ``label318`` -> the left shot's ``boundary_label`` (BaSSL's per-shot label).
    ``auto``     -> ``boundary_label`` when present, else scene-id disagreement.
    """
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


def cut_rows_for_movie(
    movie_id: str,
    split: str,
    shots: list[Shot],
    frames_dir: Path,
    label_source: str = "auto",
) -> list[dict]:
    """Build one cut-index row per adjacent shot pair in a single movie.

    Cuts whose left shot has ``boundary_label == -1`` are dropped: -1 is BaSSL's
    "ignore" marker for transitions it leaves unannotated, so they carry no
    reliable ground-truth label.
    """
    rows: list[dict] = []
    for left, right in zip(shots, shots[1:]):
        if left.boundary_label == -1:
            continue
        rows.append(
            {
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
            }
        )
    return rows
