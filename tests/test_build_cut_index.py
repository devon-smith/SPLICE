"""Unit tests for cut-index label derivation (src.data.movienet)."""

from pathlib import Path

import pytest

from src.data.movienet import CUT_INDEX_COLUMNS, Shot, cut_rows_for_movie, derive_label

FRAMES = Path("/data/240P_frames")


def _shot(idx: int, scene: int, bl: int | None) -> Shot:
    return Shot(shot_idx=idx, scene_id=scene, boundary_label=bl)


def test_derive_label_json_uses_scene_ids():
    same = (_shot(0, 3, 0), _shot(1, 3, 0))
    diff = (_shot(0, 3, 0), _shot(1, 4, 0))
    assert derive_label(*same, "json") == 0
    assert derive_label(*diff, "json") == 1


def test_derive_label_label318_uses_left_boundary_label():
    # boundary_label of the LEFT shot wins, even when scene ids agree.
    assert derive_label(_shot(0, 3, 1), _shot(1, 3, 0), "label318") == 1
    assert derive_label(_shot(0, 3, 0), _shot(1, 9, 0), "label318") == 0


def test_derive_label_label318_missing_field_raises():
    with pytest.raises(ValueError):
        derive_label(_shot(0, 3, None), _shot(1, 4, 0), "label318")


def test_derive_label_auto_prefers_boundary_label_else_scene():
    # boundary_label present -> used (here it disagrees with scene ids).
    assert derive_label(_shot(0, 3, 1), _shot(1, 3, 0), "auto") == 1
    # boundary_label absent -> fall back to scene-id disagreement.
    assert derive_label(_shot(0, 3, None), _shot(1, 4, 0), "auto") == 1
    assert derive_label(_shot(0, 3, None), _shot(1, 3, 0), "auto") == 0


def test_derive_label_unknown_source_raises():
    with pytest.raises(ValueError):
        derive_label(_shot(0, 0, 0), _shot(1, 0, 0), "bogus")


def test_cut_rows_for_movie_shape_and_labels():
    # 4 shots, scenes [0,0,0,1]; boundary after shot 2.
    shots = [_shot(0, 0, 0), _shot(1, 0, 0), _shot(2, 0, 1), _shot(3, 1, 0)]
    rows = cut_rows_for_movie("tt1", "val", shots, FRAMES, label_source="auto")
    assert len(rows) == 3  # n-1 cuts
    assert [r["y_inconsistent"] for r in rows] == [0, 0, 1]
    assert all(set(r) == set(CUT_INDEX_COLUMNS) for r in rows)
    assert all(r["split"] == "val" for r in rows)
    assert [r["shot_left_idx"] for r in rows] == [0, 1, 2]
    assert [r["shot_right_idx"] for r in rows] == [1, 2, 3]


def test_cut_rows_for_movie_keyframe_paths():
    shots = [_shot(5, 0, 0), _shot(6, 1, 0)]
    row = cut_rows_for_movie("tt9", "test", shots, FRAMES)[0]
    assert row["left_img2_path"] == "/data/240P_frames/tt9/shot_0005_img_2.jpg"
    assert row["right_img0_path"] == "/data/240P_frames/tt9/shot_0006_img_0.jpg"


def test_cut_rows_single_shot_movie_has_no_cuts():
    assert cut_rows_for_movie("tt0", "train", [_shot(0, 0, 0)], FRAMES) == []
