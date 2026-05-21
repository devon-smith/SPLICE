"""Unit tests for src.data.video_frames."""

import cv2
import numpy as np
import pytest
from PIL import Image

from src.data.video_frames import (
    extract_boundary_frames,
    extract_n_frames,
    get_video_info,
)

FPS = 24
N_FRAMES = 120  # 5 seconds


def _make_video(path, n_frames=N_FRAMES, fps=FPS, size=(160, 120)) -> str:
    """Write a synthetic video whose frames are distinguishable by index."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), (i * 2 % 256, (255 - i) % 256, (i * 5) % 256), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> str:
    return _make_video(tmp_path_factory.mktemp("vid") / "clip.mp4")


def test_boundary_frames_returns_two_distinct_images(video):
    first, last = extract_boundary_frames(video)
    assert isinstance(first, Image.Image) and isinstance(last, Image.Image)
    assert not np.array_equal(np.array(first), np.array(last))  # frames differ by index


def test_extract_n_frames(video):
    frames = extract_n_frames(video, n=3)
    assert len(frames) == 3
    assert all(isinstance(f, Image.Image) for f in frames)
    # the 3 sampled frames (start/middle/end) are mutually distinct
    arrs = [np.array(f) for f in frames]
    assert not np.array_equal(arrs[0], arrs[1])
    assert not np.array_equal(arrs[1], arrs[2])


def test_get_video_info(video):
    info = get_video_info(video)
    assert info["frame_count"] == N_FRAMES
    assert abs(info["fps"] - FPS) < 1.0
    assert info["width"] == 160 and info["height"] == 120
    assert abs(info["duration_sec"] - N_FRAMES / FPS) < 0.5


def test_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_boundary_frames("/no/such/video_xyz.mp4")


def test_corrupt_file_raises(tmp_path):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"this is not a video file, just plain bytes" * 50)
    with pytest.raises((ValueError, FileNotFoundError)):
        extract_boundary_frames(bad)


def test_extract_n_frames_rejects_bad_n(video):
    with pytest.raises(ValueError):
        extract_n_frames(video, n=0)
