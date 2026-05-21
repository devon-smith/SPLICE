"""Frame extraction from video clips, for the AI-generated-video evaluation set.

Clips are short (a few seconds), so frames are decoded sequentially -- this
avoids the seek-precision and corrupt-keyframe problems of `cap.set` and is
fully reliable. All frames are returned as RGB PIL Images (OpenCV decodes BGR).
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_MAX_FRAMES = 20_000  # safety cap; AI-gen clips are seconds long


def _open(video_path: str | Path) -> cv2.VideoCapture:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"could not open video (corrupt or unsupported codec): {path}")
    return cap


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _decode_all(video_path: str | Path) -> list[np.ndarray]:
    """Decode every readable frame (BGR). Raises if the video yields none."""
    cap = _open(video_path)
    frames: list[np.ndarray] = []
    while len(frames) < _MAX_FRAMES:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no decodable frames in video: {video_path}")
    return frames


def get_video_info(video_path: str | Path) -> dict:
    """Return ``{duration_sec, fps, frame_count, width, height, codec}``."""
    cap = _open(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
    if frame_count <= 0:  # container metadata unreliable -- count by decoding
        frame_count = len(_decode_all(video_path))
    return {
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": codec,
    }


def extract_boundary_frames(video_path: str | Path) -> tuple[Image.Image, Image.Image]:
    """Return ``(first_frame, last_frame)`` -- the first and last decodable frames."""
    frames = _decode_all(video_path)
    return _to_pil(frames[0]), _to_pil(frames[-1])


def extract_n_frames(video_path: str | Path, n: int = 3) -> list[Image.Image]:
    """Uniformly sample ``n`` frames (n=3 -> 0%, 50%, 100% of duration)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    frames = _decode_all(video_path)
    if len(frames) < n:
        idx = list(range(len(frames))) + [len(frames) - 1] * (n - len(frames))
    elif n == 1:
        idx = [len(frames) // 2]
    else:
        idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [_to_pil(frames[i]) for i in idx]


def extract_at_timestamps(video_path: str | Path, timestamps_sec: list[float]) -> list[Image.Image]:
    """Sample frames at explicit times (seconds); times are clamped into range."""
    frames = _decode_all(video_path)
    fps = float(get_video_info(video_path)["fps"]) or len(frames)
    out = []
    for t in timestamps_sec:
        idx = min(max(round(t * fps), 0), len(frames) - 1)
        out.append(_to_pil(frames[idx]))
    return out
