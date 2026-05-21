"""Block 3d: generate a synthetic 8-pair toy AI-gen dataset.

Exercises the full AI-gen harness end-to-end before real clips arrive. Each pair
is two short clips; "consistent" pairs share a background colour across the cut,
"inconsistent" pairs switch colour -- enough structure for the pipeline to run
and produce non-trivial scores (the numbers themselves are meaningless).

Writes /mnt/disks/splice-data/datasets/aigen_toy/ with per-source clip folders
and labels.csv.
"""

import csv
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_toy_aigen")

OUT = Path("/mnt/disks/splice-data/datasets/aigen_toy")
FPS, N_FRAMES, SIZE = 24, 48, 256  # 2-second 256x256 clips
SQUARE = (0, 255, 255)  # BGR -- a fixed moving foreground element

# (pair_id, source, y_inconsistent, left_bg, right_bg). y=0 -> same bg across cut.
SPECS = [
    ("001", "toy_a", 0, (40, 120, 200), (40, 120, 200)),
    ("002", "toy_a", 0, (200, 80, 40), (200, 80, 40)),
    ("003", "toy_a", 1, (40, 120, 200), (60, 200, 60)),
    ("004", "toy_a", 1, (200, 80, 40), (30, 30, 220)),
    ("005", "toy_b", 0, (90, 90, 90), (90, 90, 90)),
    ("006", "toy_b", 0, (160, 40, 160), (160, 40, 160)),
    ("007", "toy_b", 1, (90, 90, 90), (20, 200, 240)),
    ("008", "toy_b", 1, (160, 40, 160), (40, 160, 40)),
]


def write_clip(path: Path, bg: tuple[int, int, int], start_x: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SIZE, SIZE))
    for i in range(N_FRAMES):
        frame = np.full((SIZE, SIZE, 3), bg, dtype=np.uint8)
        x = (start_x + i * 3) % (SIZE - 60)
        cv2.rectangle(frame, (x, 100), (x + 60, 160), SQUARE, -1)
        writer.write(frame)
    writer.release()


def main() -> None:
    rows = []
    for pid, source, y, left_bg, right_bg in SPECS:
        clip_dir = OUT / source
        clip_dir.mkdir(parents=True, exist_ok=True)
        # right clip's motion continues from where the left clip ended
        write_clip(clip_dir / f"pair_{pid}_left.mp4", left_bg, start_x=10)
        write_clip(clip_dir / f"pair_{pid}_right.mp4", right_bg, start_x=10 + N_FRAMES * 3)
        rows.append(
            {
                "pair_id": pid,
                "source": source,
                "y_inconsistent": y,
                "notes": "consistent (same colour)" if y == 0 else "inconsistent (colour switch)",
            }
        )

    with open(OUT / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "source", "y_inconsistent", "notes"])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d toy pairs (16 clips) + labels.csv to %s", len(rows), OUT)


if __name__ == "__main__":
    main()
