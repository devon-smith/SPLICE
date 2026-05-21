"""Cache DINOv2 embeddings for every unique keyframe referenced by the cut index.

This is the only GPU-heavy stage. Each keyframe is encoded once (a shot appears
as both the "left" and "right" side of different cuts) and written to a single
resizable HDF5 file. The run is idempotent: keyframes already present in the HDF5
are skipped, so an interrupted run resumes cleanly.

Output:
  <out>/embeddings.h5   datasets: embeddings (N,768) float16, keys (N,) str
  <out>/metadata.json   provenance

Example:
  python scripts/prep/embed_keyframes.py \\
      --cut_index /mnt/disks/splice-data/outputs/cut_index/cuts.parquet \\
      --out /mnt/disks/splice-data/embeddings/dinov2_base
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.movienet import keyframe_key  # noqa: E402
from src.data.pairs import EMB_H5_NAME  # noqa: E402
from src.models.dinov2_encoder import DINOv2Encoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed_keyframes")

DEFAULT_CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
DEFAULT_OUT = "/mnt/disks/splice-data/embeddings/dinov2_base"
KEYFRAME_COLS = [
    "left_img0_path",
    "left_img1_path",
    "left_img2_path",
    "right_img0_path",
    "right_img1_path",
    "right_img2_path",
]


class KeyframeDataset(Dataset):
    """Loads and HF-preprocesses keyframes; yields ``(key, pixel_values)``."""

    def __init__(self, items: list[tuple[str, str]], processor) -> None:
        self.items = items
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        key, path = self.items[i]
        img = Image.open(path).convert("RGB")
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return key, pv


def unique_keyframes(cut_index: str) -> dict[str, str]:
    """All distinct keyframes referenced by the cut index: ``{key: absolute_path}``."""
    df = pd.read_parquet(cut_index, columns=KEYFRAME_COLS)
    paths = pd.unique(df.values.ravel())
    return {keyframe_key(p): p for p in paths}


def done_keys(h5_path: Path) -> set[str]:
    """Keys already embedded in a previous run."""
    if not h5_path.exists():
        return set()
    with h5py.File(h5_path, "r") as fh:
        return {k.decode() if isinstance(k, bytes) else str(k) for k in fh["keys"][:]}


def _append(fh: h5py.File, keys: list[str], embs: np.ndarray) -> None:
    n0 = fh["embeddings"].shape[0]
    fh["embeddings"].resize(n0 + len(keys), axis=0)
    fh["embeddings"][n0:] = embs
    fh["keys"].resize(n0 + len(keys), axis=0)
    fh["keys"][n0:] = keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model_id", default="facebook/dinov2-base")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--flush_batches", type=int, default=200, help="checkpoint interval")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / EMB_H5_NAME

    keyframes = unique_keyframes(args.cut_index)
    log.info("cut index references %d unique keyframes", len(keyframes))

    already = done_keys(h5_path)
    # only embed keyframes whose image file actually exists on disk
    todo = [(k, p) for k, p in sorted(keyframes.items()) if k not in already and Path(p).exists()]
    n_missing = len(keyframes) - len(already) - len(todo)
    log.info(
        "already embedded: %d | to embed: %d | missing on disk: %d",
        len(already),
        len(todo),
        n_missing,
    )
    if not todo:
        log.info("nothing to do -- embedding cache is complete")
        return

    encoder = DINOv2Encoder(model_id=args.model_id)
    log.info("loaded %s on %s (dim=%d)", args.model_id, encoder.device, encoder.embedding_dim)

    loader = DataLoader(
        KeyframeDataset(todo, encoder.processor),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    mode = "a" if h5_path.exists() else "w"
    with h5py.File(h5_path, mode) as fh:
        if "embeddings" not in fh:
            fh.create_dataset(
                "embeddings",
                shape=(0, encoder.embedding_dim),
                maxshape=(None, encoder.embedding_dim),
                dtype="float16",
                chunks=(512, encoder.embedding_dim),
            )
            fh.create_dataset("keys", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

        buf_keys: list[str] = []
        buf_embs: list[np.ndarray] = []
        t0 = time.time()
        for bi, (keys, pixel_values) in enumerate(tqdm(loader, desc="embedding")):
            emb = encoder.encode(pixel_values).numpy().astype(np.float16)
            buf_keys.extend(keys)
            buf_embs.append(emb)
            if (bi + 1) % args.flush_batches == 0:
                _append(fh, buf_keys, np.concatenate(buf_embs))
                fh.flush()
                buf_keys, buf_embs = [], []
        if buf_keys:
            _append(fh, buf_keys, np.concatenate(buf_embs))
            fh.flush()
        total = int(fh["embeddings"].shape[0])
        elapsed = time.time() - t0

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "image_size": 224,
                "embedding_dim": encoder.embedding_dim,
                "total_keyframes": total,
                "dtype": "float16",
                "cut_index": str(args.cut_index),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    log.info(
        "embedded %d keyframes in %.1f min (%.0f kf/s); cache now holds %d",
        len(todo),
        elapsed / 60,
        len(todo) / max(elapsed, 1e-6),
        total,
    )


if __name__ == "__main__":
    main()
