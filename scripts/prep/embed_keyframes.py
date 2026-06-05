# cache DINOv2 embeddings for every unique keyframe in the cut index
# resumable: skips keys already in the HDF5 file
# usage: python scripts/prep/embed_keyframes.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate the occasional bad keyframe

from src.data.movienet import keyframe_key
from src.data.pairs import EMB_H5_NAME
from src.models.dinov2_encoder import DINOv2Encoder

CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
OUT_DIR = "/mnt/disks/splice-data/embeddings/dinov2_base"
MODEL_ID = "facebook/dinov2-base"
BATCH_SIZE = 128
NUM_WORKERS = 8
FLUSH_BATCHES = 200  # checkpoint every N batches

KEYFRAME_COLS = [
    "left_img0_path", "left_img1_path", "left_img2_path",
    "right_img0_path", "right_img1_path", "right_img2_path",
]


class KeyframeDataset(Dataset):
    def __init__(self, items, processor):
        self.items = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, path = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
            pv = self.processor(images=img, return_tensors="pt")["pixel_values"][0]  # CS231N Lec 5: Image resizing and normalization preprocessing
        except Exception:  # noqa: BLE001 - skip unreadable frames, embed the rest
            return key, None
        return key, pv


# drop frames that failed to load, stack the rest
def collate(batch):
    good = [(k, pv) for k, pv in batch if pv is not None]
    if not good:
        return [], None
    return [k for k, _ in good], torch.stack([pv for _, pv in good])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut_index", default=CUT_INDEX)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / EMB_H5_NAME

    df = pd.read_parquet(args.cut_index, columns=KEYFRAME_COLS)
    paths = pd.unique(df.values.ravel())
    keyframes = {keyframe_key(p): p for p in paths}
    print(f"cut index references {len(keyframes)} unique keyframes")

    already = set()
    if h5_path.exists():
        with h5py.File(h5_path, "r") as fh:
            already = {k.decode() if isinstance(k, bytes) else str(k) for k in fh["keys"][:]}

    todo = [(k, p) for k, p in sorted(keyframes.items())
            if k not in already and Path(p).exists()]
    n_missing = len(keyframes) - len(already) - len(todo)
    print(f"already embedded: {len(already)} | to embed: {len(todo)} | missing: {n_missing}")
    if not todo:
        print("nothing to do -- cache is complete")
        return

    encoder = DINOv2Encoder(model_id=MODEL_ID)
    print(f"loaded {MODEL_ID} on {encoder.device} (dim={encoder.embedding_dim})")

    loader = DataLoader(
        KeyframeDataset(todo, encoder.processor),
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=collate,
    )

    mode = "a" if h5_path.exists() else "w"
    with h5py.File(h5_path, mode) as fh:
        if "embeddings" not in fh:
            fh.create_dataset("embeddings", shape=(0, encoder.embedding_dim),
                              maxshape=(None, encoder.embedding_dim),
                              dtype="float16", chunks=(512, encoder.embedding_dim))
            fh.create_dataset("keys", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())

        buf_keys, buf_embs = [], []
        t0 = time.time()
        for bi, (keys, pixel_values) in enumerate(tqdm(loader, desc="embedding")):
            if pixel_values is None:
                continue
            emb = encoder.encode(pixel_values).numpy().astype(np.float16)
            buf_keys.extend(keys)
            buf_embs.append(emb)
            if (bi + 1) % FLUSH_BATCHES == 0:
                n0 = fh["embeddings"].shape[0]
                fh["embeddings"].resize(n0 + len(buf_keys), axis=0)
                fh["embeddings"][n0:] = np.concatenate(buf_embs)
                fh["keys"].resize(n0 + len(buf_keys), axis=0)
                fh["keys"][n0:] = buf_keys
                fh.flush()
                buf_keys, buf_embs = [], []
        if buf_keys:
            n0 = fh["embeddings"].shape[0]
            fh["embeddings"].resize(n0 + len(buf_keys), axis=0)
            fh["embeddings"][n0:] = np.concatenate(buf_embs)
            fh["keys"].resize(n0 + len(buf_keys), axis=0)
            fh["keys"][n0:] = buf_keys
            fh.flush()
        total = int(fh["embeddings"].shape[0])
        elapsed = time.time() - t0

    (out_dir / "metadata.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "embedding_dim": encoder.embedding_dim,
        "total_keyframes": total,
        "dtype": "float16",
        "cut_index": str(args.cut_index),
    }, indent=2))
    print(f"embedded {len(todo)} keyframes in {elapsed/60:.1f} min; cache holds {total}")


if __name__ == "__main__":
    main()
