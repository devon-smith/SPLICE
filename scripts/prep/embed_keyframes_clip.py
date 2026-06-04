# cache CLIP ViT-L/14 image embeddings for keyframes (parallel to embed_keyframes.py)
# resumable: skips keys already in the HDF5 file. used by the fused-feature pipeline.
# usage: python scripts/prep/embed_keyframes_clip.py

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

from src.data.movienet import keyframe_key
from src.data.pairs import EMB_H5_NAME
from src.models.baselines import CLIPImageEncoder

CUT_INDEX_DEFAULT = ["/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"]
OUT_DIR = "/mnt/disks/splice-data/embeddings/clip_vitl14"
MODEL_ID = "openai/clip-vit-large-patch14"
CHUNK = 20000  # keyframes per HDF5 flush
BATCH_SIZE = 256
NUM_WORKERS = 8

KEYFRAME_COLS = [
    "left_img0_path", "left_img1_path", "left_img2_path",
    "right_img0_path", "right_img1_path", "right_img2_path",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut_index", nargs="+", default=CUT_INDEX_DEFAULT)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--limit_cuts", type=int, default=0, help="first N cuts per index; 0 = all")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / EMB_H5_NAME

    # collect all unique keyframes referenced by the given cut indexes
    keyframes = {}
    for ci in args.cut_index:
        df = pd.read_parquet(ci, columns=KEYFRAME_COLS)
        if args.limit_cuts:
            df = df.head(args.limit_cuts)
        for p in pd.unique(df.values.ravel()):
            keyframes[keyframe_key(p)] = p
        print(f"{ci}: {len(df)} cuts")
    print(f"{len(keyframes)} unique keyframes referenced")

    already = set()
    if h5_path.exists():
        with h5py.File(h5_path, "r") as fh:
            already = {k.decode() if isinstance(k, bytes) else str(k) for k in fh["keys"][:]}

    todo = [(k, p) for k, p in sorted(keyframes.items())
            if k not in already and Path(p).exists()]
    n_missing = len(keyframes) - len(already) - len(todo)
    print(f"already embedded: {len(already)} | to embed: {len(todo)} | missing: {n_missing}")
    if not todo:
        print("nothing to do -- CLIP cache is complete")
        return

    encoder = CLIPImageEncoder(model_id=MODEL_ID)
    print(f"loaded {MODEL_ID} on {encoder.device}")

    mode = "a" if h5_path.exists() else "w"
    t0 = time.time()
    with h5py.File(h5_path, mode) as fh:
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            emb_by_path = encoder.encode_paths(
                [p for _, p in chunk], batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
            )
            keys = [k for k, p in chunk if p in emb_by_path]
            embs = np.stack([emb_by_path[p] for _, p in chunk if p in emb_by_path]).astype(np.float16)
            if "embeddings" not in fh:
                dim = embs.shape[1]
                fh.create_dataset("embeddings", shape=(0, dim), maxshape=(None, dim),
                                  dtype="float16", chunks=(512, dim))
                fh.create_dataset("keys", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
            n0 = fh["embeddings"].shape[0]
            fh["embeddings"].resize(n0 + len(keys), axis=0)
            fh["embeddings"][n0:] = embs
            fh["keys"].resize(n0 + len(keys), axis=0)
            fh["keys"][n0:] = keys
            fh.flush()
            print(f"embedded {min(start + CHUNK, len(todo))}/{len(todo)} keyframes")
        total = int(fh["embeddings"].shape[0])
        dim = int(fh["embeddings"].shape[1])
    elapsed = time.time() - t0

    (out_dir / "metadata.json").write_text(json.dumps({
        "model_id": MODEL_ID,
        "embedding_dim": dim,
        "total_keyframes": total,
        "dtype": "float16",
        "normalized": True,
    }, indent=2))
    print(f"embedded {len(todo)} keyframes in {elapsed/60:.1f} min; cache holds {total} (dim {dim})")


if __name__ == "__main__":
    main()
