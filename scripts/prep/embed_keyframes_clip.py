"""Cache CLIP ViT-L/14 image embeddings for keyframes -- the v2 fusion scaffold.

A parallel to embed_keyframes.py. It caches CLIP image embeddings to an HDF5
with the *same* schema (datasets ``embeddings`` (N, dim) float16 and ``keys``
(N,) str, keyed by ``keyframe_key``) so the fused-feature builder can load it
with the same ``load_embeddings`` helper used for the DINOv2 cache.

The run is idempotent (already-cached keys are skipped) and incremental, so the
verification subset and a later full pass share one cache file.

Use ``--limit_cuts`` to embed only the first N cuts of each index -- this is the
verification subset. **Do not run the full MovieNet pass yet** (v2 work; ~1.5M
keyframes); the scaffold is for the fused-feature pipeline check only.

Example (verification subset):
  python scripts/prep/embed_keyframes_clip.py \\
      --cut_index /mnt/disks/splice-data/outputs/fusion_verify/subset_cuts.parquet \\
                  /mnt/disks/splice-data/outputs/aigen_eval/cuts.parquet \\
      --out /mnt/disks/splice-data/embeddings/clip_vitl14
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.movienet import keyframe_key  # noqa: E402
from src.data.pairs import EMB_H5_NAME  # noqa: E402
from src.models.baselines import CLIPImageEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed_keyframes_clip")

DEFAULT_CUT_INDEX = ["/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"]
DEFAULT_OUT = "/mnt/disks/splice-data/embeddings/clip_vitl14"
KEYFRAME_COLS = [
    "left_img0_path",
    "left_img1_path",
    "left_img2_path",
    "right_img0_path",
    "right_img1_path",
    "right_img2_path",
]


def unique_keyframes(cut_indexes: list[str], limit_cuts: int) -> dict[str, str]:
    """All distinct keyframes referenced by the given cut indexes: ``{key: path}``."""
    keyframes: dict[str, str] = {}
    for ci in cut_indexes:
        df = pd.read_parquet(ci, columns=KEYFRAME_COLS)
        if limit_cuts:
            df = df.head(limit_cuts)
        for p in pd.unique(df.values.ravel()):
            keyframes[keyframe_key(p)] = p
        log.info("%s: %d cuts", ci, len(df))
    return keyframes


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
    ap.add_argument("--cut_index", nargs="+", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model_id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--limit_cuts", type=int, default=0, help="first N cuts per index; 0 = all")
    ap.add_argument("--chunk", type=int, default=20000, help="keyframes per HDF5 flush")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / EMB_H5_NAME

    keyframes = unique_keyframes(args.cut_index, args.limit_cuts)
    log.info("%d unique keyframes referenced", len(keyframes))

    already = done_keys(h5_path)
    todo = [(k, p) for k, p in sorted(keyframes.items()) if k not in already and Path(p).exists()]
    n_missing = len(keyframes) - len(already) - len(todo)
    log.info(
        "already embedded: %d | to embed: %d | missing on disk: %d",
        len(already),
        len(todo),
        n_missing,
    )
    if not todo:
        log.info("nothing to do -- CLIP cache is complete for this cut index")
        return

    encoder = CLIPImageEncoder(model_id=args.model_id)
    log.info("loaded %s on %s", args.model_id, encoder.device)

    mode = "a" if h5_path.exists() else "w"
    t0 = time.time()
    with h5py.File(h5_path, mode) as fh:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start : start + args.chunk]
            emb_by_path = encoder.encode_paths(
                [p for _, p in chunk], batch_size=args.batch_size, num_workers=args.num_workers
            )
            keys = [k for k, p in chunk if p in emb_by_path]
            embs = np.stack([emb_by_path[p] for _, p in chunk if p in emb_by_path]).astype(
                np.float16
            )
            if "embeddings" not in fh:
                dim = embs.shape[1]
                fh.create_dataset(
                    "embeddings",
                    shape=(0, dim),
                    maxshape=(None, dim),
                    dtype="float16",
                    chunks=(512, dim),
                )
                fh.create_dataset("keys", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
            _append(fh, keys, embs)
            fh.flush()
            log.info("embedded %d/%d keyframes", min(start + args.chunk, len(todo)), len(todo))
        total = int(fh["embeddings"].shape[0])
        dim = int(fh["embeddings"].shape[1])
    elapsed = time.time() - t0

    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "embedding_dim": dim,
                "total_keyframes": total,
                "dtype": "float16",
                "normalized": True,
                "cut_index": list(args.cut_index),
                "limit_cuts": args.limit_cuts,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    log.info(
        "embedded %d keyframes in %.1f min; CLIP cache now holds %d (dim %d)",
        len(todo),
        elapsed / 60,
        total,
        dim,
    )


if __name__ == "__main__":
    main()
