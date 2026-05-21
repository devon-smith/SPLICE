"""Pair-feature construction for the v0 cut-continuity classifier.

A cut is scored from two 768-d DINOv2 embeddings (eL, eR). The v0 feature is::

    [ eL | eR | |eL - eR| | cos(eL, eR) ]   -> 2305-d

The two shot embeddings come from one of two modes:
  ``boundary``     eL = left shot img_2, eR = right shot img_0 (frames flanking the cut)
  ``mean_pool_3``  eL / eR = mean of that shot's three keyframe embeddings
"""

from pathlib import Path

import h5py
import numpy as np

EMBEDDING_DIM = 768
PAIR_FEATURE_DIM = 2 * EMBEDDING_DIM + EMBEDDING_DIM + 1  # concat + abs-diff + cosine
EMB_H5_NAME = "embeddings.h5"


def build_pair_features_batch(e_left: np.ndarray, e_right: np.ndarray) -> np.ndarray:
    """Vectorised feature build: ``(n,768),(n,768) -> (n,2305)`` float32."""
    e_l = np.asarray(e_left, dtype=np.float32)
    e_r = np.asarray(e_right, dtype=np.float32)
    abs_diff = np.abs(e_l - e_r)
    norm = np.linalg.norm(e_l, axis=1) * np.linalg.norm(e_r, axis=1)
    cos = np.sum(e_l * e_r, axis=1) / np.clip(norm, 1e-8, None)
    return np.concatenate([e_l, e_r, abs_diff, cos[:, None]], axis=1).astype(np.float32)


def build_pair_feature(e_left: np.ndarray, e_right: np.ndarray) -> np.ndarray:
    """Single-pair feature: ``(768,),(768,) -> (2305,)`` (convenient for tests)."""
    return build_pair_features_batch(np.asarray(e_left)[None, :], np.asarray(e_right)[None, :])[0]


def load_embeddings(emb_dir: str | Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load the keyframe embedding cache.

    Returns ``(embeddings (N,768) float32, {keyframe_key: row})``.
    """
    h5_path = Path(emb_dir) / EMB_H5_NAME
    with h5py.File(h5_path, "r") as fh:
        emb = fh["embeddings"][:].astype(np.float32)
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in fh["keys"][:]]
    return emb, {k: i for i, k in enumerate(keys)}
