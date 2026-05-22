"""Pair-feature construction for the cut-continuity classifier.

A cut is scored from two 768-d DINOv2 embeddings (eL, eR). The v0 feature is::

    [ eL | eR | |eL - eR| | cos(eL, eR) ]   -> 2305-d

The two shot embeddings come from one of two modes:
  ``boundary``     eL = left shot img_2, eR = right shot img_0 (frames flanking the cut)
  ``mean_pool_3``  eL / eR = mean of that shot's three keyframe embeddings

``build_fused_pair_features`` additionally supports a **fused** feature for the
v2 architecture experiments: the same ``[concat | abs-diff | cos]`` pair feature
is built once from DINOv2 boundary embeddings and once from CLIP boundary
embeddings, and the two halves are concatenated -> 4610-d.
"""

from pathlib import Path

import h5py
import numpy as np

EMBEDDING_DIM = 768
PAIR_FEATURE_DIM = 2 * EMBEDDING_DIM + EMBEDDING_DIM + 1  # concat + abs-diff + cosine
EMB_H5_NAME = "embeddings.h5"

# CLIP ViT-L/14 .image_embeds are 768-d -- the project's CLIP baseline model.
# The CLIP half therefore matches the DINOv2 half at 2305-d, and the fused
# feature is 4610-d. (A 512-d CLIP backbone would give 1537-d / 3842-d instead.)
CLIP_EMBEDDING_DIM = 768
CLIP_PAIR_FEATURE_DIM = 2 * CLIP_EMBEDDING_DIM + CLIP_EMBEDDING_DIM + 1
FUSED_PAIR_FEATURE_DIM = PAIR_FEATURE_DIM + CLIP_PAIR_FEATURE_DIM  # 4610


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


def build_fused_pair_features(
    dino_left: np.ndarray,
    dino_right: np.ndarray,
    clip_left: np.ndarray,
    clip_right: np.ndarray,
) -> np.ndarray:
    """Fused DINOv2+CLIP pair feature: ``[dino 2305-d | clip 2305-d] -> 4610-d``.

    Each half is the standard ``[concat | abs-diff | cos]`` pair feature -- one
    from DINOv2 boundary embeddings, one from CLIP boundary embeddings. The two
    backbones are complementary (DINOv2 captures low-level appearance, CLIP
    captures semantic identity), so a v2 head trained on the concatenation can
    draw on both. All four embedding arrays must be ``(n, 768)``.
    """
    dino_feat = build_pair_features_batch(dino_left, dino_right)
    clip_feat = build_pair_features_batch(clip_left, clip_right)
    return np.concatenate([dino_feat, clip_feat], axis=1).astype(np.float32)


def load_embeddings(emb_dir: str | Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load the keyframe embedding cache.

    Returns ``(embeddings (N,768) float32, {keyframe_key: row})``.
    """
    h5_path = Path(emb_dir) / EMB_H5_NAME
    with h5py.File(h5_path, "r") as fh:
        emb = fh["embeddings"][:].astype(np.float32)
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in fh["keys"][:]]
    return emb, {k: i for i, k in enumerate(keys)}
