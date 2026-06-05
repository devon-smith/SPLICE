# pair-feature construction for the cut-continuity classifier
# v0/v1/v2 all use the 2305-d boundary feature: [eL | eR | |eL-eR| | cos(eL,eR)]
# fused (DINOv2+CLIP) is 4610-d; hadamard adds eL*eR -> 3073-d

from pathlib import Path

import h5py
import numpy as np

EMBEDDING_DIM = 768
PAIR_FEATURE_DIM = 2 * EMBEDDING_DIM + EMBEDDING_DIM + 1  # 2305
PAIR_FEATURE_DIM_HADAMARD = PAIR_FEATURE_DIM + EMBEDDING_DIM  # 3073
EMB_H5_NAME = "embeddings.h5"

# CLIP ViT-L/14 image_embeds are 768-d, so each half matches DINOv2 at 2305-d
CLIP_EMBEDDING_DIM = 768
CLIP_PAIR_FEATURE_DIM = 2 * CLIP_EMBEDDING_DIM + CLIP_EMBEDDING_DIM + 1
FUSED_PAIR_FEATURE_DIM = PAIR_FEATURE_DIM + CLIP_PAIR_FEATURE_DIM  # 4610


# CS231N Lec 4: Backpropagation through the pair feature head
# (n,768),(n,768) -> (n,2305); with include_hadamard, (n,3073)
def build_pair_features_batch(e_left, e_right, include_hadamard=False):
    e_l = np.asarray(e_left, dtype=np.float32)
    e_r = np.asarray(e_right, dtype=np.float32)
    abs_diff = np.abs(e_l - e_r)
    norm = np.linalg.norm(e_l, axis=1) * np.linalg.norm(e_r, axis=1)
    cos = np.sum(e_l * e_r, axis=1) / np.clip(norm, 1e-8, None)
    parts = [e_l, e_r, abs_diff]
    if include_hadamard:
        parts.append(e_l * e_r)
    parts.append(cos[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32)


# single-pair feature: (768,),(768,) -> (2305,) or (3073,)
def build_pair_feature(e_left, e_right, include_hadamard=False):
    return build_pair_features_batch(
        np.asarray(e_left)[None, :], np.asarray(e_right)[None, :], include_hadamard
    )[0]


# fused DINOv2 + CLIP pair feature: [dino 2305 | clip 2305] = 4610-d
def build_fused_pair_features(dino_left, dino_right, clip_left, clip_right):
    return np.concatenate([
        build_pair_features_batch(dino_left, dino_right),
        build_pair_features_batch(clip_left, clip_right),
    ], axis=1).astype(np.float32)


# load the keyframe embedding cache -> (embeddings (N,768) float32, {key: row})
def load_embeddings(emb_dir):
    h5_path = Path(emb_dir) / EMB_H5_NAME
    with h5py.File(h5_path, "r") as fh:
        emb = fh["embeddings"][:].astype(np.float32)
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in fh["keys"][:]]
    return emb, {k: i for i, k in enumerate(keys)}
