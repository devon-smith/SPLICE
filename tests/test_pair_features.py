"""Unit tests for pair-feature construction (src.data.pairs)."""

import numpy as np

from src.data.pairs import (
    EMBEDDING_DIM,
    FUSED_PAIR_FEATURE_DIM,
    PAIR_FEATURE_DIM,
    PAIR_FEATURE_DIM_HADAMARD,
    build_fused_pair_features,
    build_pair_feature,
    build_pair_features_batch,
)

RNG = np.random.default_rng(0)


def test_feature_dim_constants():
    assert EMBEDDING_DIM == 768
    assert PAIR_FEATURE_DIM == 2305
    assert PAIR_FEATURE_DIM_HADAMARD == 3073
    assert FUSED_PAIR_FEATURE_DIM == 4610


def test_hadamard_flag_off_preserves_default_dim():
    e_l = RNG.standard_normal(768).astype(np.float32)
    e_r = RNG.standard_normal(768).astype(np.float32)
    assert build_pair_feature(e_l, e_r).shape == (2305,)
    assert build_pair_feature(e_l, e_r, include_hadamard=False).shape == (2305,)
    # default-off path matches explicit-off path byte-for-byte
    np.testing.assert_array_equal(
        build_pair_feature(e_l, e_r), build_pair_feature(e_l, e_r, include_hadamard=False)
    )


def test_hadamard_flag_on_produces_3073d_with_correct_layout():
    e_l = RNG.standard_normal(768).astype(np.float32)
    e_r = RNG.standard_normal(768).astype(np.float32)
    feat = build_pair_feature(e_l, e_r, include_hadamard=True)
    assert feat.shape == (3073,)
    assert feat.dtype == np.float32
    # layout: [eL | eR | |eL-eR| | eL*eR | cos]
    np.testing.assert_allclose(feat[:768], e_l, rtol=1e-5)
    np.testing.assert_allclose(feat[768:1536], e_r, rtol=1e-5)
    np.testing.assert_allclose(feat[1536:2304], np.abs(e_l - e_r), rtol=1e-5)
    np.testing.assert_allclose(feat[2304:3072], e_l * e_r, rtol=1e-5)
    # cosine column unchanged
    assert abs(feat[-1] - build_pair_feature(e_l, e_r)[-1]) < 1e-6


def test_hadamard_batch_consistency():
    e_l = RNG.standard_normal((5, 768)).astype(np.float32)
    e_r = RNG.standard_normal((5, 768)).astype(np.float32)
    batch = build_pair_features_batch(e_l, e_r, include_hadamard=True)
    assert batch.shape == (5, 3073)
    for i in range(5):
        np.testing.assert_allclose(
            batch[i], build_pair_feature(e_l[i], e_r[i], include_hadamard=True), rtol=1e-5
        )


def test_build_pair_feature_shape_and_layout():
    e_l = RNG.standard_normal(768).astype(np.float32)
    e_r = RNG.standard_normal(768).astype(np.float32)
    feat = build_pair_feature(e_l, e_r)
    assert feat.shape == (2305,)
    assert feat.dtype == np.float32
    np.testing.assert_allclose(feat[:768], e_l, rtol=1e-5)
    np.testing.assert_allclose(feat[768:1536], e_r, rtol=1e-5)
    np.testing.assert_allclose(feat[1536:2304], np.abs(e_l - e_r), rtol=1e-5)


def test_cosine_component():
    v = RNG.standard_normal(768).astype(np.float32)
    assert abs(build_pair_feature(v, v)[-1] - 1.0) < 1e-4  # identical -> cos 1
    a = np.zeros(768, dtype=np.float32)
    b = np.zeros(768, dtype=np.float32)
    a[0] = 1.0
    b[1] = 1.0
    assert abs(build_pair_feature(a, b)[-1]) < 1e-6  # orthogonal -> cos 0


def test_zero_vectors_no_nan():
    z = np.zeros(768, dtype=np.float32)
    feat = build_pair_feature(z, z)
    assert not np.isnan(feat).any()


def test_batch_matches_single():
    e_l = RNG.standard_normal((5, 768)).astype(np.float32)
    e_r = RNG.standard_normal((5, 768)).astype(np.float32)
    batch = build_pair_features_batch(e_l, e_r)
    assert batch.shape == (5, 2305)
    for i in range(5):
        np.testing.assert_allclose(batch[i], build_pair_feature(e_l[i], e_r[i]), rtol=1e-5)


def test_fused_pair_features_shape_and_layout():
    dino_l = RNG.standard_normal((4, 768)).astype(np.float32)
    dino_r = RNG.standard_normal((4, 768)).astype(np.float32)
    clip_l = RNG.standard_normal((4, 768)).astype(np.float32)
    clip_r = RNG.standard_normal((4, 768)).astype(np.float32)
    fused = build_fused_pair_features(dino_l, dino_r, clip_l, clip_r)
    assert fused.shape == (4, 4610)
    assert fused.dtype == np.float32
    assert not np.isnan(fused).any()
    # the two halves are the standalone DINOv2 and CLIP pair features
    np.testing.assert_allclose(
        fused[:, :2305], build_pair_features_batch(dino_l, dino_r), rtol=1e-5
    )
    np.testing.assert_allclose(
        fused[:, 2305:], build_pair_features_batch(clip_l, clip_r), rtol=1e-5
    )
