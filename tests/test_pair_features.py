"""Unit tests for pair-feature construction (src.data.pairs)."""

import numpy as np

from src.data.pairs import (
    EMBEDDING_DIM,
    PAIR_FEATURE_DIM,
    build_pair_feature,
    build_pair_features_batch,
)

RNG = np.random.default_rng(0)


def test_feature_dim_constants():
    assert EMBEDDING_DIM == 768
    assert PAIR_FEATURE_DIM == 2305


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
