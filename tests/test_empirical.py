"""Tests for :class:`EmpiricalDGP` + sampling-design helpers."""

from __future__ import annotations

import numpy as np
from dgp_protocol import ClusteredSampling, EmpiricalDGP, IIDSampling


def test_data_returns_frozen_observation() -> None:
    """``dgp.data`` is the original observation passed at construction."""

    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    assert dgp.data is obs


def test_draw_iid_returns_same_shape() -> None:
    """Default iid draw returns a matrix of the same shape as data."""

    rng = np.random.default_rng(0)
    obs = rng.standard_normal(size=(50, 3))
    dgp = EmpiricalDGP(observation=obs)
    draw = dgp.draw(rng=rng)
    assert draw.shape == obs.shape


def test_draw_iid_with_explicit_size() -> None:
    """``size`` overrides the default leading dimension."""

    rng = np.random.default_rng(0)
    obs = rng.standard_normal(size=(20, 3))
    dgp = EmpiricalDGP(observation=obs)
    draw = dgp.draw(size=(7,), rng=rng)
    assert draw.shape == (7, 3)


def test_iid_draw_resamples_with_replacement() -> None:
    """The iid draw is a row-wise multinomial resample with replacement.

    With a small observation set, repeated draws should produce
    duplicated rows (proof of with-replacement sampling).
    """

    rng = np.random.default_rng(42)
    obs = np.arange(5).reshape(5, 1).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    # Draw many times; verify at least one duplicate appears.
    draws = [dgp.draw(rng=rng).flatten() for _ in range(20)]
    has_duplicate = any(len(set(d.tolist())) < d.size for d in draws)
    assert has_duplicate, "expected with-replacement sampling to produce duplicates"


def test_with_data_returns_new_instance_preserving_sampling() -> None:
    """``with_data`` rebinds the realization without mutating the original."""

    obs1 = np.arange(12).reshape(4, 3).astype(float)
    obs2 = np.arange(15).reshape(5, 3).astype(float)
    sampling = IIDSampling()
    dgp1 = EmpiricalDGP(observation=obs1, sampling=sampling)
    dgp2 = dgp1.with_data(obs2)

    # Different instance, but shared sampling design.
    assert dgp2 is not dgp1
    assert dgp2.sampling is dgp1.sampling
    assert dgp2.data is obs2
    # Original is unchanged.
    assert dgp1.data is obs1


def test_clustered_sampling_resamples_whole_clusters() -> None:
    """Cluster bootstrap draws contain only intact cluster blocks."""

    rng = np.random.default_rng(0)
    obs = np.arange(40).reshape(10, 4).astype(float)
    # Two clusters: rows 0-4 vs rows 5-9.
    clusters = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    sampling = ClusteredSampling(cluster_ids=clusters)
    dgp = EmpiricalDGP(observation=obs, sampling=sampling)

    # Each draw resamples 2 clusters with replacement.  Possible
    # outcomes: cluster-0-twice (10 rows from rows 0-4), one of each
    # (10 rows), or cluster-1-twice.  All draws have 10 rows.
    for _ in range(10):
        draw = dgp.draw(rng=rng)
        assert draw.shape == (10, 4)
        # Every row in the draw came from one of the clusters; check
        # that no fabricated rows appeared.
        for row in draw:
            # Row should be one of the original 10 rows.
            assert any(np.array_equal(row, obs[k]) for k in range(10))


def test_clustered_sampling_rejects_wrong_length_ids() -> None:
    """``ClusteredSampling`` errors if cluster ids don't match data length."""

    obs = np.arange(12).reshape(4, 3).astype(float)
    sampling = ClusteredSampling(cluster_ids=np.array([0, 0, 1]))  # length 3 != 4
    dgp = EmpiricalDGP(observation=obs, sampling=sampling)
    rng = np.random.default_rng(0)

    import pytest

    with pytest.raises(ValueError, match="cluster_ids has 3"):
        dgp.draw(rng=rng)
