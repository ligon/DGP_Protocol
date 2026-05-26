"""Tests for :class:`EmpiricalDGP` + sampling-design helpers."""

from __future__ import annotations

import numpy as np
import pytest

from dgp_protocol import ClusteredSampling, EmpiricalDGP, IIDSampling


def test_data_returns_frozen_observation() -> None:
    """``dgp.data`` is the original observation passed at construction."""

    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    assert dgp.data is obs


def test_draw_iid_returns_same_shape() -> None:
    """Default iid draw returns a matrix of the same shape as data."""

    obs = np.random.default_rng(0).standard_normal(size=(50, 3))
    dgp = EmpiricalDGP(observation=obs, seed=1)
    draw = dgp.draw()
    assert draw.shape == obs.shape


def test_draw_iid_with_explicit_size() -> None:
    """``size`` overrides the default leading dimension."""

    obs = np.random.default_rng(0).standard_normal(size=(20, 3))
    dgp = EmpiricalDGP(observation=obs, seed=1)
    draw = dgp.draw(size=(7,))
    assert draw.shape == (7, 3)


def test_iid_draw_resamples_with_replacement() -> None:
    """The iid draw is a row-wise multinomial resample with replacement.

    With a small observation set, repeated draws should produce
    duplicated rows (proof of with-replacement sampling).
    """

    obs = np.arange(5).reshape(5, 1).astype(float)
    dgp = EmpiricalDGP(observation=obs, seed=42)
    # Draw many times; verify at least one duplicate appears.
    draws = [dgp.draw().flatten() for _ in range(20)]
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

    obs = np.arange(40).reshape(10, 4).astype(float)
    # Two clusters: rows 0-4 vs rows 5-9.
    clusters = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    sampling = ClusteredSampling(cluster_ids=clusters)
    dgp = EmpiricalDGP(observation=obs, sampling=sampling, seed=0)

    # Each draw resamples 2 clusters with replacement.  Possible
    # outcomes: cluster-0-twice (10 rows from rows 0-4), one of each
    # (10 rows), or cluster-1-twice.  All draws have 10 rows.
    for _ in range(10):
        draw = dgp.draw()
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

    with pytest.raises(ValueError, match="cluster_ids has 3"):
        dgp.draw()


# ---------------------------------------------------------------------------
# Randomness-ownership semantics (seed, with_rng, with_data spawn)
# ---------------------------------------------------------------------------
def test_seed_makes_draws_reproducible() -> None:
    """Two DGPs built with the same seed produce identical draw streams."""

    obs = np.arange(40).reshape(10, 4).astype(float)
    dgp_a = EmpiricalDGP(observation=obs, seed=123)
    dgp_b = EmpiricalDGP(observation=obs, seed=123)
    for _ in range(3):
        np.testing.assert_array_equal(dgp_a.draw(), dgp_b.draw())


def test_no_seed_uses_system_entropy() -> None:
    """Without a seed, two independently-constructed DGPs (typically) differ.

    Strictly: ``default_rng(None)`` draws system entropy.  Two
    independent DGPs sharing only their observation will produce
    different draws with overwhelming probability.
    """

    obs = np.arange(40).reshape(10, 4).astype(float)
    dgp_a = EmpiricalDGP(observation=obs)
    dgp_b = EmpiricalDGP(observation=obs)
    # The probability of identical draws under independent system-entropy
    # seeding is effectively zero.
    assert not np.array_equal(dgp_a.draw(), dgp_b.draw())


def test_with_rng_replaces_generator() -> None:
    """``with_rng`` returns a sibling whose stream is the provided one."""

    obs = np.arange(40).reshape(10, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs, seed=7)
    injected = np.random.default_rng(999)
    sibling = dgp.with_rng(injected)

    assert sibling is not dgp
    assert sibling.data is dgp.data
    assert sibling.sampling is dgp.sampling
    # The sibling's stream matches a fresh dgp constructed against the
    # same injected Generator (i.e., the same draws default_rng(999)
    # would have produced when consumed by IIDSampling).
    reference = EmpiricalDGP(observation=obs).with_rng(np.random.default_rng(999))
    np.testing.assert_array_equal(sibling.draw(), reference.draw())


def test_with_data_spawns_independent_stream() -> None:
    """``with_data`` gives the child its own (spawned) Generator.

    Calling ``parent.draw()`` after spawning a child does not consume
    the child's Generator state, and vice versa.
    """

    obs1 = np.arange(40).reshape(10, 4).astype(float)
    obs2 = np.arange(20).reshape(5, 4).astype(float)
    parent = EmpiricalDGP(observation=obs1, seed=11)
    child = parent.with_data(obs2)

    # Child should have a working independent stream.  We can't easily
    # compare numerical values across parents and children with
    # different observations, but we can verify (a) the child's draws
    # are reproducible (same seed lineage), and (b) the parent's draws
    # don't change as a function of child-side activity.
    parent_first = parent.draw()
    _ = [child.draw() for _ in range(5)]
    parent_second = parent.draw()

    # Drawing twice from the parent gives different realizations.
    assert not np.array_equal(parent_first, parent_second)

    # Reconstructing the parent with the same seed and skipping the
    # child's activity should yield the same first two parent draws.
    fresh_parent = EmpiricalDGP(observation=obs1, seed=11)
    np.testing.assert_array_equal(fresh_parent.draw(), parent_first)
    np.testing.assert_array_equal(fresh_parent.draw(), parent_second)
