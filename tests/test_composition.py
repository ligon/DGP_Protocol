"""Tests for composition primitives (:class:`TwoStageDGP`, :func:`with_data`)."""

from __future__ import annotations

import numpy as np
import pytest
from dgp_protocol import (
    DataGeneratingProcess,
    EmpiricalDGP,
    ParametricDGP,
    TwoStageDGP,
    with_data,
)


# ---------------------------------------------------------------------------
# TwoStageDGP basic behaviour
# ---------------------------------------------------------------------------
def test_two_stage_draw_returns_per_cluster_list() -> None:
    """A two-stage DGP returns a list of per-cluster realizations.

    Outer draws cluster characteristics; inner is a callable that
    builds a DGP for each cluster, given a per-cluster Generator.
    """

    # Outer: 3 clusters, each characterised by a single scalar.
    cluster_chars = np.array([[1.0], [2.0], [3.0]])
    outer = EmpiricalDGP(observation=cluster_chars, seed=1)

    # Inner: each cluster has an iid Gaussian draw with std equal to
    # the cluster's scalar characteristic.  Uses the per-cluster rng
    # via ``with_rng`` so the composite's seed drives within-cluster
    # randomness deterministically.
    def make_inner(chars, rng):
        sigma = float(chars[0])
        base = ParametricDGP(
            generator=lambda rng_inner, shape: sigma * rng_inner.standard_normal(shape),
            default_shape=(5, 2),
        )
        return base.with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=make_inner, seed=0)
    realization = ts.draw()

    assert isinstance(realization, list)
    assert len(realization) == 3  # one per cluster
    for cluster_draw in realization:
        assert cluster_draw.shape == (5, 2)


def test_two_stage_inner_called_with_cluster_chars_and_rng() -> None:
    """The ``inner`` callable receives the cluster row plus an rng."""

    cluster_chars = np.array([[10.0], [20.0]])
    outer = EmpiricalDGP(observation=cluster_chars, seed=0)

    seen_chars = []
    seen_rngs = []

    def make_inner(chars, rng):
        seen_chars.append(chars.copy())
        seen_rngs.append(rng)
        # Inner is degenerate: always returns zeros.
        return ParametricDGP(
            generator=lambda rng_inner, shape: np.zeros(shape),
            default_shape=(1, 1),
        )

    ts = TwoStageDGP(outer=outer, inner=make_inner, seed=42)
    _ = ts.draw()

    # ``inner`` was called once per outer-drawn cluster.
    assert len(seen_chars) == cluster_chars.shape[0]
    seen_values = {float(c[0]) for c in seen_chars}
    assert seen_values.issubset({10.0, 20.0})
    # Each ``rng`` is a numpy Generator and distinct across clusters.
    assert all(isinstance(r, np.random.Generator) for r in seen_rngs)
    assert seen_rngs[0] is not seen_rngs[1]


def test_two_stage_recursive_composition() -> None:
    """A three-stage composition: region -> cluster-within-region -> obs.

    Demonstrates that ``inner`` can return another :class:`TwoStageDGP`.
    """

    # Top-level: 2 regions, each characterised by a scalar.
    region_chars = np.array([[100.0], [200.0]])
    region_dgp = EmpiricalDGP(observation=region_chars, seed=0)

    def make_cluster_dgp_within_region(region_chars_row, region_rng):
        # Each region has 3 clusters, each characterised by the
        # region characteristic times a small jitter.
        base = float(region_chars_row[0])
        cluster_chars = np.array([[base + 1], [base + 2], [base + 3]])
        cluster_dgp = EmpiricalDGP(observation=cluster_chars, seed=0)

        def make_observation_dgp(cluster_chars_row, obs_rng):
            base_dgp = ParametricDGP(
                generator=lambda rng_inner, shape: np.zeros(shape)
                + cluster_chars_row[0],
                default_shape=(1, 1),
            )
            return base_dgp.with_rng(obs_rng)

        # Inject ``region_rng`` as the inner TwoStageDGP's own spawn
        # source so the whole composition is reproducible from the
        # top-level seed.
        return TwoStageDGP(outer=cluster_dgp, inner=make_observation_dgp).with_rng(
            region_rng
        )

    three_stage = TwoStageDGP(
        outer=region_dgp, inner=make_cluster_dgp_within_region, seed=1
    )

    realization = three_stage.draw()
    # Outer (region) realization has 2 rows -> 2 list entries.
    assert len(realization) == 2
    # Each entry is itself a list (the second-stage's per-cluster realizations).
    for region_realization in realization:
        assert isinstance(region_realization, list)
        assert len(region_realization) == 3  # 3 clusters per region


def test_two_stage_with_data_preserves_structure() -> None:
    """``with_data`` on a TwoStageDGP rebinds observation but keeps outer/inner."""

    cluster_chars = np.array([[1.0], [2.0]])
    outer = EmpiricalDGP(observation=cluster_chars)

    def make_inner(c, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(1, 1),
        ).with_rng(rng)

    ts1 = TwoStageDGP(outer=outer, inner=make_inner)
    fake_realization = ("placeholder", "object")
    ts2 = ts1.with_data(fake_realization)

    assert ts2 is not ts1
    assert ts2.outer is ts1.outer
    assert ts2.inner is ts1.inner
    assert ts2.data == fake_realization
    assert ts1.data is None  # unchanged


def test_two_stage_seeded_is_reproducible() -> None:
    """Two TwoStageDGPs with identically-seeded constituents agree on draws."""

    cluster_chars = np.array([[1.0], [2.0], [3.0]])

    def build():
        outer = EmpiricalDGP(observation=cluster_chars, seed=7)

        def make_inner(chars, rng):
            return ParametricDGP(
                generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
                default_shape=(4, 1),
            ).with_rng(rng)

        return TwoStageDGP(outer=outer, inner=make_inner, seed=99)

    a, b = build(), build()
    realizations_a = a.draw()
    realizations_b = b.draw()
    assert len(realizations_a) == len(realizations_b)
    for ra, rb in zip(realizations_a, realizations_b, strict=True):
        np.testing.assert_array_equal(ra, rb)


# ---------------------------------------------------------------------------
# with_data convenience function
# ---------------------------------------------------------------------------
def test_with_data_delegates_to_method() -> None:
    """The ``with_data`` free function delegates to the DGP's method."""

    obs1 = np.array([[1.0, 2.0]])
    obs2 = np.array([[3.0, 4.0]])
    dgp = EmpiricalDGP(observation=obs1)

    new = with_data(dgp, obs2)

    assert new.data is obs2
    assert dgp.data is obs1


def test_with_data_raises_for_dgp_without_method() -> None:
    """The free function errors if the DGP doesn't expose ``with_data``."""

    class _BareDGP:
        @property
        def data(self):
            return None

        def draw(self, size=None):
            return None

    dgp = _BareDGP()
    # Sanity: it is still a valid DataGeneratingProcess.
    assert isinstance(dgp, DataGeneratingProcess)
    with pytest.raises(TypeError, match="does not expose a with_data method"):
        with_data(dgp, "anything")


# ---------------------------------------------------------------------------
# Two-stage Protocol conformance
# ---------------------------------------------------------------------------
def test_two_stage_dgp_satisfies_protocol() -> None:
    """A TwoStageDGP is itself a DataGeneratingProcess by structural typing."""

    outer = EmpiricalDGP(observation=np.array([[1.0]]))

    def inner(c, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(1, 1),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner)
    assert isinstance(ts, DataGeneratingProcess)
