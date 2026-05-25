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
    returns a DGP for each cluster.
    """

    rng = np.random.default_rng(0)

    # Outer: 3 clusters, each characterised by a single scalar.
    cluster_chars = np.array([[1.0], [2.0], [3.0]])
    outer = EmpiricalDGP(observation=cluster_chars)

    # Inner: each cluster has an iid Gaussian draw with std equal to
    # the cluster's scalar characteristic.
    def make_inner(chars):
        sigma = float(chars[0])
        return ParametricDGP(
            generator=lambda rng, shape: sigma * rng.standard_normal(shape),
            default_shape=(5, 2),
        )

    ts = TwoStageDGP(outer=outer, inner=make_inner)
    realization = ts.draw(rng=rng)

    assert isinstance(realization, list)
    assert len(realization) == 3  # one per cluster
    for cluster_draw in realization:
        assert cluster_draw.shape == (5, 2)


def test_two_stage_inner_called_with_cluster_chars() -> None:
    """The ``inner`` callable receives the cluster row each time."""

    cluster_chars = np.array([[10.0], [20.0]])
    outer = EmpiricalDGP(observation=cluster_chars)

    seen_chars = []

    def make_inner(chars):
        seen_chars.append(chars.copy())
        # Inner is degenerate: always returns zeros.
        return ParametricDGP(
            generator=lambda rng, shape: np.zeros(shape),
            default_shape=(1, 1),
        )

    ts = TwoStageDGP(outer=outer, inner=make_inner)
    # Set a fixed RNG for outer so we know which clusters get drawn.
    rng_local = np.random.default_rng(42)
    _ = ts.draw(rng=rng_local)

    # ``inner`` was called once per outer-drawn cluster.
    assert len(seen_chars) == cluster_chars.shape[0]
    # Each ``chars`` is one row of the outer's draw, so values are
    # in {10.0, 20.0}.
    seen_values = {float(c[0]) for c in seen_chars}
    assert seen_values.issubset({10.0, 20.0})


def test_two_stage_recursive_composition() -> None:
    """A three-stage composition: region -> cluster-within-region -> obs.

    Demonstrates that ``inner`` can return another :class:`TwoStageDGP`.
    """

    rng = np.random.default_rng(0)

    # Top-level: 2 regions, each characterised by a scalar.
    region_chars = np.array([[100.0], [200.0]])
    region_dgp = EmpiricalDGP(observation=region_chars)

    def make_cluster_dgp_within_region(region_chars_row):
        # Each region has 3 clusters, each characterised by the
        # region characteristic times a small jitter.
        base = float(region_chars_row[0])
        cluster_chars = np.array([[base + 1], [base + 2], [base + 3]])
        cluster_dgp = EmpiricalDGP(observation=cluster_chars)

        def make_observation_dgp(cluster_chars_row):
            return ParametricDGP(
                generator=lambda rng, shape: np.zeros(shape) + cluster_chars_row[0],
                default_shape=(1, 1),
            )

        return TwoStageDGP(outer=cluster_dgp, inner=make_observation_dgp)

    three_stage = TwoStageDGP(outer=region_dgp, inner=make_cluster_dgp_within_region)

    realization = three_stage.draw(rng=rng)
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

    def make_inner(c):
        return ParametricDGP(
            generator=lambda rng, shape: rng.standard_normal(shape),
            default_shape=(1, 1),
        )

    ts1 = TwoStageDGP(outer=outer, inner=make_inner)
    fake_realization = ("placeholder", "object")
    ts2 = ts1.with_data(fake_realization)

    assert ts2 is not ts1
    assert ts2.outer is ts1.outer
    assert ts2.inner is ts1.inner
    assert ts2.data == fake_realization
    assert ts1.data is None  # unchanged


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

        def draw(self, size=None, *, rng):
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
    inner = lambda c: ParametricDGP(  # noqa: E731
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(1, 1),
    )
    ts = TwoStageDGP(outer=outer, inner=inner)
    assert isinstance(ts, DataGeneratingProcess)
