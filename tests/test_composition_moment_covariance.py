"""Analytic cluster-robust moment-covariance on :class:`TwoStageDGP`.

Exercises the architectural goal of issue #4: ``TwoStageDGP``
encapsulates its cluster structure (via ``outer_observation`` +
``inner_observations``) and exposes ``_sd_cluster_score_blocks`` so
:meth:`SampleDistribution.moment_covariance` returns the analytic
cluster-robust ``hat Omega`` without falling back to Monte Carlo
over ``draw()``.

Three regimes are exercised:

- Empirical inner -- the composite's analytic ``hat Omega`` matches
  a flat ``EmpiricalDGP(stitched, sampling=ClusteredSampling(ids))``
  on the same data.
- Parametric inner with analytic ``expect`` -- the composite
  dispatches into the inner's ``_sd_within_cluster_block`` and
  returns the between-cluster covariance term; we exercise the path
  and check structural properties rather than tie to a known numeric
  value.
- Degraded modes -- no ``inner_observations``, custom inner without
  the hook, etc. -- falling back through ``moment_covariance``'s MC
  path or raising at ``cluster_score_blocks``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as st

from dgp_protocol import (
    AnalyticUnavailable,
    ClusteredSampling,
    EmpiricalDGP,
    IIDSampling,
    ParametricDGP,
    TwoStageDGP,
)


def _gi_identity(theta, observation):
    """Trivial moment function: g_i(theta, X_i) = X_i."""

    del theta
    return np.asarray(observation, dtype=float)


# ---------------------------------------------------------------------------
# Empirical-inner equivalence: TwoStageDGP <-> EmpiricalDGP+ClusteredSampling
# ---------------------------------------------------------------------------
def _make_clustered_inner_observations(seed: int = 2029):
    """Build inner_observations + matching cluster_ids for the flat view."""

    rng = np.random.default_rng(seed)
    sizes = [3, 5, 4, 6, 5]  # heterogeneous cluster sizes
    G = len(sizes)
    cluster_offsets = 0.8 * rng.standard_normal(size=(G, 2))
    inner_observations = []
    for c, n_c in enumerate(sizes):
        block = cluster_offsets[c] + 0.6 * rng.standard_normal(size=(n_c, 2))
        inner_observations.append(block)
    stitched = np.vstack(inner_observations)
    cluster_ids = np.repeat(np.arange(G), sizes)
    outer_observation = cluster_offsets  # one row per cluster
    return outer_observation, inner_observations, stitched, cluster_ids


def test_twostage_empirical_inner_matches_clustered_empirical() -> None:
    """``TwoStageDGP`` with EmpiricalDGP inner == ``EmpiricalDGP+ClusteredSampling``.

    Same observed data, same moment function: the cluster-robust
    ``hat Omega`` produced by the composite (through its analytic
    ``_sd_cluster_score_blocks``) must equal the one produced by a
    flat ``EmpiricalDGP`` carrying matching ``ClusteredSampling``.
    """

    outer_obs, inner_obs, stitched, cluster_ids = _make_clustered_inner_observations()

    def empirical_inner(chars, rng):
        # The inner DGP is an empty-shaped EmpiricalDGP; the composite
        # will .with_data(inner_obs[c]) it before asking for
        # _sd_within_cluster_block.  The factory just needs to return
        # the right inner type.
        del chars
        return EmpiricalDGP(
            observation=np.empty((0, stitched.shape[1])), seed=0
        ).with_rng(rng)

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=empirical_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
        seed=0,
    )

    flat = EmpiricalDGP(
        observation=stitched,
        sampling=ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )

    omega_composite = ts.sample_distribution.moment_covariance(
        theta=None, gi=_gi_identity
    )
    omega_flat = flat.sample_distribution.moment_covariance(theta=None, gi=_gi_identity)

    np.testing.assert_allclose(omega_composite, omega_flat, rtol=1e-10, atol=1e-12)


def test_twostage_empirical_inner_fallback_without_hook_matches() -> None:
    """Direct row-sum fallback path (no ``_sd_within_cluster_block`` on inner)
    produces the same analytic ``hat Omega`` as the hook-driven path.

    Construct without ``outer_observation`` so the composite skips
    inner-DGP construction entirely and falls back to a direct
    moment-row-sum on the observed inner block.  Result should still
    match the cluster-robust sandwich on the stitched data.
    """

    outer_obs, inner_obs, stitched, cluster_ids = _make_clustered_inner_observations()

    def some_inner(chars, rng):  # never called on the fallback path
        del chars, rng
        raise AssertionError("inner factory should not be invoked")

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=some_inner,
        outer_observation=None,  # forces direct-rows fallback
        inner_observations=inner_obs,
        seed=0,
    )

    flat = EmpiricalDGP(
        observation=stitched,
        sampling=ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )

    omega_composite = ts.sample_distribution.moment_covariance(
        theta=None, gi=_gi_identity
    )
    omega_flat = flat.sample_distribution.moment_covariance(theta=None, gi=_gi_identity)

    np.testing.assert_allclose(omega_composite, omega_flat, rtol=1e-10, atol=1e-12)


def test_twostage_iid_limit_when_each_cluster_has_one_row() -> None:
    """Singleton clusters collapse to the IID outer-product formula."""

    rng = np.random.default_rng(0)
    G, p = 12, 3
    outer_obs = rng.standard_normal(size=(G, p))
    inner_obs = [outer_obs[c : c + 1] for c in range(G)]
    stitched = np.vstack(inner_obs)

    def empirical_inner(chars, rng_c):
        del chars
        return EmpiricalDGP(observation=np.empty((0, p)), seed=0).with_rng(rng_c)

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=empirical_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
        seed=0,
    )
    iid_flat = EmpiricalDGP(observation=stitched, sampling=IIDSampling(), seed=0)

    omega_composite = ts.sample_distribution.moment_covariance(
        theta=None, gi=_gi_identity
    )
    omega_iid = iid_flat.sample_distribution.moment_covariance(
        theta=None, gi=_gi_identity
    )

    np.testing.assert_allclose(omega_composite, omega_iid, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Parametric-inner path: analytic μ_c via inner.expect()
# ---------------------------------------------------------------------------
def test_twostage_parametric_inner_uses_analytic_expect() -> None:
    """Parametric inner with analytic ``expect`` is exercised by the composite.

    The composite asks the parametric inner for its within-cluster
    moment sum ``n_c · μ_c(θ)``; for a scipy.stats normal inner with
    loc=cluster_chars and gi=identity, ``μ_c = cluster_chars``.  We
    check the resulting blocks have the expected structural form
    (between-cluster variance only; within-cluster variance is the
    documented residual term -- see TwoStageDGP._sd_cluster_score_blocks
    docstring caveat).
    """

    outer_obs = np.array([[1.0], [2.0], [3.0], [4.0]])
    sizes = [3, 3, 3, 3]
    inner_obs = [np.zeros((n, 1)) for n in sizes]  # values don't matter for analytic

    def parametric_inner(chars, rng):
        # Inner distribution: N(loc=chars[0], scale=1).  expect() is analytic
        # via scipy.stats.norm.expect, so the composite uses the between-
        # cluster contribution n_c * mu_c = n_c * chars[0].
        del rng
        return ParametricDGP(
            distribution=st.norm(loc=float(chars[0]), scale=1.0),
            default_shape=(1,),
        )

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=parametric_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
        seed=0,
    )

    blocks = ts.sample_distribution.cluster_score_blocks(theta=None, gi=_gi_identity)

    # Expected raw cluster sums S_c = n_c * mu_c = 3 * c_chars.
    raw_sums = np.array([sizes[c] * outer_obs[c, 0] for c in range(len(sizes))])
    N = sum(sizes)
    g_bar = raw_sums.sum() / N
    expected_blocks = (raw_sums - np.array(sizes) * g_bar) / np.sqrt(N)
    np.testing.assert_allclose(
        blocks.reshape(-1), expected_blocks, rtol=1e-10, atol=1e-12
    )


class _MultivariateDirac:
    """Tiny duck-typed multivariate distribution exposing ``expect``.

    Mass concentrated at ``mean``; ``rvs`` returns ``mean`` repeated,
    ``expect(func)`` evaluates ``func(mean)``.  Just enough surface
    for :class:`ParametricDGP` to dispatch through ``expect`` and
    exercise the multivariate analytic-inner path.
    """

    def __init__(self, mean):
        self.mean = np.asarray(mean, dtype=float)

    def rvs(self, size=None, random_state=None):
        del random_state
        n = 1 if size in (None, ()) else (size if isinstance(size, int) else size[0])
        return np.broadcast_to(self.mean, (n, self.mean.shape[0])).copy()

    def expect(self, func):
        return np.asarray(func(self.mean), dtype=float)


def test_twostage_parametric_inner_omega_is_psd_and_positive() -> None:
    """Sanity: the assembled ``hat Omega`` is symmetric PSD and non-zero.

    Uses a duck-typed multivariate distribution that exposes
    ``expect``; the composite's analytic path then evaluates
    ``mu_c = mean_c`` per cluster and forms the between-cluster
    sandwich without consulting the (all-zero) observed blocks.
    """

    outer_obs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    sizes = [2, 4, 3, 5]
    inner_obs = [np.zeros((n, 2)) for n in sizes]

    def parametric_inner(chars, rng):
        del rng
        return ParametricDGP(
            distribution=_MultivariateDirac(mean=chars),
            default_shape=(1,),
        )

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=parametric_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
        seed=0,
    )

    omega = ts.sample_distribution.moment_covariance(theta=None, gi=_gi_identity)
    np.testing.assert_allclose(omega, omega.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(omega)
    assert np.all(eigs >= -1e-10), f"non-PSD eigenvalues: {eigs}"
    # Between-cluster contribution non-trivial -- some eigenvalue > 0.
    assert np.max(eigs) > 1e-6


# ---------------------------------------------------------------------------
# Degraded paths: no inner_observations / no analytic hook -> MC or raise.
# ---------------------------------------------------------------------------
def test_cluster_score_blocks_raises_without_inner_observations() -> None:
    """``cluster_score_blocks`` raises ``AnalyticUnavailable`` when unbound."""

    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0]]), seed=0)

    def inner(chars, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(3, 1),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)  # no observations bound
    with pytest.raises(AnalyticUnavailable, match="inner_observations"):
        ts.sample_distribution.cluster_score_blocks(theta=None, gi=_gi_identity)


def test_moment_covariance_mc_fallback_when_no_observations() -> None:
    """Without observations, ``moment_covariance`` falls back to MC over draw()."""

    outer = EmpiricalDGP(observation=np.array([[0.0], [0.0], [0.0]]), seed=0)

    def inner(chars, rng):
        del chars
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(5, 1),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)
    val = ts.sample_distribution.moment_covariance(
        theta=None,
        gi=_gi_identity,
        atol=0.5,
        rtol=0.0,
        max_its=2_000,
        batch_size=100,
    )
    val_scalar = float(np.asarray(val).flat[0])
    assert np.isfinite(val_scalar)
    assert val_scalar > 0.0


def test_twostage_protocol_conformance_with_observations() -> None:
    """A TwoStageDGP with stitched observations remains a DataGeneratingProcess."""

    from dgp_protocol import DataGeneratingProcess

    outer_obs, inner_obs, _, _ = _make_clustered_inner_observations()

    def some_inner(chars, rng):
        del chars
        return EmpiricalDGP(observation=np.empty((0, 2)), seed=0).with_rng(rng)

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=some_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
    )
    assert isinstance(ts, DataGeneratingProcess)
    np.testing.assert_array_equal(ts.data, np.vstack(inner_obs))


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------
def test_twostage_validates_length_mismatch() -> None:
    """Mismatched ``outer_observation`` / ``inner_observations`` -> ValueError."""

    outer_obs = np.array([[1.0], [2.0], [3.0]])
    inner_obs = [np.zeros((2, 1)), np.zeros((3, 1))]  # only 2 entries, not 3

    def inner(chars, rng):
        del chars
        return EmpiricalDGP(observation=np.empty((0, 1)), seed=0).with_rng(rng)

    with pytest.raises(ValueError, match="they must agree"):
        TwoStageDGP(
            outer=EmpiricalDGP(observation=outer_obs, seed=0),
            inner=inner,
            outer_observation=outer_obs,
            inner_observations=inner_obs,
        )


# ---------------------------------------------------------------------------
# Pickle preservation of the new fields
# ---------------------------------------------------------------------------
def test_twostage_pickle_preserves_new_fields() -> None:
    """Round-trip through pickle preserves outer/inner observations and stitch."""

    import pickle

    outer_obs, inner_obs, _, _ = _make_clustered_inner_observations()

    def empirical_inner(chars, rng):
        del chars
        return EmpiricalDGP(observation=np.empty((0, 2)), seed=0).with_rng(rng)

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=empirical_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
        seed=42,
    )
    ts2 = pickle.loads(pickle.dumps(ts))

    np.testing.assert_array_equal(ts2.outer_observation, outer_obs)
    assert len(ts2.inner_observations) == len(inner_obs)
    for a, b in zip(ts2.inner_observations, inner_obs, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ts2.data, np.vstack(inner_obs))


# ---------------------------------------------------------------------------
# with_data preservation
# ---------------------------------------------------------------------------
def test_with_data_preserves_outer_observation_by_default() -> None:
    """``with_data(new_inner)`` keeps the existing ``outer_observation``."""

    outer_obs, inner_obs, _, _ = _make_clustered_inner_observations()
    new_inner_obs = [np.zeros((n, inner_obs[0].shape[1])) for n in [2, 3, 1, 4, 2]]

    def empirical_inner(chars, rng):
        del chars
        return EmpiricalDGP(observation=np.empty((0, 2)), seed=0).with_rng(rng)

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=empirical_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
    )
    ts2 = ts.with_data(new_inner_obs)
    np.testing.assert_array_equal(ts2.outer_observation, outer_obs)
    np.testing.assert_array_equal(ts2.data, np.vstack(new_inner_obs))


def test_with_data_can_override_outer_observation() -> None:
    """``with_data(new_inner, outer_observation=...)`` rebinds both."""

    outer_obs, inner_obs, _, _ = _make_clustered_inner_observations()

    def empirical_inner(chars, rng):
        del chars
        return EmpiricalDGP(observation=np.empty((0, 2)), seed=0).with_rng(rng)

    ts = TwoStageDGP(
        outer=EmpiricalDGP(observation=outer_obs, seed=0),
        inner=empirical_inner,
        outer_observation=outer_obs,
        inner_observations=inner_obs,
    )
    new_outer = outer_obs + 100.0
    ts2 = ts.with_data(inner_obs, outer_observation=new_outer)
    np.testing.assert_array_equal(ts2.outer_observation, new_outer)
