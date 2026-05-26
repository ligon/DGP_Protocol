"""Tests for the D-side surface (``dgp.sample_distribution``).

Covers :class:`~dgp_protocol.SampleDistribution` and its three
methods (:meth:`expect`, :meth:`cov`, :meth:`moment_covariance`)
across the three concrete containers.
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
    NumericalWarning,
    ParametricDGP,
    SampleDistribution,
    TwoStageDGP,
)


# ---------------------------------------------------------------------------
# Property exists on all containers
# ---------------------------------------------------------------------------
def test_sample_distribution_property_on_empirical() -> None:
    dgp = EmpiricalDGP(observation=np.arange(12).reshape(4, 3).astype(float))
    sd = dgp.sample_distribution
    assert isinstance(sd, SampleDistribution)
    assert sd.dgp is dgp


def test_sample_distribution_property_on_parametric_generator() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(5, 2),
        seed=0,
    )
    sd = dgp.sample_distribution
    assert isinstance(sd, SampleDistribution)
    assert sd.dgp is dgp


def test_sample_distribution_property_on_parametric_distribution() -> None:
    dgp = ParametricDGP(distribution=st.norm(), default_shape=(50,), seed=0)
    assert isinstance(dgp.sample_distribution, SampleDistribution)


def test_sample_distribution_property_on_twostage() -> None:
    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0]]), seed=0)

    def inner(chars, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(3, 2),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)
    assert isinstance(ts.sample_distribution, SampleDistribution)


# ---------------------------------------------------------------------------
# SampleDistribution.expect: dataset-level MC
# ---------------------------------------------------------------------------
def test_sd_expect_for_empirical_iid_sample_mean() -> None:
    """E_DGP[X̄] under iid bootstrap is the data column-mean (exactly in expectation)."""

    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs, seed=0)

    def stat(realization):
        return realization.mean(axis=0)

    # MC over bootstrap resamples; converges to obs.mean(axis=0).
    val = dgp.sample_distribution.expect(
        stat, atol=0.1, rtol=0.0, max_its=5_000, batch_size=100
    )
    np.testing.assert_allclose(val, obs.mean(axis=0), atol=0.3)


def test_sd_expect_for_parametric_distribution() -> None:
    """E_DGP[X̄] for scipy.norm should converge to ~0."""

    dgp = ParametricDGP(distribution=st.norm(loc=0, scale=1), default_shape=(50,))

    def stat(realization):
        return float(realization.mean())

    val = dgp.sample_distribution.expect(stat, atol=0.05, rtol=0.0, max_its=5_000)
    assert abs(val) < 0.1


def test_sd_expect_warns_on_max_its() -> None:
    """Adaptive MC at the dataset level emits NumericalWarning on budget exhaust."""

    dgp = ParametricDGP(distribution=st.norm(), default_shape=(50,), seed=0)
    with pytest.warns(NumericalWarning, match="max_its"):
        dgp.sample_distribution.expect(
            lambda r: float(r.mean()), atol=1e-9, rtol=0.0, max_its=200
        )


# ---------------------------------------------------------------------------
# SampleDistribution.cov: two-pass MC
# ---------------------------------------------------------------------------
def test_sd_cov_scalar_stat_returns_scalar() -> None:
    """Var of the sample mean of a unit-normal draw of size 50 is ~ 1/50."""

    dgp = ParametricDGP(distribution=st.norm(), default_shape=(50,), seed=0)
    val = dgp.sample_distribution.cov(
        lambda r: float(r.mean()), atol=0.005, rtol=0.0, max_its=5_000, batch_size=200
    )
    # True variance of the sample mean is 1/50 = 0.02; MC should get close.
    assert abs(float(val) - 0.02) < 0.01


def test_sd_cov_vector_stat_returns_matrix() -> None:
    """For a vector statistic, sd.cov returns a square covariance matrix."""

    mvn = st.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2))
    dgp = ParametricDGP(distribution=mvn, default_shape=(100,), seed=0)
    # Per-realization column-mean is a (2,)-vector.
    val = dgp.sample_distribution.cov(
        lambda r: r.mean(axis=0),
        atol=0.002,
        rtol=0.0,
        max_its=3_000,
        batch_size=100,
    )
    assert val.shape == (2, 2)
    # Cov of sample mean of bivariate iid N(0, I_2) of size 100 is (1/100)*I.
    np.testing.assert_allclose(val, np.eye(2) / 100, atol=0.005)


# ---------------------------------------------------------------------------
# SampleDistribution.moment_covariance: MC fallback
# ---------------------------------------------------------------------------
def test_sd_moment_covariance_mc_fallback_on_parametric() -> None:
    """Var_DGP[g_bar_N(theta)] for gi(theta, X) = X under iid N(0,1).

    g_bar_N = (1/sqrt(N)) sum_i X_i has variance 1 by construction.
    """

    dgp = ParametricDGP(distribution=st.norm(), default_shape=(50,), seed=0)

    def gi(theta, realization):
        # Single-moment: just return X as a (N, 1) matrix.
        return np.asarray(realization).reshape(-1, 1)

    val = dgp.sample_distribution.moment_covariance(
        theta=None, gi=gi, atol=0.05, rtol=0.0, max_its=3_000, batch_size=100
    )
    # 1x1 matrix or scalar; either way ~1.
    val_scalar = float(np.asarray(val).flat[0])
    assert abs(val_scalar - 1.0) < 0.2


def test_sd_moment_covariance_analytic_hook_takes_priority() -> None:
    """If _sd_moment_covariance is defined, sample_distribution uses it.

    We define a minimal user DGP exposing the hook to verify dispatch.
    """

    class FakeDGP:
        def draw(self, size=None):
            return np.zeros((1, 1))  # not used in this test

        @property
        def sample_distribution(self):
            return SampleDistribution(self)

        def _sd_moment_covariance(self, theta, gi, **kw):
            # Return a sentinel value to prove this path was taken.
            return np.array([[42.0]])

    val = FakeDGP().sample_distribution.moment_covariance(theta=None, gi=lambda t, x: x)
    np.testing.assert_array_equal(val, np.array([[42.0]]))


def test_sd_moment_covariance_hook_analytic_unavailable_falls_back() -> None:
    """A hook that raises AnalyticUnavailable triggers MC fallback."""

    class FlakyDGP:
        def __init__(self):
            self._rng = np.random.default_rng(0)

        def draw(self, size=None):
            return self._rng.standard_normal((10, 1))

        @property
        def sample_distribution(self):
            return SampleDistribution(self)

        def _sd_moment_covariance(self, theta, gi, **kw):
            raise AnalyticUnavailable("nope")

    dgp = FlakyDGP()

    def gi(theta, realization):
        return np.asarray(realization)

    val = dgp.sample_distribution.moment_covariance(
        theta=None, gi=gi, atol=0.5, rtol=0.0, max_its=2_000, batch_size=100
    )
    # 10-row standard normal draw -> g_bar_N = sum / sqrt(10) ~ N(0,1).
    val_scalar = float(np.asarray(val).flat[0])
    assert 0.3 < val_scalar < 2.0  # generous; just verify finite/positive/MC-ish


# ---------------------------------------------------------------------------
# SampleDistribution works for TwoStageDGP (where P-side refused)
# ---------------------------------------------------------------------------
def test_sd_expect_for_twostage_with_flat_aggregator() -> None:
    """sample_distribution.expect handles TwoStageDGP draws via a user aggregator."""

    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0], [3.0]]), seed=0)

    def inner(chars, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(4, 2),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)

    def flat_mean(lst):
        return np.vstack(lst).mean(axis=0)

    val = ts.sample_distribution.expect(
        flat_mean, atol=0.2, rtol=0.0, max_its=3_000, batch_size=100
    )
    assert val.shape == (2,)
    np.testing.assert_allclose(val, np.zeros(2), atol=0.4)


# ---------------------------------------------------------------------------
# EmpiricalDGP under ClusteredSampling: cluster-block bootstrap path
# ---------------------------------------------------------------------------
def _make_clustered_observation(seed: int = 2029):
    """Synthesize an observation with positive intra-cluster correlation.

    10 clusters of 5 rows each (50 rows total, 2 columns).  Each cluster
    carries a cluster-level offset on top of within-cluster iid noise,
    so rows within a cluster are positively correlated; rows across
    clusters are independent.
    """

    rng = np.random.default_rng(seed)
    n_clusters = 10
    rows_per_cluster = 5
    cluster_offsets = 0.8 * rng.standard_normal(size=(n_clusters, 2))
    within_noise = 0.6 * rng.standard_normal(size=(n_clusters * rows_per_cluster, 2))
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    observation = cluster_offsets[cluster_ids] + within_noise
    return observation, cluster_ids


def test_sd_expect_for_empirical_cluster_bootstrap() -> None:
    """sample_distribution.expect works for ClusteredSampling EmpiricalDGP.

    The cluster bootstrap is unbiased for the sample mean, so MC of
    the column-mean statistic should converge to the data column-mean.
    """

    obs, cluster_ids = _make_clustered_observation()
    cdgp = EmpiricalDGP(
        observation=obs,
        sampling=ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )
    sample_mean = cdgp.sample_distribution.expect(
        lambda r: r.mean(axis=0),
        atol=0.05,
        rtol=0.0,
        max_its=3000,
        batch_size=100,
    )
    np.testing.assert_allclose(sample_mean, obs.mean(axis=0), atol=0.15)


def test_sd_moment_covariance_cluster_robust_exceeds_iid() -> None:
    """Cluster-robust omega_hat is visibly larger than iid omega_hat
    when intra-cluster correlation is positive.

    Same observation, two sampling designs: the cluster-bootstrap
    moment-covariance preserves the within-cluster correlation, so the
    moment-vector sampling variance is larger than what the
    (mis-specified) iid bootstrap would estimate.
    """

    obs, cluster_ids = _make_clustered_observation()
    iid_dgp = EmpiricalDGP(observation=obs, seed=0)
    clustered_dgp = EmpiricalDGP(
        observation=obs,
        sampling=ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )

    theta = obs.mean(axis=0)

    def gi(theta, X):
        return X - theta

    omega_iid = iid_dgp.sample_distribution.moment_covariance(
        theta=theta, gi=gi, atol=0.1, rtol=0.0, max_its=2000, batch_size=100
    )
    omega_cluster = clustered_dgp.sample_distribution.moment_covariance(
        theta=theta, gi=gi, atol=0.1, rtol=0.0, max_its=2000, batch_size=100
    )

    # Both 2x2 positive-semidefinite; compare traces.  For this
    # synthesized data the cluster-robust trace is empirically ~6x
    # the iid trace; 1.5x is a generous lower bound.
    trace_iid = float(np.trace(np.asarray(omega_iid)))
    trace_cluster = float(np.trace(np.asarray(omega_cluster)))
    assert trace_cluster > 1.5 * trace_iid, (
        f"expected cluster-robust trace > 1.5 * iid trace; "
        f"got cluster={trace_cluster:.3f}, iid={trace_iid:.3f}"
    )


# ---------------------------------------------------------------------------
# Analytic _sd_moment_covariance on EmpiricalDGP / ParametricDGP
# (Phase A wiring: dispatches to SamplingDesign.moment_covariance_estimator)
# ---------------------------------------------------------------------------
def test_empirical_iid_sd_moment_covariance_is_analytic() -> None:
    """EmpiricalDGP._sd_moment_covariance uses the closed-form formula,
    no MC, no NumericalWarning even at impossibly-tight tolerance."""

    obs, _ = _make_clustered_observation()
    iid_dgp = EmpiricalDGP(observation=obs, seed=0)
    theta = obs.mean(axis=0)

    def gi(theta, X):
        return X - theta

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", NumericalWarning)
        # If MC fired, we'd warn (or hit max_its) at this tolerance.
        omega = iid_dgp.sample_distribution.moment_covariance(
            theta=theta, gi=gi, atol=1e-12, rtol=0.0
        )

    # Same as calling the SamplingDesign's estimator directly.
    direct = iid_dgp.sampling.moment_covariance_estimator(
        obs, theta=theta, gi=gi, centered=True
    )
    np.testing.assert_allclose(omega, direct, atol=1e-12)


def test_empirical_clustered_sd_moment_covariance_is_analytic() -> None:
    """Same for ClusteredSampling: closed-form, no MC."""

    obs, cluster_ids = _make_clustered_observation()
    cdgp = EmpiricalDGP(
        observation=obs,
        sampling=ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )
    theta = obs.mean(axis=0)

    def gi(theta, X):
        return X - theta

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", NumericalWarning)
        omega = cdgp.sample_distribution.moment_covariance(
            theta=theta, gi=gi, atol=1e-12, rtol=0.0
        )

    direct = cdgp.sampling.moment_covariance_estimator(
        obs, theta=theta, gi=gi, centered=True
    )
    np.testing.assert_allclose(omega, direct, atol=1e-12)


def test_parametric_with_sampling_and_observation_takes_analytic() -> None:
    """ParametricDGP(sampling=..., observation=...) uses the analytic path."""

    obs, cluster_ids = _make_clustered_observation()

    def gen(rng, shape):
        return rng.standard_normal(shape)

    pdgp = ParametricDGP(
        generator=gen,
        default_shape=obs.shape,
        observation=obs,
        sampling=ClusteredSampling(cluster_ids=cluster_ids),
        seed=0,
    )
    theta = obs.mean(axis=0)

    def gi(theta, X):
        return X - theta

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", NumericalWarning)
        omega = pdgp.sample_distribution.moment_covariance(
            theta=theta, gi=gi, atol=1e-12, rtol=0.0
        )

    # Same answer as if we'd wrapped the observation in EmpiricalDGP.
    edgp = EmpiricalDGP(
        observation=obs, sampling=ClusteredSampling(cluster_ids=cluster_ids)
    )
    direct = edgp.sampling.moment_covariance_estimator(
        obs, theta=theta, gi=gi, centered=True
    )
    np.testing.assert_allclose(omega, direct, atol=1e-12)


def test_parametric_without_sampling_falls_back_to_mc() -> None:
    """ParametricDGP(sampling=None) raises AnalyticUnavailable -> MC fallback."""

    dgp = ParametricDGP(distribution=st.norm(), default_shape=(50,), seed=0)

    def gi(theta, realization):
        return np.asarray(realization).reshape(-1, 1)

    # No sampling => AnalyticUnavailable from the hook => SD MCs.
    val = dgp.sample_distribution.moment_covariance(
        theta=None, gi=gi, atol=0.1, rtol=0.0, max_its=2_000, batch_size=100
    )
    val_scalar = float(np.asarray(val).flat[0])
    # Var(g_bar_N) for g_i = X_i iid N(0,1) is 1.
    assert abs(val_scalar - 1.0) < 0.3


def test_parametric_with_sampling_but_no_observation_falls_back_to_mc() -> None:
    """ParametricDGP(sampling=..., observation=None) also MCs."""

    dgp = ParametricDGP(
        distribution=st.norm(),
        default_shape=(50,),
        sampling=IIDSampling(),
        observation=None,
        seed=0,
    )

    def gi(theta, realization):
        return np.asarray(realization).reshape(-1, 1)

    # observation=None -> AnalyticUnavailable -> MC.
    val = dgp.sample_distribution.moment_covariance(
        theta=None, gi=gi, atol=0.1, rtol=0.0, max_its=2_000, batch_size=100
    )
    val_scalar = float(np.asarray(val).flat[0])
    assert abs(val_scalar - 1.0) < 0.3


def test_parametric_sampling_field_round_trips_pickle() -> None:
    """The new sampling field survives pickle (incl. cloudpickle path)."""

    import pickle

    obs = np.arange(40).reshape(10, 4).astype(float)
    ids = np.array([0] * 5 + [1] * 5)
    pdgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=obs.shape,
        observation=obs,
        sampling=ClusteredSampling(cluster_ids=ids),
        seed=0,
    )
    pdgp2 = pickle.loads(pickle.dumps(pdgp))
    assert isinstance(pdgp2.sampling, ClusteredSampling)
    np.testing.assert_array_equal(pdgp2.sampling.cluster_ids, ids)
    # The analytic path works post-pickle.
    omega = pdgp2.sample_distribution.moment_covariance(
        theta=obs.mean(axis=0), gi=lambda t, X: X - t
    )
    assert omega.shape == (4, 4)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------
def test_sd_repr_mentions_underlying_dgp() -> None:
    dgp = EmpiricalDGP(observation=np.array([[1.0]]))
    assert "SampleDistribution" in repr(dgp.sample_distribution)
