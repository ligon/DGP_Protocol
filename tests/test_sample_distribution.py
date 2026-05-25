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
    EmpiricalDGP,
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
# Repr
# ---------------------------------------------------------------------------
def test_sd_repr_mentions_underlying_dgp() -> None:
    dgp = EmpiricalDGP(observation=np.array([[1.0]]))
    assert "SampleDistribution" in repr(dgp.sample_distribution)
