"""Tests for the P-side surface (per-observation marginal moments).

Covers the free functions :func:`dgp_protocol.expect`,
:func:`~dgp_protocol.mean`, :func:`~dgp_protocol.var`,
:func:`~dgp_protocol.cov` and their dispatch behaviour on the three
concrete containers.
"""

from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
from dgp_protocol import (
    AnalyticUnavailable,
    ClusteredSampling,
    EmpiricalDGP,
    NumericalWarning,
    ParametricDGP,
    TwoStageDGP,
    cov,
    expect,
    mean,
    var,
)


# ---------------------------------------------------------------------------
# Adaptive MC convergence (free function -> MC path)
# ---------------------------------------------------------------------------
def test_expect_converges_for_simple_parametric_dgp() -> None:
    """Free function expect converges via MC for a generator-only DGP."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    # generator-only DGP raises AnalyticUnavailable -> free function MCs.
    m = expect(dgp, lambda x: x, atol=0.05, rtol=0.0, max_its=10_000)
    assert abs(float(m)) < 0.1


def test_expect_emits_warning_when_max_its_reached() -> None:
    """Tight tolerance with small budget triggers NumericalWarning."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    with pytest.warns(NumericalWarning, match="max_its"):
        expect(dgp, lambda x: x, atol=1e-6, rtol=0.0, max_its=500)


def test_expect_rtol_handles_large_mean() -> None:
    """rtol scales tolerance with |mean|; large-mean cases converge fast."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: 50.0 + 0.01 * rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    m = expect(dgp, lambda x: x, atol=0.0, rtol=1e-3, max_its=10_000)
    assert abs(m - 50.0) < 0.1


# ---------------------------------------------------------------------------
# EmpiricalDGP under IIDSampling: analytic
# ---------------------------------------------------------------------------
def test_empirical_iid_mean_is_exact() -> None:
    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    np.testing.assert_array_equal(dgp.mean(), obs.mean(axis=0))
    np.testing.assert_array_equal(mean(dgp), obs.mean(axis=0))


def test_empirical_iid_var_is_exact_ddof_1() -> None:
    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    np.testing.assert_allclose(dgp.var(), obs.var(axis=0, ddof=1))
    np.testing.assert_allclose(var(dgp), obs.var(axis=0, ddof=1))


def test_empirical_iid_cov_is_exact_ddof_1() -> None:
    obs = np.random.default_rng(0).standard_normal(size=(50, 3))
    dgp = EmpiricalDGP(observation=obs)
    np.testing.assert_allclose(dgp.cov(), np.cov(obs, rowvar=False, ddof=1))


def test_empirical_iid_expect_is_exact() -> None:
    """expect(dgp, func) is the row-average for an iid empirical DGP."""

    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    # E_F̂[X_1] = mean of col 0 of obs.
    result = expect(dgp, lambda row: row[0])
    assert float(result) == obs[:, 0].mean()


def test_empirical_overrides_ignore_kwargs_silently() -> None:
    """Analytic exact methods accept (irrelevant) MC kwargs without error."""

    obs = np.arange(8).reshape(2, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    np.testing.assert_array_equal(
        dgp.mean(atol=1e-9, rtol=0.0, max_its=10), obs.mean(axis=0)
    )


# ---------------------------------------------------------------------------
# EmpiricalDGP under non-iid sampling: refuse (NotImplementedError propagates)
# ---------------------------------------------------------------------------
def test_empirical_clustered_mean_refuses() -> None:
    obs = np.arange(20).reshape(10, 2).astype(float)
    cdgp = EmpiricalDGP(
        observation=obs,
        sampling=ClusteredSampling(np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])),
    )
    with pytest.raises(NotImplementedError, match="ClusteredSampling"):
        cdgp.mean()


def test_empirical_clustered_mean_via_free_function_propagates() -> None:
    """Free function does NOT catch NotImplementedError (it's a refusal)."""

    obs = np.arange(20).reshape(10, 2).astype(float)
    cdgp = EmpiricalDGP(
        observation=obs,
        sampling=ClusteredSampling(np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])),
    )
    with pytest.raises(NotImplementedError, match="ClusteredSampling"):
        mean(cdgp)


def test_empirical_clustered_var_refuses() -> None:
    obs = np.arange(20).reshape(10, 2).astype(float)
    cdgp = EmpiricalDGP(
        observation=obs,
        sampling=ClusteredSampling(np.array([0, 0, 0, 1, 1, 2, 2, 2, 3, 3])),
    )
    with pytest.raises(NotImplementedError):
        cdgp.var()
    with pytest.raises(NotImplementedError):
        cdgp.cov()
    with pytest.raises(NotImplementedError):
        cdgp.expect(lambda row: row[0])


# ---------------------------------------------------------------------------
# ParametricDGP with scipy.stats: analytic dispatch
# ---------------------------------------------------------------------------
def test_parametric_with_scipy_norm_mean_is_analytic() -> None:
    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=2), default_shape=(100,), seed=0
    )
    assert dgp.mean() == 0.0


def test_parametric_with_scipy_norm_var_is_analytic() -> None:
    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=2), default_shape=(100,), seed=0
    )
    assert dgp.var() == 4.0


def test_parametric_with_scipy_norm_expect_uses_scipy() -> None:
    """expect(func) with scipy distribution routes to dist.expect (analytic)."""

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=1), default_shape=(100,), seed=0
    )
    val = dgp.expect(lambda x: x**2)
    np.testing.assert_allclose(val, 1.0, atol=1e-6)


def test_parametric_with_scipy_norm_expect_forwards_kwargs() -> None:
    """dist_kwargs (lb / ub) flow to scipy.expect."""

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=1), default_shape=(100,), seed=0
    )
    val = dgp.expect(lambda x: 1.0, lb=0)
    np.testing.assert_allclose(val, 0.5, atol=1e-6)


def test_parametric_with_scipy_norm_cov_raises_analytic_unavailable() -> None:
    """norm has no .cov attribute -> AnalyticUnavailable on direct method call."""

    dgp = ParametricDGP(distribution=st.norm(loc=0, scale=1), default_shape=(200,))
    with pytest.raises(AnalyticUnavailable, match="cov"):
        dgp.cov()


def test_parametric_with_scipy_norm_cov_via_free_function_does_mc() -> None:
    """Free function cov() catches AnalyticUnavailable and MCs to ~variance."""

    dgp = ParametricDGP(distribution=st.norm(loc=0, scale=1), default_shape=(200,))
    # For univariate, cov collapses to variance ~1.
    val = cov(dgp, atol=0.05, rtol=0.0, max_its=5_000)
    assert abs(float(val) - 1.0) < 0.15


def test_parametric_with_scipy_mvn_mean_uses_attribute() -> None:
    """multivariate_normal.mean is an attribute, not a method."""

    mvn = st.multivariate_normal(mean=[1.0, 2.0], cov=[[1, 0.5], [0.5, 1]])
    dgp = ParametricDGP(distribution=mvn, default_shape=(100,), seed=0)
    np.testing.assert_array_equal(dgp.mean(), np.array([1.0, 2.0]))


def test_parametric_with_scipy_mvn_cov_uses_attribute() -> None:
    mvn = st.multivariate_normal(mean=[1.0, 2.0], cov=[[1, 0.5], [0.5, 1]])
    dgp = ParametricDGP(distribution=mvn, default_shape=(100,), seed=0)
    np.testing.assert_array_equal(dgp.cov(), np.array([[1, 0.5], [0.5, 1]]))


def test_parametric_with_scipy_mvn_var_via_free_function_does_mc() -> None:
    """mvn has no .var; free function var() catches AnalyticUnavailable and MCs."""

    mvn = st.multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 4]])
    dgp = ParametricDGP(distribution=mvn, default_shape=(500,), seed=0)
    v = var(dgp, atol=0.1, rtol=0.0, max_its=5_000)
    np.testing.assert_allclose(np.asarray(v), np.array([1.0, 4.0]), atol=0.3)


def test_parametric_draws_use_distribution_rvs() -> None:
    """When distribution= is set, draw() uses distribution.rvs."""

    mvn = st.multivariate_normal(mean=[1.0, -1.0], cov=np.eye(2))
    dgp_a = ParametricDGP(distribution=mvn, default_shape=(50,), seed=42)
    dgp_b = ParametricDGP(distribution=mvn, default_shape=(50,), seed=42)
    draw_a = dgp_a.draw()
    assert draw_a.shape == (50, 2)
    np.testing.assert_array_equal(draw_a, dgp_b.draw())


# ---------------------------------------------------------------------------
# ParametricDGP with generator only: raises AnalyticUnavailable; free fn MCs
# ---------------------------------------------------------------------------
def test_parametric_generator_only_method_raises_analytic_unavailable() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(),
        seed=0,
    )
    with pytest.raises(AnalyticUnavailable, match="generator-based"):
        dgp.mean()


def test_parametric_generator_only_free_mean_does_mc() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: 3.0 + rng.standard_normal(shape),
        default_shape=(),
        seed=0,
    )
    m = mean(dgp, atol=0.05, rtol=0.0, max_its=10_000)
    assert abs(m - 3.0) < 0.1


# ---------------------------------------------------------------------------
# Duck-typed distribution dispatch
# ---------------------------------------------------------------------------
def test_user_defined_distribution_like_dispatches() -> None:
    """Any object with .rvs/.mean/.var qualifies via duck typing."""

    class MyDist:
        def rvs(self, size, random_state):
            return 7.0 * np.ones(size if size else 1)

        def mean(self):
            return 7.0

        def var(self):
            return 0.0

    dgp = ParametricDGP(distribution=MyDist(), default_shape=(5,), seed=0)
    assert dgp.mean() == 7.0
    assert dgp.var() == 0.0


# ---------------------------------------------------------------------------
# TwoStageDGP: refuses all P-side ops
# ---------------------------------------------------------------------------
def test_twostage_p_side_refuses() -> None:
    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0]]), seed=0)

    def inner(chars, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(3, 2),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)
    for op_name in ("mean", "var", "cov"):
        with pytest.raises(NotImplementedError, match="sample_distribution"):
            getattr(ts, op_name)()
    with pytest.raises(NotImplementedError, match="sample_distribution"):
        ts.expect(lambda row: row)


def test_twostage_p_side_via_free_function_propagates() -> None:
    """Free functions propagate NotImplementedError from TwoStageDGP."""

    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0]]), seed=0)

    def inner(chars, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(3, 2),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)
    with pytest.raises(NotImplementedError, match="sample_distribution"):
        mean(ts)


# ---------------------------------------------------------------------------
# AnalyticUnavailable subclass relationship
# ---------------------------------------------------------------------------
def test_analytic_unavailable_is_notimplementederror_subclass() -> None:
    """Legacy except NotImplementedError handlers still catch AnalyticUnavailable."""

    assert issubclass(AnalyticUnavailable, NotImplementedError)


# ---------------------------------------------------------------------------
# Mutual exclusivity / construction errors
# ---------------------------------------------------------------------------
def test_parametric_requires_one_of_generator_or_distribution() -> None:
    with pytest.raises(ValueError, match="neither"):
        ParametricDGP(default_shape=(10,))


def test_parametric_rejects_both_generator_and_distribution() -> None:
    def gen(rng, shape):
        return rng.standard_normal(shape)

    with pytest.raises(ValueError, match="both"):
        ParametricDGP(generator=gen, distribution=st.norm(), default_shape=(10,))


# ---------------------------------------------------------------------------
# Aggregator dispatch over return types (via free expect)
# ---------------------------------------------------------------------------
def test_aggregator_handles_numpy_scalar_returns() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    val = expect(dgp, lambda x: float(x), atol=0.05, rtol=0.0, max_its=5_000)
    assert abs(float(val)) < 0.1


def test_aggregator_handles_numpy_array_returns() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(10, 2),
        seed=0,
    )
    # func of one row (length-2 vector); result is per-coord mean of (10*MC) rows.
    val = expect(dgp, lambda row: row, atol=0.2, rtol=0.0, max_its=5_000)
    assert val.shape == (2,)
    np.testing.assert_allclose(val, np.zeros(2), atol=0.3)


def test_aggregator_handles_pandas_series_func_return() -> None:
    """func returning a Series aggregates as a Series with the row's index."""

    def gen(rng, shape):
        return rng.standard_normal(shape)

    dgp = ParametricDGP(generator=gen, default_shape=(10, 2), seed=0)

    def to_series(row):
        return pd.Series(row, index=["a", "b"])

    val = expect(dgp, to_series, atol=0.2, rtol=0.0, max_its=5_000)
    assert isinstance(val, pd.Series)
    assert list(val.index) == ["a", "b"]


def test_aggregator_rejects_unsupported_type() -> None:
    """Returning a python list from func raises NotImplementedError."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(5,),
        seed=0,
    )
    with pytest.raises(NotImplementedError, match="cannot aggregate"):
        expect(dgp, lambda x: [float(x), float(x)])


# ---------------------------------------------------------------------------
# kwargs handling on the free functions
# ---------------------------------------------------------------------------
def test_mc_fallback_warns_on_unhandled_dist_kwargs() -> None:
    """Backend kwargs to a generator-only DGP raise NumericalWarning."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    with pytest.warns(NumericalWarning, match="lb"):
        expect(dgp, lambda x: x, lb=0, atol=0.1, max_its=2_000)


def test_analytic_dispatch_does_not_warn_when_kwargs_consumed() -> None:
    """No warning when scipy.expect successfully consumes lb/ub."""

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=1), default_shape=(50,), seed=0
    )
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", NumericalWarning)
        expect(dgp, lambda x: 1.0, lb=0)
