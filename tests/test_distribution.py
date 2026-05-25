"""Tests for the :class:`DistributionalFeatures` mixin.

Covers:

- Adaptive MC convergence (atol / rtol numpy.allclose-style, max_its
  warning).
- :class:`EmpiricalDGP` analytic overrides (mean / var / cov exact).
- :class:`ParametricDGP` with ``generator=`` (pure MC fallback) vs
  ``distribution=`` (analytic dispatch via duck-typed attribute /
  method lookup).
- scipy.stats analytic backends: univariate (callable .mean / .var /
  .expect) and multivariate (attribute .mean / .cov).
- Aggregator dispatch over numpy arrays, scalars, pandas DataFrames
  and Series.
- :class:`TwoStageDGP`: mean / var / cov raise NotImplementedError;
  expect works with a user-supplied flat aggregator.
- ``dist_kwargs`` passthrough to scipy.expect (``lb`` / ``ub``) and
  warning on MC fallback.
- Mutual exclusivity of ``generator`` and ``distribution`` on
  :class:`ParametricDGP`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
from dgp_protocol import (
    DistributionalFeatures,
    EmpiricalDGP,
    NumericalWarning,
    ParametricDGP,
    TwoStageDGP,
)


# ---------------------------------------------------------------------------
# Adaptive MC convergence
# ---------------------------------------------------------------------------
def test_expect_converges_for_simple_parametric_dgp() -> None:
    """expect terminates well under max_its for a tight-enough tol.

    A ParametricDGP that draws a single standard-normal scalar per
    realization; the MC mean has SE = 1/sqrt(n), so atol=0.05 needs
    ~400 samples -- well under max_its.
    """

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    mean = dgp.expect(lambda x: x, atol=0.05, rtol=0.0, max_its=10_000)
    # Within +/- 0.1 of the true mean (0) with very high probability.
    assert abs(mean) < 0.1


def test_expect_emits_warning_when_max_its_reached() -> None:
    """Tight tolerance with small budget triggers NumericalWarning."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    with pytest.warns(NumericalWarning, match="max_its"):
        # atol=1e-6 needs ~1e12 samples; we cap at 500.
        dgp.expect(lambda x: x, atol=1e-6, rtol=0.0, max_its=500)


def test_expect_rtol_handles_large_mean() -> None:
    """rtol scales tolerance with |mean|; large-mean cases converge fast."""

    # Constant-50 with tiny noise: mean is ~50, std ~0.01.
    dgp = ParametricDGP(
        generator=lambda rng, shape: 50.0 + 0.01 * rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    mean = dgp.expect(lambda x: x, atol=0.0, rtol=1e-3, max_its=10_000)
    # rtol=1e-3 means SE/|mean| < 1e-3, so accuracy is ~50 * 1e-3 = 0.05.
    assert abs(mean - 50.0) < 0.1


# ---------------------------------------------------------------------------
# EmpiricalDGP analytic overrides
# ---------------------------------------------------------------------------
def test_empirical_mean_is_exact() -> None:
    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    np.testing.assert_array_equal(dgp.mean(), obs.mean(axis=0))


def test_empirical_var_is_exact_with_ddof_1() -> None:
    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    np.testing.assert_allclose(dgp.var(), obs.var(axis=0, ddof=1))


def test_empirical_cov_is_exact_with_ddof_1() -> None:
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(size=(50, 3))
    dgp = EmpiricalDGP(observation=obs)
    np.testing.assert_allclose(dgp.cov(), np.cov(obs, rowvar=False, ddof=1))


def test_empirical_overrides_ignore_kwargs_silently() -> None:
    """Analytic exact methods should not error on (irrelevant) MC kwargs."""

    obs = np.arange(8).reshape(2, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs)
    # No warning, no error; result is the exact analytic mean.
    np.testing.assert_array_equal(
        dgp.mean(atol=1e-9, rtol=0.0, max_its=10), obs.mean(axis=0)
    )


# ---------------------------------------------------------------------------
# ParametricDGP with generator: MC fallback
# ---------------------------------------------------------------------------
def test_parametric_generator_only_uses_mc_for_mean() -> None:
    """Without distribution=, mean must converge via MC to the true value."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: 3.0 + rng.standard_normal(shape),
        default_shape=(),
        seed=0,
    )
    mean = dgp.mean(atol=0.05, rtol=0.0, max_its=10_000)
    assert abs(mean - 3.0) < 0.1


def test_parametric_generator_only_var_converges_to_truth() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: 2.0 * rng.standard_normal(shape),
        default_shape=(),
        seed=0,
    )
    # True variance is 4; with single-sample draws _per_coord_var
    # returns NaN, so use multi-sample draws.
    dgp_n = ParametricDGP(
        generator=lambda rng, shape: 2.0 * rng.standard_normal(shape),
        default_shape=(200,),
        seed=0,
    )
    v = dgp_n.var(atol=0.2, rtol=0.0, max_its=5_000)
    assert abs(float(v) - 4.0) < 0.5
    del dgp


# ---------------------------------------------------------------------------
# ParametricDGP with scipy.stats: analytic dispatch
# ---------------------------------------------------------------------------
def test_parametric_with_scipy_norm_mean_is_analytic() -> None:
    """``dgp.mean()`` should return scipy's exact 0.0, no Monte Carlo."""

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=2),
        default_shape=(100,),
        seed=0,
    )
    # Direct scipy.norm.mean() is 0.0 exactly.
    assert dgp.mean() == 0.0


def test_parametric_with_scipy_norm_var_is_analytic() -> None:
    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=2),
        default_shape=(100,),
        seed=0,
    )
    assert dgp.var() == 4.0  # scale^2


def test_parametric_with_scipy_norm_expect_uses_scipy() -> None:
    """``dgp.expect(func)`` should route to ``norm.expect`` (analytic).

    scipy.stats.norm.expect(lambda x: x**2) returns the second moment
    by numerical integration; for unit-variance norm that is 1.0
    exactly (up to integration tolerance, much tighter than 1e-3 MC).
    """

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=1),
        default_shape=(100,),
        seed=0,
    )
    val = dgp.expect(lambda x: x**2)
    np.testing.assert_allclose(val, 1.0, atol=1e-6)


def test_parametric_with_scipy_norm_expect_forwards_kwargs() -> None:
    """``dist_kwargs`` (``lb``, ``ub``) flow to scipy.expect."""

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=1),
        default_shape=(100,),
        seed=0,
    )
    # E[1 | X > 0] in scipy.expect's convention is P(X > 0) = 0.5.
    val = dgp.expect(lambda x: 1.0, lb=0)
    np.testing.assert_allclose(val, 0.5, atol=1e-6)


def test_parametric_with_scipy_norm_cov_falls_back_to_mc() -> None:
    """``norm`` has no ``.cov``; ``dgp.cov()`` uses MC reducer.

    For 1d univariate, ``_per_coord_cov`` returns the sample variance;
    average over MC samples converges to the true variance (=1).
    """

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=1),
        default_shape=(200,),
        seed=0,
    )
    val = dgp.cov(atol=0.05, rtol=0.0, max_its=5_000)
    assert abs(float(val) - 1.0) < 0.15


def test_parametric_with_scipy_mvn_mean_uses_attribute() -> None:
    """``multivariate_normal.mean`` is an attribute (not a method).

    The mixin's _try_analytic handles non-callable attributes.
    """

    mvn = st.multivariate_normal(mean=[1.0, 2.0], cov=[[1, 0.5], [0.5, 1]])
    dgp = ParametricDGP(
        distribution=mvn,
        default_shape=(100,),
        seed=0,
    )
    np.testing.assert_array_equal(dgp.mean(), np.array([1.0, 2.0]))


def test_parametric_with_scipy_mvn_cov_uses_attribute() -> None:
    mvn = st.multivariate_normal(mean=[1.0, 2.0], cov=[[1, 0.5], [0.5, 1]])
    dgp = ParametricDGP(
        distribution=mvn,
        default_shape=(100,),
        seed=0,
    )
    np.testing.assert_array_equal(dgp.cov(), np.array([[1, 0.5], [0.5, 1]]))


def test_parametric_with_scipy_mvn_var_falls_back_to_mc() -> None:
    """mvn has no .var attribute; per-coord variance via MC."""

    mvn = st.multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 4]])
    dgp = ParametricDGP(distribution=mvn, default_shape=(500,), seed=0)
    v = dgp.var(atol=0.1, rtol=0.0, max_its=5_000)
    np.testing.assert_allclose(np.asarray(v), np.array([1.0, 4.0]), atol=0.3)


def test_parametric_draws_use_distribution_rvs() -> None:
    """When distribution= is set, draw() should use distribution.rvs."""

    mvn = st.multivariate_normal(mean=[1.0, -1.0], cov=np.eye(2))
    # Two DGPs constructed with the same seed should agree on their
    # first draws (verifies that draw() actually consults the
    # distribution and that the DGP's own Generator drives it).
    dgp_a = ParametricDGP(distribution=mvn, default_shape=(50,), seed=42)
    dgp_b = ParametricDGP(distribution=mvn, default_shape=(50,), seed=42)
    draw_a = dgp_a.draw()
    assert draw_a.shape == (50, 2)
    np.testing.assert_array_equal(draw_a, dgp_b.draw())


# ---------------------------------------------------------------------------
# Duck-typed user-defined "distribution-like" object
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
    np.testing.assert_array_equal(dgp.draw(), np.array([7.0] * 5))


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
        ParametricDGP(
            generator=gen,
            distribution=st.norm(),
            default_shape=(10,),
        )


# ---------------------------------------------------------------------------
# Aggregator dispatch over return types
# ---------------------------------------------------------------------------
def test_aggregator_handles_numpy_scalar_returns() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    # Reducer returns python scalar; aggregator promotes to 0-d.
    val = dgp.expect(lambda x: float(x), atol=0.05, rtol=0.0, max_its=5_000)
    assert abs(float(val)) < 0.1


def test_aggregator_handles_numpy_array_returns() -> None:
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(10, 2),
        seed=0,
    )
    val = dgp.expect(lambda X: X.mean(axis=0), atol=0.1, rtol=0.0, max_its=5_000)
    assert val.shape == (2,)
    np.testing.assert_allclose(val, np.zeros(2), atol=0.2)


def test_aggregator_handles_pandas_dataframe() -> None:
    """DataFrames aggregate by re-wrapping with original index/columns."""

    def gen(rng, shape):
        arr = rng.standard_normal(size=(shape[0], 2))
        return pd.DataFrame(arr, columns=["a", "b"])

    dgp = ParametricDGP(generator=gen, default_shape=(50,), seed=0)
    # Reduce each DataFrame to a 1-row DataFrame of column means.
    val = dgp.expect(
        lambda df: pd.DataFrame(
            [df.mean().to_numpy()], columns=df.columns, index=["m"]
        ),
        atol=0.1,
        rtol=0.0,
        max_its=5_000,
    )
    assert isinstance(val, pd.DataFrame)
    assert list(val.columns) == ["a", "b"]
    assert list(val.index) == ["m"]


def test_aggregator_preserves_dataframe_subclass() -> None:
    """Aggregating ``DataFrame`` subclasses re-wraps as the original type.

    Mirrors how ``datamat.DataMat`` (a ``pd.DataFrame`` subclass) is
    preserved by ``type(first)(arr, index=..., columns=...)`` re-wrap.
    """

    class TinyMat(pd.DataFrame):
        """Minimal DataMat-like subclass for the aggregator test."""

        @property
        def _constructor(self):
            return TinyMat

    def gen(rng, shape):
        arr = rng.standard_normal(size=(1, 3))
        return TinyMat(arr, columns=["x", "y", "z"], index=["m"])

    dgp = ParametricDGP(generator=gen, default_shape=(50,), seed=0)
    val = dgp.expect(lambda df: df, atol=0.2, rtol=0.0, max_its=5_000)
    assert isinstance(val, TinyMat)
    assert list(val.columns) == ["x", "y", "z"]
    assert list(val.index) == ["m"]


def test_aggregator_handles_pandas_series() -> None:
    def gen(rng, shape):
        arr = rng.standard_normal(size=shape)
        return pd.Series(arr, index=[f"r{k}" for k in range(arr.shape[0])])

    dgp = ParametricDGP(generator=gen, default_shape=(3,), seed=0)
    val = dgp.expect(lambda s: s, atol=0.1, rtol=0.0, max_its=5_000)
    assert isinstance(val, pd.Series)
    assert list(val.index) == ["r0", "r1", "r2"]


def test_aggregator_rejects_unsupported_type() -> None:
    """Returning a python list from func raises NotImplementedError."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(5,),
        seed=0,
    )
    with pytest.raises(NotImplementedError, match="cannot aggregate"):
        dgp.expect(lambda X: [float(X[0]), float(X[1])])


def test_aggregator_rejects_inconsistent_shape() -> None:
    """Func returning variable-shape outputs across draws raises ValueError."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    call_count = {"n": 0}

    def varying(_x):
        call_count["n"] += 1
        # Return shape (1,) then (2,) -- inconsistent.
        return np.zeros(1) if call_count["n"] < 100 else np.zeros(2)

    with pytest.raises(ValueError, match="inconsistent shapes"):
        dgp.expect(varying, batch_size=200, max_its=200)


# ---------------------------------------------------------------------------
# kwargs handling
# ---------------------------------------------------------------------------
def test_mc_fallback_warns_on_unhandled_dist_kwargs() -> None:
    """Backend kwargs to a generator-only DGP raise NumericalWarning."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(),
        default_shape=(),
        seed=0,
    )
    with pytest.warns(NumericalWarning, match="lb"):
        dgp.expect(lambda x: x, lb=0, atol=0.1, max_its=2_000)


def test_analytic_dispatch_does_not_warn_when_kwargs_consumed() -> None:
    """No warning when scipy.expect successfully consumes lb/ub."""

    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=1),
        default_shape=(50,),
        seed=0,
    )
    # Should produce no NumericalWarning -- scipy.expect consumed lb.
    import warnings as warnings_mod

    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("error", NumericalWarning)
        dgp.expect(lambda x: 1.0, lb=0)


# ---------------------------------------------------------------------------
# TwoStageDGP
# ---------------------------------------------------------------------------
def test_twostage_mean_raises_not_implemented() -> None:
    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0]]), seed=0)

    def inner(chars, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(3, 2),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)
    with pytest.raises(NotImplementedError, match="lists of per-cluster"):
        ts.mean()
    with pytest.raises(NotImplementedError, match="lists of per-cluster"):
        ts.var()
    with pytest.raises(NotImplementedError, match="lists of per-cluster"):
        ts.cov()


def test_twostage_expect_works_with_flat_aggregator() -> None:
    """expect() works on a TwoStageDGP when ``func`` flattens to numpy."""

    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0], [3.0]]), seed=0)

    def inner(chars, rng):
        return ParametricDGP(
            generator=lambda rng_inner, shape: rng_inner.standard_normal(shape),
            default_shape=(4, 2),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=inner, seed=0)

    def flat_mean(lst):
        return np.vstack(lst).mean(axis=0)

    val = ts.expect(flat_mean, atol=0.2, rtol=0.0, max_its=2_000)
    assert val.shape == (2,)
    np.testing.assert_allclose(val, np.zeros(2), atol=0.3)


# ---------------------------------------------------------------------------
# Mixin attribute discovery
# ---------------------------------------------------------------------------
def test_subclass_without_distribution_attribute_uses_mc() -> None:
    """An inheriting class with no ``.distribution`` attribute still works."""

    class MinimalDGP(DistributionalFeatures):
        def __init__(self, seed: int):
            self._rng = np.random.default_rng(seed)

        @property
        def data(self):
            return None

        def draw(self, size=None):
            return self._rng.standard_normal()

    m = MinimalDGP(seed=0)
    val = m.expect(lambda x: x, atol=0.1, rtol=0.0, max_its=5_000)
    assert abs(float(val)) < 0.2
