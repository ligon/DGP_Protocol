"""Tests pinning the pickle behaviour of the DGP containers.

The package's ``ParametricDGP`` and ``TwoStageDGP`` override
``__reduce__`` to use :mod:`cloudpickle` internally for their
callable fields (``generator`` / ``inner``), so stdlib
``pickle.dumps(dgp)`` works transparently for the lambda /
closure / nested-function idioms that are the natural shapes for
those fields.  Round-trips preserve the ``_rng`` state, so a
DGP unpickled to the same seed produces the same draws.

Matrix exercised:

- ``EmpiricalDGP`` with ``IIDSampling`` (default) and
  ``ClusteredSampling`` -- no callables, stdlib-pickle-native.
- ``ParametricDGP`` with: ``distribution=scipy.stats.norm()``;
  ``generator=`` a module-level function; ``generator=`` a lambda
  closing over a local variable; ``generator=`` a nested function
  closing over a local variable.
- ``TwoStageDGP`` with: ``inner=`` a module-level function;
  ``inner=`` a lambda closing over the outer scope (the natural
  idiom for hierarchical composition).
- ``cloudpickle.dumps`` / ``cloudpickle.loads`` directly (the
  parallel-worker path that joblib / Dask / Ray use under the
  hood).
- Bound observation preservation under pickle.
- ``_rng`` spawn-capability preserved across pickle (so
  parallel-fan-out continues to work on the unpickled instance).
"""

from __future__ import annotations

import pickle

import cloudpickle
import numpy as np
import scipy.stats as st

from dgp_protocol import (
    ClusteredSampling,
    EmpiricalDGP,
    ParametricDGP,
    TwoStageDGP,
)


# ---------------------------------------------------------------------------
# Module-level callables (stdlib-picklable by qualified-name lookup).
# ---------------------------------------------------------------------------
def _module_level_generator(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    return rng.standard_normal(size=shape)


def _module_level_inner(chars: np.ndarray, rng: np.random.Generator) -> ParametricDGP:
    return ParametricDGP(
        generator=_module_level_generator,
        default_shape=(5,),
    ).with_rng(rng)


# ---------------------------------------------------------------------------
# EmpiricalDGP: stdlib-pickle-native (no callables).
# ---------------------------------------------------------------------------
def test_pickle_empirical_iid() -> None:
    obs = np.arange(20).reshape(5, 4).astype(float)
    dgp = EmpiricalDGP(observation=obs, seed=0)
    dgp2 = pickle.loads(pickle.dumps(dgp))
    assert isinstance(dgp2, EmpiricalDGP)
    np.testing.assert_array_equal(dgp2.observation, obs)
    np.testing.assert_array_equal(dgp.draw(), dgp2.draw())


def test_pickle_empirical_clustered() -> None:
    obs = np.arange(40).reshape(10, 4).astype(float)
    ids = np.array([0] * 5 + [1] * 5)
    dgp = EmpiricalDGP(
        observation=obs, sampling=ClusteredSampling(cluster_ids=ids), seed=0
    )
    dgp2 = pickle.loads(pickle.dumps(dgp))
    assert isinstance(dgp2, EmpiricalDGP)
    assert isinstance(dgp2.sampling, ClusteredSampling)
    np.testing.assert_array_equal(dgp2.sampling.cluster_ids, ids)
    np.testing.assert_array_equal(dgp.draw(), dgp2.draw())


# ---------------------------------------------------------------------------
# ParametricDGP: every flavour round-trips via stdlib pickle.
# ---------------------------------------------------------------------------
def test_pickle_parametric_with_module_level_generator() -> None:
    dgp = ParametricDGP(generator=_module_level_generator, default_shape=(20,), seed=0)
    dgp2 = pickle.loads(pickle.dumps(dgp))
    np.testing.assert_array_equal(dgp.draw(), dgp2.draw())


def test_pickle_parametric_with_scipy_distribution() -> None:
    dgp = ParametricDGP(
        distribution=st.norm(loc=0, scale=2), default_shape=(50,), seed=0
    )
    dgp2 = pickle.loads(pickle.dumps(dgp))
    np.testing.assert_array_equal(dgp.draw(), dgp2.draw())
    # And the analytic dispatch still works (proves the distribution
    # object is alive on the other side).
    assert dgp2.mean() == 0.0
    assert dgp2.var() == 4.0


def test_pickle_parametric_with_lambda_generator_closing_over_local() -> None:
    """The case stdlib pickle CAN'T do without our __reduce__ override.

    Lambda + closure over a local variable (``sigma``).  Without
    __reduce__ + cloudpickle, ``pickle.dumps`` would raise
    PicklingError("Can't pickle <function <lambda>>").
    """

    sigma = 2.5
    dgp = ParametricDGP(
        generator=lambda rng, shape: sigma * rng.standard_normal(size=shape),
        default_shape=(50,),
        seed=0,
    )
    dgp2 = pickle.loads(pickle.dumps(dgp))
    np.testing.assert_array_equal(dgp.draw(), dgp2.draw())
    # Closure value preserved.
    big = dgp2.draw(size=(2000,))
    assert 2.0 < big.std() < 3.0  # near sigma = 2.5


def test_pickle_parametric_with_nested_function_generator() -> None:
    """Nested-function generator (not module-level) round-trips."""

    def make_dgp(loc):
        def gen(rng, shape):
            return loc + rng.standard_normal(size=shape)

        return ParametricDGP(generator=gen, default_shape=(50,), seed=0)

    dgp = make_dgp(loc=7.0)
    dgp2 = pickle.loads(pickle.dumps(dgp))
    np.testing.assert_array_equal(dgp.draw(), dgp2.draw())
    # The closed-over ``loc`` is preserved.
    big = dgp2.draw(size=(2000,))
    assert abs(float(big.mean()) - 7.0) < 0.2


def test_pickle_preserves_bound_observation() -> None:
    """An observation supplied to ParametricDGP round-trips intact."""

    obs = np.arange(10).reshape(5, 2).astype(float)
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(5, 2),
        observation=obs,
        seed=0,
    )
    dgp2 = pickle.loads(pickle.dumps(dgp))
    np.testing.assert_array_equal(dgp2.data, obs)


# ---------------------------------------------------------------------------
# TwoStageDGP: outer + inner both round-trip.
# ---------------------------------------------------------------------------
def test_pickle_twostage_with_module_level_inner() -> None:
    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0]]), seed=0)
    ts = TwoStageDGP(outer=outer, inner=_module_level_inner, seed=0)
    ts2 = pickle.loads(pickle.dumps(ts))
    r1 = ts.draw()
    r2 = ts2.draw()
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_pickle_twostage_with_lambda_inner_and_lambda_inner_generator() -> None:
    """The natural hierarchical-composition idiom: lambda inner that
    returns a lambda-generator ParametricDGP.  Both lambdas must
    round-trip via cloudpickle for this to work."""

    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0], [3.0]]), seed=0)

    def make_inner(chars, rng):
        mu = float(chars[0])
        return ParametricDGP(
            generator=lambda r, sh: mu + r.standard_normal(sh),
            default_shape=(4, 1),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=make_inner, seed=0)
    ts2 = pickle.loads(pickle.dumps(ts))
    r1 = ts.draw()
    r2 = ts2.draw()
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2, strict=True):
        np.testing.assert_array_equal(a, b)


def test_pickle_twostage_with_parametric_lambda_outer() -> None:
    """Outer DGP is itself a ParametricDGP with a lambda generator.

    Exercises the recursive __reduce__ chain: TwoStageDGP.__reduce__
    pickles outer via stdlib pickle, which invokes
    ParametricDGP.__reduce__, which routes the lambda through
    cloudpickle.  All seamless from the caller's perspective.
    """

    sigma_outer = 1.5
    outer = ParametricDGP(
        generator=lambda rng, shape: sigma_outer * rng.standard_normal(shape),
        default_shape=(3, 1),
        seed=0,
    )

    def make_inner(chars, rng):
        return ParametricDGP(
            generator=lambda r, sh: float(chars[0]) + r.standard_normal(sh),
            default_shape=(2, 1),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=make_inner, seed=0)
    ts2 = pickle.loads(pickle.dumps(ts))
    r1 = ts.draw()
    r2 = ts2.draw()
    for a, b in zip(r1, r2, strict=True):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# cloudpickle direct path (the joblib / Dask / Ray worker pickling path).
# ---------------------------------------------------------------------------
def test_cloudpickle_direct_round_trip_parametric_lambda() -> None:
    sigma = 3.0
    dgp = ParametricDGP(
        generator=lambda rng, shape: sigma * rng.standard_normal(shape),
        default_shape=(50,),
        seed=0,
    )
    dgp2 = cloudpickle.loads(cloudpickle.dumps(dgp))
    np.testing.assert_array_equal(dgp.draw(), dgp2.draw())


def test_cloudpickle_direct_round_trip_twostage_lambda() -> None:
    outer = EmpiricalDGP(observation=np.array([[1.0], [2.0]]), seed=0)

    def make_inner(chars, rng):
        mu = float(chars[0])
        return ParametricDGP(
            generator=lambda r, sh: mu + r.standard_normal(sh),
            default_shape=(3, 1),
        ).with_rng(rng)

    ts = TwoStageDGP(outer=outer, inner=make_inner, seed=0)
    ts2 = cloudpickle.loads(cloudpickle.dumps(ts))
    for a, b in zip(ts.draw(), ts2.draw(), strict=True):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Post-pickle behaviour: _rng remains spawn-capable.
# ---------------------------------------------------------------------------
def test_pickle_preserves_rng_spawn_capability() -> None:
    """The unpickled DGP's _rng is still a Generator that supports
    spawn(), so the parallel-fan-out idiom keeps working."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(5,),
        seed=42,
    )
    dgp2 = pickle.loads(pickle.dumps(dgp))

    # spawn() works on the unpickled rng.
    spawned = dgp2._rng.spawn(3)
    assert len(spawned) == 3
    assert all(isinstance(s, np.random.Generator) for s in spawned)

    # The standard parallel-fan-out idiom is supported on the
    # unpickled DGP.
    workers = [dgp2.with_rng(s) for s in spawned]
    draws = [w.draw() for w in workers]
    # Each worker draws independently; their draws differ.
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])
