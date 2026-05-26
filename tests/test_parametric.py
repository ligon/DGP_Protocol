"""Tests for :class:`ParametricDGP`."""

from __future__ import annotations

import numpy as np

from dgp_protocol import ParametricDGP


def test_data_defaults_to_none() -> None:
    """An unbound ParametricDGP has ``data is None``."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape), default_shape=(5, 2)
    )
    assert dgp.data is None


def test_draw_uses_default_shape_when_size_none() -> None:
    """``draw()`` without ``size`` falls back to ``default_shape``."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(20, 3),
        seed=0,
    )
    draw = dgp.draw()
    assert draw.shape == (20, 3)


def test_draw_respects_explicit_size() -> None:
    """``draw(size=...)`` overrides ``default_shape``."""

    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(20, 3),
        seed=0,
    )
    draw = dgp.draw(size=(7, 5))
    assert draw.shape == (7, 5)


def test_with_data_binds_observation_preserving_generator() -> None:
    """``with_data`` returns a new DGP with the observation set."""

    gen = lambda rng, shape: rng.standard_normal(shape)  # noqa: E731
    shape = (10, 2)
    dgp = ParametricDGP(generator=gen, default_shape=shape)

    obs = np.zeros(shape)
    bound = dgp.with_data(obs)

    assert bound is not dgp
    assert bound.data is obs
    assert bound.generator is gen
    assert bound.default_shape == shape
    # Original is still unbound.
    assert dgp.data is None


def test_generator_can_carry_state_via_closure() -> None:
    """Generators that close over outside state work correctly."""

    sigma = 3.0

    def gen(rng, shape):
        return sigma * rng.standard_normal(shape)

    dgp = ParametricDGP(generator=gen, default_shape=(1000, 1), seed=0)
    draws = dgp.draw()
    # Empirical std should be near sigma.
    assert abs(float(draws.std()) - sigma) < 0.3


# ---------------------------------------------------------------------------
# Randomness-ownership semantics (seed, with_rng, with_data spawn)
# ---------------------------------------------------------------------------
def test_seed_makes_draws_reproducible() -> None:
    """Two DGPs built with the same seed produce identical draw streams."""

    def gen(rng, shape):
        return rng.standard_normal(shape)

    a = ParametricDGP(generator=gen, default_shape=(50, 2), seed=42)
    b = ParametricDGP(generator=gen, default_shape=(50, 2), seed=42)
    for _ in range(3):
        np.testing.assert_array_equal(a.draw(), b.draw())


def test_with_rng_replaces_generator() -> None:
    """``with_rng`` returns a sibling whose stream is the provided Generator."""

    def gen(rng, shape):
        return rng.standard_normal(shape)

    parent = ParametricDGP(generator=gen, default_shape=(50, 2), seed=7)
    injected = np.random.default_rng(999)
    sibling = parent.with_rng(injected)

    # A reference dgp with the same injected stream should produce the
    # same draw.
    reference = ParametricDGP(generator=gen, default_shape=(50, 2)).with_rng(
        np.random.default_rng(999)
    )
    np.testing.assert_array_equal(sibling.draw(), reference.draw())


def test_with_data_spawns_independent_stream() -> None:
    """``with_data`` gives the child its own (spawned) Generator.

    The parent's stream is unaffected by child-side activity.
    """

    def gen(rng, shape):
        return rng.standard_normal(shape)

    parent = ParametricDGP(generator=gen, default_shape=(10, 1), seed=11)
    child = parent.with_data(np.zeros((10, 1)))

    parent_first = parent.draw()
    _ = [child.draw() for _ in range(5)]
    parent_second = parent.draw()

    # Drawing twice from the parent gives different realizations.
    assert not np.array_equal(parent_first, parent_second)

    # Reconstructing the parent with the same seed and skipping the
    # child activity yields the same first two parent draws.
    fresh = ParametricDGP(generator=gen, default_shape=(10, 1), seed=11)
    np.testing.assert_array_equal(fresh.draw(), parent_first)
    np.testing.assert_array_equal(fresh.draw(), parent_second)
