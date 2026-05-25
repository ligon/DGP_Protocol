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

    rng = np.random.default_rng(0)
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(20, 3),
    )
    draw = dgp.draw(rng=rng)
    assert draw.shape == (20, 3)


def test_draw_respects_explicit_size() -> None:
    """``draw(size=...)`` overrides ``default_shape``."""

    rng = np.random.default_rng(0)
    dgp = ParametricDGP(
        generator=lambda rng, shape: rng.standard_normal(shape),
        default_shape=(20, 3),
    )
    draw = dgp.draw(size=(7, 5), rng=rng)
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

    rng = np.random.default_rng(0)
    dgp = ParametricDGP(generator=gen, default_shape=(1000, 1))
    draws = dgp.draw(rng=rng)
    # Empirical std should be near sigma.
    assert abs(float(draws.std()) - sigma) < 0.3
