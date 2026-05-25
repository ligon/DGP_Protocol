"""Tests for the :class:`DataGeneratingProcess` Protocol surface."""

from __future__ import annotations

import numpy as np
from dgp_protocol import DataGeneratingProcess, EmpiricalDGP, ParametricDGP


def test_protocol_runtime_checkable_for_empirical() -> None:
    """``EmpiricalDGP`` satisfies the Protocol at runtime."""

    obs = np.arange(12).reshape(4, 3)
    dgp = EmpiricalDGP(observation=obs)
    assert isinstance(dgp, DataGeneratingProcess)


def test_protocol_runtime_checkable_for_parametric() -> None:
    """``ParametricDGP`` satisfies the Protocol at runtime."""

    def gen(rng, shape):
        return rng.standard_normal(size=shape)

    dgp = ParametricDGP(generator=gen, default_shape=(10, 2))
    assert isinstance(dgp, DataGeneratingProcess)


def test_arbitrary_object_does_not_satisfy_protocol() -> None:
    """An object without ``data``/``draw`` is rejected by isinstance."""

    class _NotADGP:
        pass

    assert not isinstance(_NotADGP(), DataGeneratingProcess)


def test_minimal_duck_satisfies_protocol() -> None:
    """An ad-hoc duck-typed object with the right shape passes.

    ``draw`` takes only ``size`` -- the DGP owns its own rng, so the
    Protocol does not declare an ``rng`` parameter.
    """

    class _DuckDGP:
        @property
        def data(self):
            return None

        def draw(self, size=None):
            return None

    assert isinstance(_DuckDGP(), DataGeneratingProcess)


def test_imports_succeed() -> None:
    """Re-exports from ``dgp_protocol.__init__`` resolve."""

    import dgp_protocol

    assert hasattr(dgp_protocol, "DataGeneratingProcess")
    assert hasattr(dgp_protocol, "EmpiricalDGP")
    assert hasattr(dgp_protocol, "ParametricDGP")
    assert hasattr(dgp_protocol, "TwoStageDGP")
    assert hasattr(dgp_protocol, "IIDSampling")
    assert hasattr(dgp_protocol, "ClusteredSampling")
    assert hasattr(dgp_protocol, "with_data")
    assert hasattr(dgp_protocol, "__version__")
