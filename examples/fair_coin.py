"""Fair-coin Bernoulli: the smallest interesting ParametricDGP.

Illustrative; not part of the public ``dgp_protocol`` API.  See
``AGENTS.md`` §1 for why concrete DGPs do not live under ``src/``.

This file constructs a ``ParametricDGP`` for the distribution

    P(X = 1) = P(X = 0) = 1/2.

The generator is spelled ``rng.binomial(n=1, p=0.5, ...)`` rather than
``rng.integers(0, 2, ...)`` so the model intent (a Bernoulli with named
parameter) is visible in the source -- equivalent draws, very different
readability.

The realization shape is the tabular ``(N, p)`` convention used in the
test suite: ``N`` rows of observations, ``p == 1`` column for this
univariate case.  Switching to a bare ``(N,)`` would break that
convention without simplifying anything important here.

The DGP carries no observed realization (``observation=None``); it is a
pure Monte Carlo specification.  Bind one with ``dgp.with_data(obs)``
if you want ``dgp.data`` to return something.

Run directly::

    python examples/fair_coin.py

Or load into an IPython session so ``dgp`` (and the other module
symbols) land in your namespace::

    %run examples/fair_coin.py
    # dgp, fair_coin, DEFAULT_SHAPE now available

    # Or, equivalently, start IPython with the file pre-loaded:
    #   ipython -i examples/fair_coin.py
"""

from __future__ import annotations

import numpy as np
from dgp_protocol import DataGeneratingProcess, ParametricDGP


def fair_coin(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Generate a fair-coin Bernoulli realization of the given shape."""

    return rng.binomial(n=1, p=0.5, size=shape)


# Per-realization default shape: ``N`` rows by ``p == 1`` column.
DEFAULT_SHAPE: tuple[int, int] = (100, 1)


dgp = ParametricDGP(
    generator=fair_coin,
    default_shape=DEFAULT_SHAPE,
)


def _demo() -> None:
    """Sanity-check the DGP: protocol conformance + a small draw."""

    # Structural typing: the DGP satisfies the runtime-checkable Protocol.
    assert isinstance(dgp, DataGeneratingProcess)
    assert dgp.data is None

    rng = np.random.default_rng(0)

    sample = dgp.draw(rng=rng)
    print(f"default draw: shape={sample.shape}  dtype={sample.dtype}")
    print(f"  sample mean ({sample.shape[0]} draws) = {float(sample.mean()):.4f}")

    # Explicit ``size`` overrides ``default_shape``.
    big = dgp.draw(size=(10_000, 1), rng=rng)
    print(f"sized draw:   shape={big.shape}")
    print(f"  sample mean ({big.shape[0]:,} draws) = {float(big.mean()):.4f}")


if __name__ == "__main__":
    _demo()
