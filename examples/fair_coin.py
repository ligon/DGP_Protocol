"""Fair-coin Bernoulli: the smallest interesting ParametricDGP.

Illustrative; not part of the public ``dgp_protocol`` API.  See
``AGENTS.md`` §1 for why concrete DGPs do not live in the package
proper (``dgp_protocol/``) and are kept out under ``examples/`` or
in consumer packages instead.

This file constructs a ``ParametricDGP`` for the distribution

    P(X = 1) = P(X = 0) = 1/2.

The generator is spelled ``rng.binomial(n=1, p=0.5, ...)`` rather than
``rng.integers(0, 2, ...)`` so the model intent (a Bernoulli with named
parameter) is visible in the source -- equivalent draws, very different
readability.

The realization shape is the tabular ``(N, p)`` convention used in the
test suite: ``N`` rows of observations, ``p == 1`` column for this
univariate case.

The DGP is **bound to an observed realization** drawn once with a
fixed seed (``OBS_SEED``) so ``dgp.data`` returns a real sample.  A
downstream consumer can then compute the analog estimator

    p_hat = float(dgp.data.mean())

-- the sample analog of the population parameter
``p == E_F[X] == 0.5``, in Manski's analog-estimation sense.  The
observation's seed is intentionally distinct from the DGP's own
:meth:`draw` seed so the two streams (observed realization vs fresh
draws) are independent.

The DGP owns its own Generator.  ``seed=0`` makes the :meth:`draw`
stream reproducible across runs; pass ``seed=None`` (or omit) for
system-entropy seeding.  ``dgp.draw()`` never takes an ``rng``
argument; use ``dgp.with_rng(rng)`` to swap in a specific Generator
for parallel fan-out.

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

# Independent seed for the bound observation; deliberately distinct
# from the DGP's own draw seed so the observed realization and the
# fresh-draw stream do not share randomness.
OBS_SEED: int = 2026

OBSERVATION: np.ndarray = fair_coin(np.random.default_rng(OBS_SEED), DEFAULT_SHAPE)


dgp = ParametricDGP(
    generator=fair_coin,
    default_shape=DEFAULT_SHAPE,
    observation=OBSERVATION,
    seed=0,
)


def _demo() -> None:
    """Sanity-check the DGP: protocol conformance, observed data, fresh draws."""

    # Structural typing: the DGP satisfies the runtime-checkable Protocol.
    assert isinstance(dgp, DataGeneratingProcess)
    assert dgp.data is not None
    assert dgp.data.shape == DEFAULT_SHAPE

    # Analog estimation: p_hat is the sample analog of E_F[X] under F̂.
    p_hat = float(dgp.data.mean())
    print(f"observed:     shape={dgp.data.shape}  p_hat = {p_hat:.4f}")

    # Fresh draws (independent of the bound observation).
    sample = dgp.draw()
    print(f"fresh draw:   shape={sample.shape}  dtype={sample.dtype}")
    print(f"  sample mean ({sample.shape[0]} draws) = {float(sample.mean()):.4f}")

    # Explicit ``size`` overrides ``default_shape``.
    big = dgp.draw(size=(10_000, 1))
    print(f"sized draw:   shape={big.shape}")
    print(f"  sample mean ({big.shape[0]:,} draws) = {float(big.mean()):.4f}")


if __name__ == "__main__":
    _demo()
