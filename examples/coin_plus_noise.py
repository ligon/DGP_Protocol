"""Coin-plus-noise: a continuous DGP composed atop the fair-coin DGP.

Illustrative; not part of the public ``dgp_protocol`` API.

This file constructs

    Y = X + e,
    X ~ Bernoulli(p = 0.5)   (provided by examples.fair_coin),
    e ~ N(0, 1),              (independent of X).

Marginally, Y is a 50/50 mixture of N(0, 1) and N(1, 1):

    E[Y]   = p       = 0.5,
    Var[Y] = p(1-p) + 1
           = 0.25 + 1
           = 1.25.

These are the population moments an analog-estimation consumer
recovers by computing ``mean(dgp.data)`` and ``var(dgp.data)``.

Composition mechanics
---------------------
The generator closes over the imported ``coin_dgp`` instance.  Each
call to this DGP's :meth:`draw` advances *two* independent RNG
streams:

- ``coin_dgp._rng`` (seeded inside :mod:`examples.fair_coin`,
  ``seed=0``) supplies the binary X via ``coin_dgp.draw(size=...)``.
- This DGP's own ``_rng`` (seeded below, ``seed=0``) supplies the
  Gaussian noise via the ``rng`` argument the protocol passes to the
  generator.

Reproducibility of a draw therefore requires both seeds to be
fixed.  This is the natural consequence of "each DGP owns its
randomness" -- composite DGPs that embed other DGPs simply embed
their RNG ownership too.  If you want a *single* seed to control
everything, replace ``coin_dgp.draw(size=...)`` with
``coin_dgp.with_rng(rng.spawn(1)[0]).draw(size=...)`` to spawn a
child stream off this DGP's own ``rng`` -- a more elaborate pattern
documented under :meth:`ParametricDGP.with_rng`.

Bound observation
-----------------
The DGP is bound to an observed realization ``Y_OBSERVATION``
constructed once at module load time from the fair-coin
example's existing observed ``X`` plus an independent Gaussian
noise sample (seed ``E_OBS_SEED`` distinct from any other seed used
here).  ``dgp.data`` therefore returns a fixed ``(N, 1)`` array of
real-valued observations a downstream consumer can estimate from.

Run directly::

    python examples/coin_plus_noise.py

Or load into an IPython session::

    %run examples/coin_plus_noise.py
"""

from __future__ import annotations

# Allow running as a script (``python examples/coin_plus_noise.py`` or
# ``%run examples/coin_plus_noise.py``) by putting the repo root on
# ``sys.path`` so ``examples`` resolves as a package.  Also works
# unchanged under ``python -m examples.coin_plus_noise``.
if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "examples"

import numpy as np

from dgp_protocol import DataGeneratingProcess, ParametricDGP
from examples.fair_coin import DEFAULT_SHAPE
from examples.fair_coin import OBSERVATION as X_OBSERVATION
from examples.fair_coin import dgp as coin_dgp


def coin_plus_noise(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Generate ``X + e`` where X is binary and e is N(0, 1)."""

    x = coin_dgp.draw(size=shape)
    e = rng.standard_normal(size=shape)
    return x + e


# Independent seed for the noise component of the bound observation;
# deliberately distinct from coin_dgp's own seed (0) and from this
# DGP's draw seed (also 0) so all three streams are independent.
E_OBS_SEED: int = 2027

E_OBSERVATION: np.ndarray = np.random.default_rng(E_OBS_SEED).standard_normal(
    X_OBSERVATION.shape
)
Y_OBSERVATION: np.ndarray = X_OBSERVATION + E_OBSERVATION


dgp = ParametricDGP(
    generator=coin_plus_noise,
    default_shape=DEFAULT_SHAPE,
    observation=Y_OBSERVATION,
    seed=0,
)


def _demo() -> None:
    """Sanity-check: protocol conformance, observed data, fresh draws."""

    assert isinstance(dgp, DataGeneratingProcess)
    assert dgp.data is not None
    assert dgp.data.shape == DEFAULT_SHAPE

    # Analog estimators recovered from the bound observation.
    y_mean = float(dgp.data.mean())
    y_var = float(dgp.data.var(ddof=1))
    print(
        f"observed:    shape={dgp.data.shape}  "
        f"mean = {y_mean:.4f}  var = {y_var:.4f}"
    )
    print("  (model: E[Y] = p = 0.5000;  Var[Y] = p(1-p) + 1 = 1.2500)")

    # Fresh draws; coin's seed + this dgp's seed both fixed, so reproducible.
    sample = dgp.draw()
    print(f"fresh draw:  shape={sample.shape}  " f"mean = {float(sample.mean()):.4f}")

    big = dgp.draw(size=(10_000, 1))
    print(
        f"sized draw:  shape={big.shape}  "
        f"mean = {float(big.mean()):.4f}  "
        f"var  = {float(big.var(ddof=1)):.4f}"
    )


if __name__ == "__main__":
    _demo()
