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

Alternative construction via scipy.stats
----------------------------------------
The same distribution is also constructed as ``dgp_scipy`` using
the ``distribution=`` constructor argument with
``scipy.stats.bernoulli(p=0.5)``.  When ``ParametricDGP`` is given
a scipy.stats-style frozen distribution (or any duck-typed
equivalent), the :meth:`mean` / :meth:`var` / :meth:`cov` /
:meth:`expect` methods preferentially delegate to the
distribution's analytic implementations -- no Monte Carlo, no
adaptive-convergence loop.  ``dgp_scipy.mean()`` therefore returns
``0.5`` exactly and immediately, whereas ``dgp.mean()`` (custom
generator) raises :class:`~dgp_protocol.AnalyticUnavailable`
because no analytic backend is supplied.

The free function ``dgp_protocol.mean(dgp)`` papers over the
difference: it catches ``AnalyticUnavailable`` and falls back to
adaptive Monte Carlo for the custom-generator case while taking
the analytic shortcut for the scipy case.  Same call site, two
underlying paths.

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
import scipy.stats as st

from dgp_protocol import AnalyticUnavailable, DataGeneratingProcess, ParametricDGP
from dgp_protocol import mean as marginal_mean


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

# Same distribution constructed via scipy.stats.bernoulli.  Different
# draw mechanism (scipy.stats.bernoulli.rvs vs numpy.binomial), same
# marginal distribution -- and analytic .mean() / .var() / .cov() /
# .expect() come along for free via duck-typed dispatch.
dgp_scipy = ParametricDGP(
    distribution=st.bernoulli(p=0.5),
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

    # ----- scipy-backed alternative: analytic features for free -----
    print()
    print("scipy-backed alternative (dgp_scipy):")

    # ``.item()`` coerces both 0-d scalars (scipy's analytic return)
    # and (1,)-shape arrays (MC fallback aggregating over rows of a
    # (N, 1) draw) to a Python float.  See the module docstring for
    # the analytic-vs-MC dispatch story.
    def _scalar(x):
        return np.asarray(x).flatten()[0]

    # Analytic dispatch: no MC, no convergence loop.
    print(
        f"  dgp_scipy.mean() (analytic):  {_scalar(dgp_scipy.mean()):.4f}"
        f"     # exactly p = 0.5"
    )
    print(
        f"  dgp_scipy.var()  (analytic):  {_scalar(dgp_scipy.var()):.4f}"
        f"     # exactly p(1-p) = 0.25"
    )

    # Contrast: the custom-generator dgp refuses analytic methods.
    try:
        dgp.mean()
    except AnalyticUnavailable as exc:
        first_line = str(exc).splitlines()[0]
        print(f"  dgp.mean()       raised:  {first_line}")

    # The free function takes the analytic shortcut for the scipy
    # backend and falls back to adaptive MC for the custom-generator
    # case -- same call site, two paths under the hood.
    print(
        f"  marginal.mean(dgp_scipy) = "
        f"{_scalar(marginal_mean(dgp_scipy)):.4f}  (analytic shortcut)"
    )
    print(
        f"  marginal.mean(dgp)       = "
        f"{_scalar(marginal_mean(dgp, atol=0.01, rtol=0.0)):.4f}"
        f"  (MC fallback, atol=0.01)"
    )


if __name__ == "__main__":
    _demo()
