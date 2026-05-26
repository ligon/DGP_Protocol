"""Two-way fixed effects panel DGP.

Illustrative; not part of the public ``dgp_protocol`` API.

Model
-----
::

    y_{i,t} = a_i + b_t + c * x_{i,t} + e_{i,t},

with

- ``i = 0..N-1`` indexing individuals (units),
- ``t = 0..T-1`` indexing time periods,
- ``a_i ~ N(0, SIGMA_A**2)`` -- *individual* fixed effects,
- ``b_t ~ N(0, SIGMA_B**2)`` -- *time* fixed effects,
- ``c`` the structural coefficient on x (default 1.0),
- ``x_{i,t} ~ N(0, 1)`` drawn iid per (i, t) on each realization,
- ``e_{i,t} ~ N(0, SIGMA_E**2)`` drawn iid per (i, t) on each
  realization, independent of x.

``a_i`` and ``b_t`` are treated as **fixed parameters** of this DGP
instance (the "fixed effects" framing): they are drawn *once* at
module load (seeded with ``FE_SEED``) and then held fixed across
the bound observation and every fresh ``dgp.draw()`` call.  ``x`` and
``e`` are the only stochastic ingredients that vary across
realizations.

Output format
-------------
Each realization is a :class:`pandas.DataFrame` in long format with

- shape ``(N*T, 2)``,
- columns ``['y', 'x']``,
- a :class:`pandas.MultiIndex` ``(i, t)`` over individuals and
  time periods.

This format exercises the DataFrame-aware aggregator in
:mod:`dgp_protocol._mc` -- free :func:`dgp_protocol.expect`,
:func:`~dgp_protocol.mean`, :func:`~dgp_protocol.var`, and
:func:`~dgp_protocol.cov` on this DGP return pandas-shaped results
that preserve the column labels.

Bound observation
-----------------
The DGP is bound to an observed realization (``OBSERVATION``,
drawn once with seed ``OBS_SEED``, independent of ``DGP_SEED``) so
``dgp.data`` returns a fixed panel a downstream consumer can
estimate from.  Fresh draws via ``dgp.draw()`` come from this DGP's
own RNG stream (seeded with ``DGP_SEED``) and are useful for Monte
Carlo / bootstrap studies (e.g., sampling distribution of an
estimator of ``c``).

Estimating ``c``
----------------
The structural coefficient ``c`` is what a downstream consumer
typically estimates from ``dgp.data``.  Two natural analog
estimators:

- **Naive OLS** of y on x (ignoring fixed effects): unbiased in
  this spec because ``x`` is independent of ``(a, b, e)``, but
  inefficient because ``a_i + b_t + e_{i,t}`` is left in the
  residual.
- **TWFE** (within transformation): demean y and x by unit and
  time, then OLS.  Same expectation, lower variance.

The demo at the bottom of this file computes both for ``dgp.data``
to show they recover ``c`` (and that TWFE has noticeably tighter
residuals).

Run directly::

    python examples/twfe.py

Or as a module::

    python -m examples.twfe

Or load into IPython::

    %run examples/twfe.py
"""

from __future__ import annotations

# Allow running as a script (``python examples/twfe.py`` or
# ``%run examples/twfe.py``) by putting the repo root on
# ``sys.path`` so ``examples`` resolves as a package.  Also works
# unchanged under ``python -m examples.twfe``.
if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "examples"

import numpy as np
import pandas as pd

from dgp_protocol import DataGeneratingProcess, ParametricDGP

# ---------------------------------------------------------------------------
# Panel dimensions and structural parameters.
# ---------------------------------------------------------------------------
N: int = 50  # number of individuals
T: int = 10  # number of time periods
C: float = 1.0  # true coefficient on x
SIGMA_E: float = 1.0  # error std dev
SIGMA_A: float = 1.0  # individual-FE std dev
SIGMA_B: float = 0.5  # time-FE std dev

# ---------------------------------------------------------------------------
# Seeds.
# ---------------------------------------------------------------------------
FE_SEED: int = 1729  # one-time draw of the fixed effects A_VEC, B_VEC
DGP_SEED: int = 0  # this DGP's own _rng (drives fresh draws of x, e)
OBS_SEED: int = 2028  # the bound observation (independent of DGP_SEED)

# ---------------------------------------------------------------------------
# Fixed effects: drawn once at module load, then held constant.
# ---------------------------------------------------------------------------
_fe_rng = np.random.default_rng(FE_SEED)
A_VEC: np.ndarray = SIGMA_A * _fe_rng.standard_normal(size=N)
B_VEC: np.ndarray = SIGMA_B * _fe_rng.standard_normal(size=T)

# Per-realization shape: (N*T, 2) -- long-format panel with y, x.
DEFAULT_SHAPE: tuple[int, int] = (N * T, 2)

# Reusable MultiIndex over (i, t); the same labels appear on every
# draw, which is what the DataFrame aggregator requires for stacking
# MC samples.
PANEL_INDEX: pd.MultiIndex = pd.MultiIndex.from_product(
    [range(N), range(T)], names=["i", "t"]
)


def twfe_panel(rng: np.random.Generator, shape: tuple[int, ...]) -> pd.DataFrame:
    """Generate one TWFE panel realization.

    ``shape`` is *ignored*: the panel dimensions are fixed by the
    module-level constants ``N`` and ``T``.  ``rng`` drives the
    fresh draws of ``x`` and ``e``; the fixed effects ``A_VEC`` /
    ``B_VEC`` are closed over from module scope.

    Returns
    -------
    A :class:`pandas.DataFrame` of shape ``(N*T, 2)`` with columns
    ``['y', 'x']`` and MultiIndex ``(i, t)``.
    """

    del shape  # panel dims are fixed by N, T
    x = rng.standard_normal(size=(N, T))
    e = SIGMA_E * rng.standard_normal(size=(N, T))
    y = A_VEC[:, None] + B_VEC[None, :] + C * x + e
    return pd.DataFrame(
        {"y": y.ravel(), "x": x.ravel()},
        index=PANEL_INDEX,
    )


# Bind a fixed observed realization, independent of the DGP's draw stream.
OBSERVATION: pd.DataFrame = twfe_panel(np.random.default_rng(OBS_SEED), DEFAULT_SHAPE)


dgp = ParametricDGP(
    generator=twfe_panel,
    default_shape=DEFAULT_SHAPE,
    observation=OBSERVATION,
    seed=DGP_SEED,
)


# ---------------------------------------------------------------------------
# Demo.
# ---------------------------------------------------------------------------
def _ols_naive(panel: pd.DataFrame) -> float:
    """Naive OLS of y on x (ignoring fixed effects)."""

    y = panel["y"].to_numpy()
    x = panel["x"].to_numpy()
    xc = x - x.mean()
    yc = y - y.mean()
    return float((xc @ yc) / (xc @ xc))


def _ols_twfe(panel: pd.DataFrame) -> float:
    """Within-transformed OLS of y on x (sweeps out i and t means)."""

    grand_y = panel["y"].mean()
    grand_x = panel["x"].mean()
    y_dm = (
        panel["y"]
        - panel["y"].groupby(level="i").transform("mean")
        - panel["y"].groupby(level="t").transform("mean")
        + grand_y
    )
    x_dm = (
        panel["x"]
        - panel["x"].groupby(level="i").transform("mean")
        - panel["x"].groupby(level="t").transform("mean")
        + grand_x
    )
    return float((x_dm @ y_dm) / (x_dm @ x_dm))


def _demo() -> None:
    """Sanity-check: protocol conformance, bound data, fresh draws,
    naive vs TWFE estimates of c."""

    assert isinstance(dgp, DataGeneratingProcess)
    assert isinstance(dgp.data, pd.DataFrame)
    assert dgp.data.shape == DEFAULT_SHAPE
    assert list(dgp.data.columns) == ["y", "x"]
    assert dgp.data.index.names == ["i", "t"]

    print(f"panel:        N = {N},  T = {T},  N*T = {N * T} observations")
    print(f"truth:        c = {C:.2f},  sigma_e = {SIGMA_E:.2f}")
    print(
        f"FE summary:   |a_i|_max = {np.max(np.abs(A_VEC)):.3f},"
        f"  |b_t|_max = {np.max(np.abs(B_VEC)):.3f}"
    )
    print()

    obs = dgp.data
    print(f"observed:     shape={obs.shape}  columns={list(obs.columns)}")
    print(f"  index: {obs.index.names} (MultiIndex, first 3 rows shown below)")
    print(obs.head(3).to_string())
    print(
        f"  marginals: mean(y) = {obs['y'].mean():+.4f},"
        f"  mean(x) = {obs['x'].mean():+.4f}"
    )
    c_naive_obs = _ols_naive(obs)
    c_twfe_obs = _ols_twfe(obs)
    print(
        f"  naive OLS c-hat:  {c_naive_obs:.4f}"
        f"   |  TWFE c-hat: {c_twfe_obs:.4f}"
        f"   (truth: {C:.2f})"
    )
    print()

    fresh = dgp.draw()
    print(f"fresh draw:   shape={fresh.shape}  " f"mean(y) = {fresh['y'].mean():+.4f}")
    c_naive_f = _ols_naive(fresh)
    c_twfe_f = _ols_twfe(fresh)
    print(f"  naive OLS c-hat:  {c_naive_f:.4f}" f"   |  TWFE c-hat: {c_twfe_f:.4f}")


if __name__ == "__main__":
    _demo()
