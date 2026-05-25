"""Shared Monte Carlo machinery for distributional features.

Internal module.  The adaptive batched MC driver and the type-aware
aggregator are shared between :mod:`dgp_protocol.expect` (P-side:
per-observation marginal moments) and
:mod:`dgp_protocol.sample_distribution` (D-side: dataset-level
sampling moments).

Public entry points (re-exported by the consuming modules):

- :func:`adaptive_mc` -- generic adaptive batched MC driver.
- :func:`split_mc_kwargs` -- partition kwargs into MC-control vs
  backend-specific dictionaries.
- Default tolerances and budget: :data:`DEFAULT_ATOL`,
  :data:`DEFAULT_RTOL`, :data:`DEFAULT_MAX_ITS`,
  :data:`DEFAULT_BATCH_SIZE`.

The aggregator handles numpy scalars / arrays / pandas DataFrames /
pandas Series (including subclasses such as ``datamat.DataMat`` /
``datamat.DataVec`` that preserve identity via ``type(first)(...)``).
Other return types raise :class:`NotImplementedError`.

Convergence: a numpy.allclose-style check applied element-wise on
the running MC standard error,

    max(se) <= atol + rtol * max(|mean|).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from .warnings import NumericalWarning

# ---------------------------------------------------------------------------
# Defaults and kwargs split.
# ---------------------------------------------------------------------------

MC_KWARG_NAMES: tuple[str, ...] = ("atol", "rtol", "max_its", "batch_size")
"""Kwargs reserved for the adaptive-MC machinery."""

DEFAULT_ATOL: float = 1e-3
DEFAULT_RTOL: float = 1e-2
DEFAULT_MAX_ITS: int = 100_000
DEFAULT_BATCH_SIZE: int = 200


def split_mc_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition kwargs into ``(mc_kwargs, dist_kwargs)``.

    Used by overriders so that MC-control kwargs (``atol``, ``rtol``,
    ``max_its``, ``batch_size``) are kept locally, while everything
    else is forwarded to a discovered analytic backend.
    """

    mc_set = set(MC_KWARG_NAMES)
    mc = {k: v for k, v in kwargs.items() if k in mc_set}
    dist = {k: v for k, v in kwargs.items() if k not in mc_set}
    return mc, dist


# ---------------------------------------------------------------------------
# Analytic-attribute discovery (for scipy.stats-style backends).
# ---------------------------------------------------------------------------


def try_analytic(dist: Any, name: str) -> tuple[Any, bool]:
    """Try to obtain ``dist.<name>`` (callable or attribute).

    Returns ``(value, True)`` if the attribute exists and yields a
    value without raising :class:`NotImplementedError` /
    :class:`TypeError`; ``(None, False)`` otherwise.

    Calling convention: if the attribute is callable, calls it with no
    arguments; otherwise returns the attribute directly.  Handles the
    scipy.stats heterogeneity in which univariate frozen distributions
    expose ``.mean()`` as a method whereas
    :class:`scipy.stats.multivariate_normal` exposes ``.mean`` as an
    attribute array.
    """

    attr = getattr(dist, name, None)
    if attr is None:
        return None, False
    if callable(attr):
        try:
            return attr(), True
        except (NotImplementedError, TypeError):
            return None, False
    return attr, True


# ---------------------------------------------------------------------------
# Aggregators: dispatch on return type, compute (running mean, running SE).
# ---------------------------------------------------------------------------


def aggregate(samples: list[Any]) -> tuple[Any, np.ndarray]:
    """Compute ``(mean, MC SE)`` over a list of consistent-type samples.

    The mean is returned in the original sample type (preserving
    DataFrame index / columns, and subclass identity for DataMat etc.);
    the SE is always a numpy array (used only for the convergence
    check, not handed back to the user).
    """

    first = samples[0]
    if isinstance(first, np.ndarray):
        return _np_aggregate(samples)
    if isinstance(first, int | float | np.number):
        return _np_aggregate([np.asarray(s) for s in samples])
    # Pandas (and subclasses such as DataMat / DataVec) via optional import.
    try:
        import pandas as pd  # noqa: PLC0415  (lazy: pandas is optional)
    except ImportError:
        pd = None
    if pd is not None:
        if isinstance(first, pd.DataFrame):
            return _pd_frame_aggregate(samples)
        if isinstance(first, pd.Series):
            return _pd_series_aggregate(samples)
    raise NotImplementedError(
        f"aggregate: cannot aggregate samples of type "
        f"{type(first).__name__}.  Supported return types: numpy "
        f"array, python scalar, pandas DataFrame (incl. DataMat), "
        f"pandas Series (incl. DataVec).  If your func returns a "
        f"heterogeneous structure (e.g., a list from a TwoStageDGP), "
        f"compose it to flatten to a consistent shape first."
    )


def _np_aggregate(samples: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Numpy-array aggregation along a stacked leading axis."""

    arrs = [np.asarray(s) for s in samples]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(
            f"aggregate: samples have inconsistent shapes {shapes}; "
            f"expected all-equal."
        )
    stack = np.stack(arrs, axis=0)
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    if n < 2:
        se = np.full_like(np.asarray(mean, dtype=float), np.inf)
    else:
        se = stack.std(axis=0, ddof=1) / np.sqrt(n)
    return mean, np.asarray(se, dtype=float)


def _pd_frame_aggregate(samples: list[Any]) -> tuple[Any, np.ndarray]:
    """DataFrame aggregation (incl. ``datamat.DataMat`` by inheritance)."""

    first = samples[0]
    for k, s in enumerate(samples[1:], start=1):
        if not first.index.equals(s.index):
            raise ValueError(
                f"aggregate: sample {k} index differs from sample 0; "
                f"expected identical labels across MC samples."
            )
        if not first.columns.equals(s.columns):
            raise ValueError(
                f"aggregate: sample {k} columns differ from sample 0; "
                f"expected identical labels across MC samples."
            )
    arr = np.stack([s.to_numpy(dtype=float) for s in samples], axis=0)
    n = arr.shape[0]
    mean_arr = arr.mean(axis=0)
    if n < 2:
        se = np.full_like(mean_arr, np.inf)
    else:
        se = arr.std(axis=0, ddof=1) / np.sqrt(n)
    mean = type(first)(mean_arr, index=first.index, columns=first.columns)
    return mean, np.asarray(se, dtype=float)


def _pd_series_aggregate(samples: list[Any]) -> tuple[Any, np.ndarray]:
    """Series aggregation (incl. ``datamat.DataVec`` by inheritance)."""

    first = samples[0]
    for k, s in enumerate(samples[1:], start=1):
        if not first.index.equals(s.index):
            raise ValueError(
                f"aggregate: sample {k} index differs from sample 0; "
                f"expected identical labels across MC samples."
            )
    arr = np.stack([s.to_numpy(dtype=float) for s in samples], axis=0)
    n = arr.shape[0]
    mean_arr = arr.mean(axis=0)
    if n < 2:
        se = np.full_like(mean_arr, np.inf)
    else:
        se = arr.std(axis=0, ddof=1) / np.sqrt(n)
    mean = type(first)(mean_arr, index=first.index)
    return mean, np.asarray(se, dtype=float)


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce a possibly-DataFrame mean to a numpy array for tol comparison."""

    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, int | float | np.number):
        return np.asarray(x, dtype=float)
    if hasattr(x, "to_numpy"):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


def _converged(mean: Any, se: np.ndarray, atol: float, rtol: float) -> bool:
    """``numpy.allclose``-style convergence check on the MC SE."""

    mean_arr = _to_numpy(mean)
    thresh = atol + rtol * np.abs(mean_arr)
    return bool(np.all(se <= thresh))


# ---------------------------------------------------------------------------
# Adaptive MC driver.
# ---------------------------------------------------------------------------


def adaptive_mc(
    sampler: Callable[[], Any],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    max_its: int = DEFAULT_MAX_ITS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    context: str = "expect",
) -> Any:
    """Generic adaptive batched MC estimator.

    ``sampler`` is called repeatedly (zero arguments); its return
    values are aggregated.  Convergence: max(SE) <= atol + rtol *
    max(|mean|).  On budget exhaustion, emits
    :class:`NumericalWarning` and returns the best available estimate.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}.")
    if max_its < 1:
        raise ValueError(f"max_its must be >= 1; got {max_its}.")

    samples: list[Any] = []
    mean: Any = None
    se: np.ndarray = np.array([np.inf])
    while len(samples) < max_its:
        target = min(len(samples) + batch_size, max_its)
        while len(samples) < target:
            samples.append(sampler())
        mean, se = aggregate(samples)
        if len(samples) >= 2 and _converged(mean, se, atol, rtol):
            return mean
    warnings.warn(
        (
            f"{context}: max_its={max_its} reached with max MC SE "
            f"{float(np.max(se)):.3g} exceeding tol "
            f"(atol={atol}, rtol={rtol}); returning best estimate."
        ),
        NumericalWarning,
        stacklevel=3,
    )
    return mean
