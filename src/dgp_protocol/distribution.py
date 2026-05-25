"""Distributional-feature mixin: ``expect``, ``mean``, ``var``, ``cov``.

The mixin adds analytic-when-available, Monte-Carlo-otherwise convenience
methods to DGPs.  Concrete DGPs override any method whose analytic form
they know (e.g., :class:`~dgp_protocol.empirical.EmpiricalDGP` knows its
``mean``/``var``/``cov`` exactly); the mixin's defaults sample via
``self.draw()`` until an element-wise MC standard-error tolerance is
met.

Adaptive convergence
--------------------
:meth:`DistributionalFeatures.expect` draws batches of ``batch_size``
samples until::

    max(se) <= atol + rtol * max(|mean|)

(``numpy.allclose`` style, applied element-wise on array / DataFrame /
DataMat returns).  Defaults are ``atol=1e-3``, ``rtol=1e-2``,
``max_its=100_000``, ``batch_size=200``.  If the iteration budget is
exhausted without convergence, a
:class:`~dgp_protocol.warnings.NumericalWarning` is emitted and the
best available estimate is returned.

Returns from ``func``
---------------------
The internal aggregator handles:

- numpy scalars / arrays of consistent shape across MC samples;
- python ``int`` / ``float`` (treated as 0-d arrays);
- :class:`pandas.DataFrame` (and subclasses, including
  ``datamat.DataMat``) of identical index / columns across samples;
- :class:`pandas.Series` (and subclasses, including ``datamat.DataVec``).

Other return types raise :class:`NotImplementedError` with a hint.

scipy.stats-style analytic backends
-----------------------------------
If a DGP exposes a ``.distribution`` attribute (the convention adopted
by :class:`~dgp_protocol.parametric.ParametricDGP` when constructed
with ``distribution=``), the mixin preferentially delegates to the
distribution's analytic methods:

- ``dgp.expect(func, **kwargs)`` -> ``dist.expect(func, **kwargs)`` if
  available (kwargs forwarded; e.g., scipy's ``lb``, ``ub``, ``args``,
  ``conditional``).
- ``dgp.mean()`` -> ``dist.mean()`` (callable) or ``dist.mean``
  (attribute).
- ``dgp.var()`` -> ``dist.var`` analogously.
- ``dgp.cov()`` -> ``dist.cov`` analogously.

The dispatcher is duck-typed: any object with the relevant attributes
qualifies, including user-defined "distribution-like" classes.

Mixin overrides in concrete DGPs should accept and forward
``**kwargs`` to preserve the kwargs-passthrough contract; the mixin
exposes :func:`_split_mc_kwargs` for partitioning into MC-control vs
backend-specific kwargs.
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

# Kwargs reserved for the adaptive-MC machinery; anything else flows
# to the analytic backend.  Tuple, not frozenset, so the names are
# discoverable from a single source.
_MC_KWARG_NAMES: tuple[str, ...] = ("atol", "rtol", "max_its", "batch_size")

_DEFAULT_ATOL: float = 1e-3
_DEFAULT_RTOL: float = 1e-2
_DEFAULT_MAX_ITS: int = 100_000
_DEFAULT_BATCH_SIZE: int = 200


def _split_mc_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition kwargs into ``(mc_kwargs, dist_kwargs)``.

    Used by overriders so that MC-control kwargs (``atol``, ``rtol``,
    ``max_its``, ``batch_size``) are kept locally, while everything
    else is forwarded to a discovered analytic backend.
    """

    mc_set = set(_MC_KWARG_NAMES)
    mc = {k: v for k, v in kwargs.items() if k in mc_set}
    dist = {k: v for k, v in kwargs.items() if k not in mc_set}
    return mc, dist


# ---------------------------------------------------------------------------
# Analytic-attribute discovery.
# ---------------------------------------------------------------------------


def _try_analytic(dist: Any, name: str) -> tuple[Any, bool]:
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


def _aggregate(samples: list[Any]) -> tuple[Any, np.ndarray]:
    """Compute ``(mean, MC SE)`` over a list of consistent-type samples.

    The mean is returned in the original sample type (preserving
    DataFrame index / columns, and subclass identity for DataMat etc.);
    the SE is always a numpy array (it is used only for the
    convergence check, not handed back to the user).
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
        f"_aggregate: cannot aggregate samples of type "
        f"{type(first).__name__}.  Supported return types: numpy "
        f"array, python scalar, pandas DataFrame (incl. DataMat), "
        f"pandas Series (incl. DataVec).  If your func returns a "
        f"heterogeneous structure (e.g., a list from a TwoStageDGP), "
        f"compose it to flatten to a consistent shape before passing "
        f"to expect."
    )


def _np_aggregate(samples: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Numpy-array aggregation along a stacked leading axis."""

    arrs = [np.asarray(s) for s in samples]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(
            f"_np_aggregate: samples have inconsistent shapes {shapes}; "
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
                f"_pd_frame_aggregate: sample {k} index differs from "
                f"sample 0; expected identical labels across MC samples."
            )
        if not first.columns.equals(s.columns):
            raise ValueError(
                f"_pd_frame_aggregate: sample {k} columns differ from "
                f"sample 0; expected identical labels across MC samples."
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
                f"_pd_series_aggregate: sample {k} index differs from "
                f"sample 0; expected identical labels across MC samples."
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


def _adaptive_mc_expect(
    dgp: Any,
    func: Callable[[Any], Any],
    *,
    atol: float,
    rtol: float,
    max_its: int,
    batch_size: int,
) -> Any:
    """Adaptive batched MC estimator of ``E_DGP[func(X)]``."""

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
            samples.append(func(dgp.draw()))
        mean, se = _aggregate(samples)
        if len(samples) >= 2 and _converged(mean, se, atol, rtol):
            return mean
    # Hit max_its without convergence.
    warnings.warn(
        (
            f"expect: max_its={max_its} reached with max MC SE "
            f"{float(np.max(se)):.3g} exceeding tol "
            f"(atol={atol}, rtol={rtol}); returning best estimate."
        ),
        NumericalWarning,
        stacklevel=3,
    )
    return mean


# ---------------------------------------------------------------------------
# Per-draw reducers used by mean / var / cov defaults.
# ---------------------------------------------------------------------------


def _per_coord_mean(X: Any) -> Any:
    """Per-coordinate mean of a single draw.

    Treats the leading axis as the sample axis.  For 0-d / 1-d draws
    returns a scalar; for 2-d (rows x cols) returns a vector.
    """

    if hasattr(X, "mean") and np.ndim(X) >= 2:
        return X.mean(axis=0)
    arr = np.asarray(X)
    if arr.ndim <= 1:
        return float(arr.mean()) if arr.size else float("nan")
    return arr.mean(axis=0)


def _per_coord_var(X: Any) -> Any:
    """Per-coordinate sample variance (ddof=1) of a single draw."""

    arr = np.asarray(X, dtype=float)
    if arr.ndim <= 1:
        return float(arr.var(ddof=1)) if arr.size > 1 else float("nan")
    return arr.var(axis=0, ddof=1)


def _per_coord_cov(X: Any) -> Any:
    """Per-coordinate sample covariance of a single draw."""

    arr = np.asarray(X, dtype=float)
    if arr.ndim <= 1:
        return float(arr.var(ddof=1)) if arr.size > 1 else float("nan")
    return np.cov(arr, rowvar=False, ddof=1)


# ---------------------------------------------------------------------------
# Mixin.
# ---------------------------------------------------------------------------


class DistributionalFeatures:
    """Mixin adding ``expect`` / ``mean`` / ``var`` / ``cov`` to a DGP.

    The mixin assumes the inheriting class provides ``draw()`` (the
    :class:`~dgp_protocol.protocol.DataGeneratingProcess` contract).
    No state is required on ``self``; an optional ``.distribution``
    attribute, when present, drives analytic dispatch to scipy.stats-
    style objects.

    Subclasses may override any method with an analytic implementation.
    Overrides should continue to accept and forward ``**kwargs`` (with
    :func:`_split_mc_kwargs` to partition into MC-vs-backend kwargs).
    """

    def expect(
        self,
        func: Callable[[Any], Any],
        *,
        atol: float = _DEFAULT_ATOL,
        rtol: float = _DEFAULT_RTOL,
        max_its: int = _DEFAULT_MAX_ITS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        **dist_kwargs: Any,
    ) -> Any:
        """Estimate ``E[func(X)]`` under self's distribution.

        Tries ``self.distribution.expect(func, **dist_kwargs)`` first
        when a ``.distribution`` attribute is present (e.g.,
        :class:`ParametricDGP` constructed with ``distribution=``);
        falls back to adaptive batched Monte Carlo otherwise.  If the
        MC fallback is taken with non-empty ``dist_kwargs``, the
        unused kwargs raise :class:`NumericalWarning`.
        """

        dist = getattr(self, "distribution", None)
        if dist is not None:
            method = getattr(dist, "expect", None)
            if callable(method):
                try:
                    return method(func, **dist_kwargs)
                except (NotImplementedError, TypeError):
                    pass
        if dist_kwargs:
            warnings.warn(
                (
                    f"{type(self).__name__}.expect: backend kwargs "
                    f"{sorted(dist_kwargs)} ignored; no analytic backend "
                    f"available, falling back to Monte Carlo."
                ),
                NumericalWarning,
                stacklevel=2,
            )
        return _adaptive_mc_expect(
            self,
            func,
            atol=atol,
            rtol=rtol,
            max_its=max_its,
            batch_size=batch_size,
        )

    def mean(self, **kwargs: Any) -> Any:
        """``E[X]`` under self's distribution; per-coordinate for tabular X."""

        return self._analytic_or_mc("mean", _per_coord_mean, **kwargs)

    def var(self, **kwargs: Any) -> Any:
        """Per-coordinate variance under self's distribution."""

        return self._analytic_or_mc("var", _per_coord_var, **kwargs)

    def cov(self, **kwargs: Any) -> Any:
        """Per-coordinate covariance matrix under self's distribution."""

        return self._analytic_or_mc("cov", _per_coord_cov, **kwargs)

    def _analytic_or_mc(
        self,
        attr_name: str,
        reducer: Callable[[Any], Any],
        **kwargs: Any,
    ) -> Any:
        """Shared dispatcher: direct analytic attribute, else MC reducer.

        Direct analytic dispatch is attempted only when no backend
        kwargs are supplied (since a bare attribute access cannot honor
        them).  When backend kwargs are present, the call routes
        through :meth:`expect` so the analytic backend's ``expect``
        path can absorb them.
        """

        mc_kw, dist_kw = _split_mc_kwargs(kwargs)
        dist = getattr(self, "distribution", None)
        if dist is not None and not dist_kw:
            value, found = _try_analytic(dist, attr_name)
            if found:
                return value
        # Either no distribution, no analytic attribute, or backend
        # kwargs are present (route through expect to absorb them).
        if dist_kw:
            # Route through expect; it will try dist.expect(reducer, **dist_kw)
            # and either succeed or warn-and-MC.
            return self.expect(reducer, **mc_kw, **dist_kw)
        return _adaptive_mc_expect(
            self,
            reducer,
            atol=mc_kw.get("atol", _DEFAULT_ATOL),
            rtol=mc_kw.get("rtol", _DEFAULT_RTOL),
            max_its=mc_kw.get("max_its", _DEFAULT_MAX_ITS),
            batch_size=mc_kw.get("batch_size", _DEFAULT_BATCH_SIZE),
        )
