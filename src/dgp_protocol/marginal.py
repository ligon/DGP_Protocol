"""Per-observation marginal expectation -- the P-side surface.

Reading P (analog-estimation framing per Manski 1988): ``func`` takes
a *single observation* and returns a scalar / vector.  ``expect(dgp,
func)`` returns

    E_F[func(X)]

where F is the per-observation marginal distribution implied by
``dgp`` and X is one observation drawn from F.

A "single observation" is one row of the tabular ``(N, p)``-shaped
draw produced by ``dgp.draw()`` -- or, for scalar / 1-d draws, one
element along the leading axis.  This interpretation **implicitly
assumes rows are iid** under F.  For DGPs where that holds
(:class:`~dgp_protocol.empirical.EmpiricalDGP` with
:class:`~dgp_protocol.sampling.IIDSampling`,
:class:`~dgp_protocol.parametric.ParametricDGP` with iid generators),
the marginal-of-row answer is the analog-estimation primitive.  For
non-iid DGPs (clustered, hierarchical), per-observation marginal
moments are mathematically defined but lose their analog-estimation
interpretation; the corresponding concrete container raises
:class:`NotImplementedError` to make this visible at the call site.
Dataset-level operations belong on the parallel surface,
:attr:`dgp.sample_distribution`.

Dispatch (free function pattern, per ``AGENTS.md`` §5):

1. If the DGP defines ``expect`` / ``mean`` / ``var`` / ``cov``, the
   free function calls that method (forwarding all kwargs).
2. If the method raises
   :class:`~dgp_protocol.exceptions.AnalyticUnavailable`, the free
   function falls back to adaptive batched Monte Carlo over rows of
   repeated ``dgp.draw()`` calls.
3. If the method raises :class:`NotImplementedError`, the free
   function propagates -- the DGP has refused the operation.
4. If the DGP does not define the method at all, the free function
   falls back to MC.

Mean/var/cov are sugar over :func:`expect`; concrete containers may
override any of them with their own analytic shortcuts.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from ._mc import (
    DEFAULT_ATOL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_ITS,
    DEFAULT_RTOL,
    adaptive_mc,
    split_mc_kwargs,
)
from .exceptions import AnalyticUnavailable
from .warnings import NumericalWarning


def _rows_of(draw: Any) -> Iterator[Any]:
    """Iterate observations within a single draw.

    - 0-d ndarray / python scalar: yields the scalar value.
    - 1-d ndarray: yields each element (treated as scalar observations).
    - 2+-d ndarray: yields each row along the leading axis.
    - pandas Series: yields each scalar element along the index.
    - pandas DataFrame: yields each row as a Series.
    - python ``list``: raises :class:`TypeError` (e.g., a
      :class:`~dgp_protocol.composition.TwoStageDGP` draw is a list
      of per-cluster arrays without a single row-shape -- use
      :attr:`dgp.sample_distribution` for dataset-level operations).
    """

    if isinstance(draw, list):
        raise TypeError(
            "expect / mean / var / cov: cannot iterate observations from "
            "a list-of-arrays draw (typical of TwoStageDGP).  Use "
            "dgp.sample_distribution methods for dataset-level operations "
            "on heterogeneous draws."
        )
    # Pandas handling: optional, lazy import.
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        pd = None
    if pd is not None and isinstance(draw, pd.DataFrame):
        for _, row in draw.iterrows():
            yield row
        return
    if pd is not None and isinstance(draw, pd.Series):
        yield from draw
        return
    arr = np.asarray(draw)
    if arr.ndim == 0:
        yield arr.item()
        return
    yield from arr


def _adaptive_mc_marginal(
    dgp: Any,
    func: Callable[[Any], Any],
    *,
    atol: float,
    rtol: float,
    max_its: int,
    batch_size: int,
) -> Any:
    """Adaptive batched MC: average ``func(row)`` over rows of repeated draws.

    Each call to the internal sampler advances through the rows of the
    current draw; when the rows are exhausted, a fresh ``dgp.draw()``
    is requested.  The convergence budget (``max_its``) counts
    individual ``func`` evaluations, not draws.
    """

    row_iter: Iterator[Any] = iter(())

    def sampler() -> Any:
        nonlocal row_iter
        while True:
            try:
                return func(next(row_iter))
            except StopIteration:
                row_iter = _rows_of(dgp.draw())

    return adaptive_mc(
        sampler,
        atol=atol,
        rtol=rtol,
        max_its=max_its,
        batch_size=batch_size,
        context="expect",
    )


def _warn_unused_dist_kwargs(name: str, dist_kw: dict[str, Any]) -> None:
    if dist_kw:
        warnings.warn(
            (
                f"{name}: backend kwargs {sorted(dist_kw)} ignored; "
                f"no analytic backend available, falling back to Monte Carlo."
            ),
            NumericalWarning,
            stacklevel=3,
        )


def expect(
    dgp: Any,
    func: Callable[[Any], Any],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    max_its: int = DEFAULT_MAX_ITS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **dist_kwargs: Any,
) -> Any:
    """``E_F[func(X)]`` where X is one observation under the marginal.

    See module docstring for the dispatch rules.  Extra kwargs are
    forwarded to ``dgp.expect`` when the analytic path is available
    (e.g., scipy's ``lb`` / ``ub`` / ``conditional``); on the MC
    fallback they trigger a :class:`NumericalWarning`.
    """

    if hasattr(dgp, "expect"):
        try:
            return dgp.expect(
                func,
                atol=atol,
                rtol=rtol,
                max_its=max_its,
                batch_size=batch_size,
                **dist_kwargs,
            )
        except AnalyticUnavailable:
            pass
    _warn_unused_dist_kwargs("expect", dist_kwargs)
    return _adaptive_mc_marginal(
        dgp,
        func,
        atol=atol,
        rtol=rtol,
        max_its=max_its,
        batch_size=batch_size,
    )


def mean(dgp: Any, **kwargs: Any) -> Any:
    """``E_F[X]`` -- the per-observation marginal mean."""

    if hasattr(dgp, "mean"):
        try:
            return dgp.mean(**kwargs)
        except AnalyticUnavailable:
            pass
    mc_kw, dist_kw = split_mc_kwargs(kwargs)
    _warn_unused_dist_kwargs("mean", dist_kw)
    return _adaptive_mc_marginal(
        dgp,
        lambda x: x,
        atol=mc_kw.get("atol", DEFAULT_ATOL),
        rtol=mc_kw.get("rtol", DEFAULT_RTOL),
        max_its=mc_kw.get("max_its", DEFAULT_MAX_ITS),
        batch_size=mc_kw.get("batch_size", DEFAULT_BATCH_SIZE),
    )


def var(dgp: Any, **kwargs: Any) -> Any:
    """``Var_F[X]`` -- per-coordinate variance under the marginal.

    Default MC fallback is two-pass: ``E[X^2] - E[X]^2`` via two
    adaptive-MC calls.  Concrete containers with analytic forms
    (e.g., EmpiricalDGP under IIDSampling) override this and return
    the closed-form answer in one call.
    """

    if hasattr(dgp, "var"):
        try:
            return dgp.var(**kwargs)
        except AnalyticUnavailable:
            pass
    mc_kw, dist_kw = split_mc_kwargs(kwargs)
    _warn_unused_dist_kwargs("var", dist_kw)
    mc_call = {
        "atol": mc_kw.get("atol", DEFAULT_ATOL),
        "rtol": mc_kw.get("rtol", DEFAULT_RTOL),
        "max_its": mc_kw.get("max_its", DEFAULT_MAX_ITS),
        "batch_size": mc_kw.get("batch_size", DEFAULT_BATCH_SIZE),
    }
    mean_val = _adaptive_mc_marginal(dgp, lambda x: x, **mc_call)
    second = _adaptive_mc_marginal(dgp, _elementwise_square, **mc_call)
    return second - _elementwise_square(mean_val)


def cov(dgp: Any, **kwargs: Any) -> Any:
    """``Cov_F[X]`` -- covariance matrix under the marginal.

    Default MC fallback is two-pass: ``E[XX^T] - E[X]E[X]^T`` via two
    adaptive-MC calls.  For scalar marginals returns a scalar
    variance; for vector marginals returns a square matrix.
    """

    if hasattr(dgp, "cov"):
        try:
            return dgp.cov(**kwargs)
        except AnalyticUnavailable:
            pass
    mc_kw, dist_kw = split_mc_kwargs(kwargs)
    _warn_unused_dist_kwargs("cov", dist_kw)
    mc_call = {
        "atol": mc_kw.get("atol", DEFAULT_ATOL),
        "rtol": mc_kw.get("rtol", DEFAULT_RTOL),
        "max_its": mc_kw.get("max_its", DEFAULT_MAX_ITS),
        "batch_size": mc_kw.get("batch_size", DEFAULT_BATCH_SIZE),
    }
    mean_val = _adaptive_mc_marginal(dgp, lambda x: x, **mc_call)
    mean_arr = np.atleast_1d(np.asarray(mean_val, dtype=float))
    outer = _adaptive_mc_marginal(dgp, _outer_self, **mc_call)
    outer_arr = np.atleast_2d(np.asarray(outer, dtype=float))
    cov_arr = outer_arr - np.outer(mean_arr, mean_arr)
    if mean_arr.size == 1:
        return float(cov_arr.flat[0])
    return cov_arr


def _elementwise_square(x: Any) -> Any:
    """``x`` times ``x`` element-wise, preserving scalar-ness."""

    if isinstance(x, np.ndarray):
        return x * x
    if isinstance(x, int | float | np.number):
        return x * x
    return x * x


def _outer_self(x: Any) -> Any:
    """``x x^T`` (or x*x for scalars) for cov MC fallback."""

    arr = np.atleast_1d(np.asarray(x, dtype=float))
    return np.outer(arr, arr)
