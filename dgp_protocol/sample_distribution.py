"""Dataset-level distribution view: ``SampleDistribution``.

Reading D (sampling-distribution framing): ``stat_func`` takes a
*whole realization* (the output of ``dgp.draw()``) and returns a
scalar / vector.  ``dgp.sample_distribution.expect(stat_func)``
returns

    E_DGP[stat_func(realization)]

-- the expected value of a sample statistic under the DGP's
distribution over datasets.  Cousin methods:

- :meth:`SampleDistribution.cov`: ``Cov_DGP[stat_func(realization)]``,
  two-pass MC by default.
- :meth:`SampleDistribution.moment_covariance`: the analog-estimation
  primitive ``Var_DGP[ḡ_N(θ)]`` for the sample moment vector
  ``ḡ_N = (1/sqrt(N)) Σ_i g_i(θ, X_i)``.  ManifoldGMM's ``omega_hat``
  consumes this when present.
- :meth:`SampleDistribution.cluster_score_blocks`: the per-i.i.d.-unit
  centered moment-score blocks ``(G, k)`` satisfying
  ``blocks.T @ blocks == hat Omega``.  This is the compositional
  primitive: hierarchical DGPs (e.g. :class:`~dgp_protocol.TwoStageDGP`)
  derive their own ``hat Omega`` by assembling per-cluster blocks
  from this surface on their inner DGPs.  ``moment_covariance``
  itself is now derived: ``blocks.T @ blocks`` with PSD projection.

Use cases:

- Sampling distributions of estimators / test statistics.
- Monte Carlo power studies.
- Cluster-robust / HAC moment-covariance estimators (when the DGP
  exposes a specialised override).

The view is constructed lazily by ``dgp.sample_distribution`` (a
``@property`` on each container).  Concrete DGPs may supply analytic
overrides via private hook methods on themselves
(``_sd_expect``, ``_sd_cov``, ``_sd_cluster_score_blocks``,
``_sd_moment_covariance``); the view detects them via
:func:`hasattr` and falls back to adaptive MC otherwise.
``_sd_cluster_score_blocks`` is preferred over ``_sd_moment_covariance``:
when both are defined the blocks hook drives ``moment_covariance``
(via ``blocks.T @ blocks`` + PSD projection), so concrete types only
need to provide the blocks hook and the matrix is derived.  The
``_sd_moment_covariance`` hook is retained for backward compat with
user-defined DGPs.  Following the dispatch convention shared with
:mod:`dgp_protocol.expect`:

- A hook that raises
  :class:`~dgp_protocol.exceptions.AnalyticUnavailable` triggers MC
  fallback.
- A hook that raises :class:`NotImplementedError` is propagated.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
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


def _warn_unused_dist_kwargs(name: str, dist_kw: dict[str, Any]) -> None:
    if dist_kw:
        warnings.warn(
            (
                f"sample_distribution.{name}: backend kwargs "
                f"{sorted(dist_kw)} ignored; no analytic backend "
                f"available, falling back to Monte Carlo."
            ),
            NumericalWarning,
            stacklevel=3,
        )


class SampleDistribution:
    """Dataset-level distribution view of a DGP.

    Wraps a :class:`~dgp_protocol.protocol.DataGeneratingProcess` and
    exposes operations whose argument is a whole realization (matrix)
    rather than a single observation (row).  Constructed by
    ``dgp.sample_distribution``; users normally do not instantiate
    directly.
    """

    __slots__ = ("_dgp",)

    def __init__(self, dgp: Any) -> None:
        self._dgp = dgp

    def __repr__(self) -> str:
        return f"SampleDistribution({self._dgp!r})"

    @property
    def dgp(self) -> Any:
        """The underlying DGP."""

        return self._dgp

    # ------------------------------------------------------------------
    # expect: E_DGP[stat_func(realization)]
    # ------------------------------------------------------------------
    def expect(
        self,
        stat_func: Callable[[Any], Any],
        *,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        max_its: int = DEFAULT_MAX_ITS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **dist_kwargs: Any,
    ) -> Any:
        """Estimate ``E_DGP[stat_func(realization)]``.

        Tries ``dgp._sd_expect(stat_func, **kw)`` if available;
        otherwise adaptive batched MC averaging ``stat_func(dgp.draw())``
        over repeated draws.
        """

        hook = getattr(self._dgp, "_sd_expect", None)
        if callable(hook):
            try:
                return hook(
                    stat_func,
                    atol=atol,
                    rtol=rtol,
                    max_its=max_its,
                    batch_size=batch_size,
                    **dist_kwargs,
                )
            except AnalyticUnavailable:
                pass
        _warn_unused_dist_kwargs("expect", dist_kwargs)
        return adaptive_mc(
            lambda: stat_func(self._dgp.draw()),
            atol=atol,
            rtol=rtol,
            max_its=max_its,
            batch_size=batch_size,
            context="sample_distribution.expect",
        )

    # ------------------------------------------------------------------
    # cov: Cov_DGP[stat_func(realization)]
    # ------------------------------------------------------------------
    def cov(
        self,
        stat_func: Callable[[Any], Any],
        **kwargs: Any,
    ) -> Any:
        """Estimate ``Cov_DGP[stat_func(realization)]``.

        Default: two-pass MC, ``E[ss^T] - E[s]E[s]^T`` where
        ``s = stat_func(realization)``.  Concrete DGPs may supply
        a single-pass / analytic override via ``_sd_cov``.
        """

        hook = getattr(self._dgp, "_sd_cov", None)
        if callable(hook):
            try:
                return hook(stat_func, **kwargs)
            except AnalyticUnavailable:
                pass
        mc_kw, dist_kw = split_mc_kwargs(kwargs)
        _warn_unused_dist_kwargs("cov", dist_kw)
        mc_call: dict[str, Any] = {
            "atol": mc_kw.get("atol", DEFAULT_ATOL),
            "rtol": mc_kw.get("rtol", DEFAULT_RTOL),
            "max_its": mc_kw.get("max_its", DEFAULT_MAX_ITS),
            "batch_size": mc_kw.get("batch_size", DEFAULT_BATCH_SIZE),
        }
        mean_s = self.expect(stat_func, **mc_call)
        mean_s_arr = np.atleast_1d(np.asarray(mean_s, dtype=float))

        def outer_sample(realization: Any) -> np.ndarray:
            s = np.atleast_1d(np.asarray(stat_func(realization), dtype=float))
            return np.outer(s, s)

        outer = self.expect(outer_sample, **mc_call)
        outer_arr = np.atleast_2d(np.asarray(outer, dtype=float))
        cov_arr = outer_arr - np.outer(mean_s_arr, mean_s_arr)
        if mean_s_arr.size == 1:
            return float(cov_arr.flat[0])
        return cov_arr

    # ------------------------------------------------------------------
    # cluster_score_blocks: (G, k) per-i.i.d.-unit centered moment scores.
    # ------------------------------------------------------------------
    def cluster_score_blocks(
        self,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Per-i.i.d.-unit centered moment-score blocks ``(G, k)``.

        Satisfies ``blocks.T @ blocks == hat Omega`` (up to PSD
        projection).  Tries ``dgp._sd_cluster_score_blocks(theta, gi,
        centered=..., **kw)`` if available; raises
        :class:`AnalyticUnavailable` otherwise.  The compositional
        primitive consumed by hierarchical DGPs to assemble their own
        ``hat Omega`` without losing the per-cluster structure.
        """

        hook = getattr(self._dgp, "_sd_cluster_score_blocks", None)
        if callable(hook):
            return hook(theta, gi, centered=centered, **kwargs)
        raise AnalyticUnavailable(
            f"{type(self._dgp).__name__}.cluster_score_blocks: no "
            f"_sd_cluster_score_blocks hook on the DGP.  Use "
            f"sample_distribution.moment_covariance for the full "
            f"matrix (which has its own MC fallback), or supply the "
            f"hook on the concrete DGP type."
        )

    # ------------------------------------------------------------------
    # moment_covariance: Var_DGP[g_bar_N(theta)]
    # ------------------------------------------------------------------
    def moment_covariance(
        self,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        **kwargs: Any,
    ) -> Any:
        """Estimate ``Var_DGP[g_bar_N(theta)]`` -- the analog-estimation
        moment-vector sampling covariance.

        ``g_bar_N(theta) = (1/sqrt(N)) Σ_i gi(theta, X_i)`` where the
        sum is over rows of one realization.  Returns the
        ``(k, k)``-shaped sampling covariance under the DGP's
        distribution over datasets.

        Dispatch order:

        1. ``dgp._sd_cluster_score_blocks(theta, gi, ...)`` if defined --
           returns ``(G, k)`` blocks; ``hat Omega = blocks.T @ blocks``
           with PSD projection.  This is the preferred analytic path.
        2. ``dgp._sd_moment_covariance(theta, gi, ...)`` if defined --
           legacy direct-matrix hook, retained for back-compat with
           user-defined DGPs.
        3. MC fallback: build ``g_bar`` per draw, take the sample
           covariance over draws.
        """

        from .sampling import _array_namespace, _project_psd  # local: avoid cycle

        blocks_hook = getattr(self._dgp, "_sd_cluster_score_blocks", None)
        if callable(blocks_hook):
            try:
                blocks = blocks_hook(theta, gi, **kwargs)
                xp = _array_namespace(blocks)
                omega = blocks.T @ blocks
                return _project_psd(omega, xp)
            except AnalyticUnavailable:
                pass

        hook = getattr(self._dgp, "_sd_moment_covariance", None)
        if callable(hook):
            try:
                return hook(theta, gi, **kwargs)
            except AnalyticUnavailable:
                pass

        def per_draw(realization: Any) -> np.ndarray:
            moments = np.asarray(gi(theta, realization), dtype=float)
            if moments.ndim < 2:
                moments = moments.reshape(-1, 1)
            n_rows = moments.shape[0]
            # g_bar_N = (1/sqrt(N)) sum_i g_i (kept on this scale so the
            # MC covariance estimator returns the omega_hat the design
            # note's analog moment-covariance is conventionally
            # normalised to).
            return moments.sum(axis=0) / np.sqrt(n_rows)

        return self.cov(per_draw, **kwargs)
