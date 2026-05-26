"""Sampling-design helpers for :class:`EmpiricalDGP`.

These objects describe the *dependence structure* of an observed
sample and provide two recipes consistent with that structure:

- :meth:`SamplingDesign.bootstrap_resample` -- produces a fresh
  realization from an observed dataset (cluster-block bootstrap,
  iid multinomial bootstrap, etc.).
- :meth:`SamplingDesign.moment_covariance_estimator` -- the analog-
  estimation moment-vector sampling covariance
  :math:`\\hat{\\Omega}(\\theta) = \\widehat{\\operatorname{Var}}_{DGP}
  [\\bar g_N(\\theta)]`.  Closed-form (no MC) for the empirical
  distribution under the given sampling design; consumers
  (ManifoldGMM's ``omega_hat``, free
  :meth:`SampleDistribution.moment_covariance`) prefer it over the
  MC fallback when available.

Default is :class:`IIDSampling` (independent rows).
:class:`ClusteredSampling` provides cluster-block bootstrap and the
cluster-robust sandwich covariance for data with intra-cluster
correlation.  Both designs accept an optional ``weights`` field for
per-observation sampling weights -- following the convention used
in the v1 ManifoldGMM ``MomentRestriction.omega_hat`` we port from,
weights affect *centering* only (the weighted-mean used to subtract
:math:`\\bar g` before forming the outer product) and not the outer-
product weighting itself.  See module-level constants for the
formula.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SamplingDesign(Protocol):
    """A bootstrap-resampling recipe + analytic moment-covariance estimator.

    Required members:

    - :meth:`bootstrap_resample`: produces a fresh realization
      consistent with the assumed dependence structure.
    - :meth:`moment_covariance_estimator`: closed-form
      ``Var_DGP[bar g_N(theta)]`` for the empirical distribution
      under the sampling design (no Monte Carlo).
    """

    def bootstrap_resample(
        self,
        observation: Any,
        *,
        size: tuple[int, ...] | None,
        rng: np.random.Generator,
    ) -> Any:
        """Return a fresh realization sampled from ``observation``."""

    def moment_covariance_estimator(
        self,
        observation: Any,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
    ) -> np.ndarray:
        """Closed-form estimate of ``Var_DGP[bar g_N(theta)]``."""


# ---------------------------------------------------------------------------
# Shared helpers (private).
# ---------------------------------------------------------------------------


def _evaluate_moments(
    gi: Callable[[Any, Any], Any], theta: Any, observation: Any
) -> np.ndarray:
    """Evaluate ``gi(theta, observation)`` and coerce to a float ``(N, k)`` array."""

    raw = gi(theta, observation)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _per_column_counts(moments: np.ndarray) -> np.ndarray:
    """Per-column non-NaN counts; length ``k``."""

    return (~np.isnan(moments)).sum(axis=0)


def _maybe_weighted_mean(moments: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Unweighted ``nanmean`` (axis 0) or weighted mean ``sum(w * g) / N``.

    The weighted convention divides by ``N``, not ``sum(w)``, to
    preserve unbiasedness when ``E[w_i] = 1`` (matches ManifoldGMM's
    ``MomentRestriction._weighted_mean``).
    """

    if weights is None:
        return np.nanmean(moments, axis=0)
    n = moments.shape[0]
    w = np.asarray(weights, dtype=float).reshape(n, 1)
    return np.nansum(w * moments, axis=0) / n


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    """Symmetrise + eigenvalue-clip to enforce numerical PSD.

    Mirrors ManifoldGMM's ``_project_psd_numpy``: returns
    ``V diag(max(0, lambda)) V^T`` after symmetrising
    ``(matrix + matrix.T) / 2``.  Defensive against the tiny negative
    eigenvalues that finite-precision matrix products produce.
    """

    symmetrised = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrised)
    clipped = np.clip(eigenvalues, 0.0, None)
    return eigenvectors @ np.diag(clipped) @ eigenvectors.T


def _centered_scaled(
    moments: np.ndarray, weights: np.ndarray | None, centered: bool
) -> np.ndarray:
    """Return ``(centered) / sqrt(N_k)`` per column.

    ``N_k`` is the per-column non-NaN count.  When ``centered`` is
    False, no mean subtraction (the moments are taken as already
    centered or under a null where ``E[g] = 0`` does not require it).
    """

    counts = _per_column_counts(moments).astype(float)
    # Avoid division-by-zero for empty columns.
    safe_counts = np.where(counts > 0, counts, 1.0)
    scale = np.sqrt(safe_counts)
    if centered:
        mean = _maybe_weighted_mean(moments, weights)
        centered_mat = moments - mean.reshape(1, -1)
    else:
        centered_mat = moments
    return centered_mat / scale.reshape(1, -1)


# ---------------------------------------------------------------------------
# Concrete sampling designs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IIDSampling:
    """Independent identically-distributed rows.

    The bootstrap is a multinomial resample of rows with replacement.
    The closed-form moment-covariance is the (centered) outer product
    ``(1/N) sum_i (g_i - bar g)(g_i - bar g)^T``.

    Parameters
    ----------
    weights:
        Optional ``(N,)``-shape per-observation sampling weights.
        When supplied, the mean used for centering is the weighted
        analog ``bar g^* = (1/N) sum w_i g_i`` (division by ``N``,
        not ``sum w``, preserves unbiasedness under ``E[w_i] = 1``).
        Weights do *not* enter the outer-product weighting (this
        matches the v1 ManifoldGMM convention we port; users wanting
        a fully Horvitz-Thompson-style weighted variance should
        compose at a higher layer).
    """

    weights: Any = field(default=None)

    def bootstrap_resample(
        self,
        observation: Any,
        *,
        size: tuple[int, ...] | None,
        rng: np.random.Generator,
    ) -> Any:
        arr = np.asarray(observation)
        target_n = size[0] if size is not None else arr.shape[0]
        idx = rng.integers(0, arr.shape[0], size=target_n)
        return arr[idx]

    def moment_covariance_estimator(
        self,
        observation: Any,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
    ) -> np.ndarray:
        """``hat Omega = scaled^T scaled`` where ``scaled_i = (g_i - bar g) / sqrt(N)``.

        Equivalent to ``(1/N) sum_i (g_i - bar g)(g_i - bar g)^T`` with
        per-column ``N_k`` accounting for NaN entries.
        """

        moments = _evaluate_moments(gi, theta, observation)
        scaled = _centered_scaled(moments, self.weights, centered)
        omega = scaled.T @ scaled
        return _project_psd(omega)


@dataclass(frozen=True)
class ClusteredSampling:
    """Observations clustered by integer label; iid across clusters,
    arbitrary correlation within.

    The bootstrap is a *cluster (block) bootstrap*: resample clusters
    with replacement, concatenate their member rows verbatim.  The
    closed-form moment-covariance is the cluster-robust sandwich
    ``hat Omega = sum_c S_c S_c^T / N`` where
    ``S_c = sum_{i in c} (g_i - bar g)`` and ``N_k`` is per-column
    non-NaN count (NaN entries are treated as zero contribution to
    the per-cluster sum, matching the v1 ManifoldGMM convention).
    Collapses to :class:`IIDSampling`'s formula when every cluster has
    size one.

    Parameters
    ----------
    cluster_ids:
        Length-``N`` integer or hashable array.  Each unique value
        identifies one cluster.  Cluster membership is fixed; the
        bootstrap resamples *whole clusters*, not within-cluster
        rows.
    weights:
        Optional ``(N,)``-shape per-observation sampling weights;
        same convention as :class:`IIDSampling.weights`.
    """

    cluster_ids: Any
    weights: Any = field(default=None)

    def bootstrap_resample(
        self,
        observation: Any,
        *,
        size: tuple[int, ...] | None,
        rng: np.random.Generator,
    ) -> Any:
        arr = np.asarray(observation)
        ids = np.asarray(self.cluster_ids)
        if ids.size != arr.shape[0]:
            raise ValueError(
                f"cluster_ids has {ids.size} entries; expected {arr.shape[0]} "
                "(one per observation)."
            )
        # Resolve to contiguous cluster codes 0..G-1.
        unique, codes = np.unique(ids, return_inverse=True)
        codes = np.asarray(codes, dtype=np.int64)
        G = unique.size
        target_G = size[0] if size is not None else G
        # Group rows by cluster code.
        rows_by_cluster: list[np.ndarray] = [np.where(codes == g)[0] for g in range(G)]
        # Resample G clusters with replacement (or target_G of them).
        sampled_clusters = rng.integers(0, G, size=target_G)
        # Concatenate the rows of the sampled clusters.
        row_indices = np.concatenate([rows_by_cluster[g] for g in sampled_clusters])
        return arr[row_indices]

    def moment_covariance_estimator(
        self,
        observation: Any,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
    ) -> np.ndarray:
        """Cluster-robust sandwich covariance of the moment vector."""

        moments = _evaluate_moments(gi, theta, observation)
        ids = np.asarray(self.cluster_ids)
        if ids.size != moments.shape[0]:
            raise ValueError(
                f"cluster_ids has {ids.size} entries; expected "
                f"{moments.shape[0]} (one per observation)."
            )
        scaled = _centered_scaled(moments, self.weights, centered)
        # NaN -> 0 so missing entries don't contaminate the cluster sum
        # (matches ManifoldGMM's _group_sum cleanup).
        cleaned = np.where(np.isnan(scaled), 0.0, scaled)
        # Resolve to contiguous codes 0..G-1 and accumulate per-cluster sums.
        _, codes = np.unique(ids, return_inverse=True)
        codes = np.asarray(codes, dtype=np.int64)
        num_clusters = int(codes.max()) + 1 if codes.size else 0
        grouped = np.zeros((num_clusters, cleaned.shape[1]), dtype=cleaned.dtype)
        np.add.at(grouped, codes, cleaned)
        omega = grouped.T @ grouped
        return _project_psd(omega)
