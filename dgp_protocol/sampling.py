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

JAX dispatch
------------
``moment_covariance_estimator`` is array-namespace polymorphic: if
the moment function ``gi`` returns a JAX array (e.g. when a consumer
like ManifoldGMM is computing the criterion through a JAX trace),
the entire ``Omega`` computation runs in ``jnp`` (segment-sum,
eigendecomposition, PSD projection), preserving autodiff through
``theta``.  When ``gi`` returns numpy, the existing numpy path runs
bit-for-bit unchanged.  JAX is an optional dependency: detection
checks the module of ``type(arr)`` without importing ``jax``.
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

    Optional members consulted by the compositional surface in
    :mod:`dgp_protocol.sample_distribution`:

    - :meth:`cluster_score_blocks`: the per-i.i.d.-unit centered
      moment-score blocks ``(G, k)`` satisfying
      ``blocks.T @ blocks == hat Omega``.  When supplied, downstream
      composites (e.g. :class:`~dgp_protocol.TwoStageDGP`) can derive
      cluster-robust ``hat Omega`` for hierarchical compositions
      without round-tripping through the full moment-covariance matrix.
      Implementations of :class:`SamplingDesign` that do not supply
      ``cluster_score_blocks`` are still usable via the existing
      :meth:`moment_covariance_estimator` surface.
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
    ) -> Any:
        """Closed-form estimate of ``Var_DGP[bar g_N(theta)]``."""


# ---------------------------------------------------------------------------
# Array-namespace dispatch (numpy vs jax.numpy).
# ---------------------------------------------------------------------------


def _is_jax_array(arr: Any) -> bool:
    """Detect a JAX array without importing ``jax``.

    Returns True when ``type(arr).__module__`` lives under the ``jax``
    or ``jaxlib`` package roots, False otherwise (including the case
    where ``jax`` is not installed).  Robust to JAX internal-module
    refactoring because we match on the top-level package, not on the
    concrete class.
    """

    cls = type(arr)
    root = cls.__module__.split(".", 1)[0]
    return root in {"jax", "jaxlib"}


def _array_namespace(arr: Any) -> Any:
    """Return the array module (``numpy`` or ``jax.numpy``) for ``arr``.

    Imports ``jax.numpy`` lazily on first JAX array encountered so the
    package's "numpy + cloudpickle only" runtime baseline is preserved
    in the common case.
    """

    if _is_jax_array(arr):
        import jax.numpy as jnp

        return jnp
    return np


# ---------------------------------------------------------------------------
# Shared helpers (private).  All accept an ``xp`` namespace so they can
# operate on numpy OR jax.numpy arrays uniformly.
# ---------------------------------------------------------------------------


def _evaluate_moments(
    gi: Callable[[Any, Any], Any], theta: Any, observation: Any
) -> Any:
    """Evaluate ``gi(theta, observation)`` and reshape to ``(N, k)``.

    Returns whatever array type ``gi`` produces (numpy or JAX), without
    a forced ``np.asarray`` coercion.  Callers downstream pick the
    matching ``xp`` namespace via :func:`_array_namespace`.
    """

    raw = gi(theta, observation)
    xp = _array_namespace(raw)
    arr = xp.asarray(raw)
    # Cast non-float dtypes (e.g. int) to float for the divide / sqrt below;
    # leave float32 / float64 alone.
    if not xp.issubdtype(arr.dtype, xp.floating):
        arr = arr.astype(xp.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _per_column_counts(moments: Any, xp: Any) -> Any:
    """Per-column non-NaN counts; length ``k``."""

    return (~xp.isnan(moments)).sum(axis=0)


def _maybe_weighted_mean(moments: Any, weights: Any | None, xp: Any) -> Any:
    """Unweighted ``nanmean`` (axis 0) or weighted mean ``sum(w * g) / N``.

    The weighted convention divides by ``N``, not ``sum(w)``, to
    preserve unbiasedness when ``E[w_i] = 1`` (matches ManifoldGMM's
    ``MomentRestriction._weighted_mean``).
    """

    if weights is None:
        return xp.nanmean(moments, axis=0)
    n = moments.shape[0]
    w = xp.asarray(weights, dtype=moments.dtype).reshape(n, 1)
    return xp.nansum(w * moments, axis=0) / n


def _project_psd(matrix: Any, xp: Any) -> Any:
    """Symmetrise + eigenvalue-clip to enforce numerical PSD.

    Mirrors ManifoldGMM's ``_project_psd_numpy``: returns
    ``V diag(max(0, lambda)) V^T`` after symmetrising
    ``(matrix + matrix.T) / 2``.  Defensive against the tiny negative
    eigenvalues that finite-precision matrix products produce.
    """

    symmetrised = 0.5 * (matrix + matrix.T)
    linalg = xp.linalg
    eigenvalues, eigenvectors = linalg.eigh(symmetrised)
    clipped = xp.clip(eigenvalues, 0.0, None)
    return eigenvectors @ xp.diag(clipped) @ eigenvectors.T


def _centered_scaled(moments: Any, weights: Any | None, centered: bool, xp: Any) -> Any:
    """Return ``(centered) / sqrt(N_k)`` per column.

    ``N_k`` is the per-column non-NaN count.  When ``centered`` is
    False, no mean subtraction (the moments are taken as already
    centered or under a null where ``E[g] = 0`` does not require it).
    """

    counts = _per_column_counts(moments, xp).astype(moments.dtype)
    # Avoid division-by-zero for empty columns.
    safe_counts = xp.where(counts > 0, counts, 1.0)
    scale = xp.sqrt(safe_counts)
    if centered:
        mean = _maybe_weighted_mean(moments, weights, xp)
        centered_mat = moments - mean.reshape(1, -1)
    else:
        centered_mat = moments
    return centered_mat / scale.reshape(1, -1)


def _cluster_codes_numpy(cluster_ids: Any) -> tuple[np.ndarray, int]:
    """Resolve raw ``cluster_ids`` to contiguous int64 codes 0..G-1.

    Host-side numpy computation; ``cluster_ids`` is a property of the
    sampling design (not a function of theta), so this is fine to do
    outside any autodiff trace.  Returns ``(codes, num_clusters)``.
    """

    ids = np.asarray(cluster_ids)
    if ids.ndim != 1:
        ids = ids.reshape(-1)
    _, codes = np.unique(ids, return_inverse=True)
    codes = np.asarray(codes, dtype=np.int64)
    G = int(codes.max()) + 1 if codes.size else 0
    return codes, G


def _group_sum(scaled: Any, codes_np: np.ndarray, num_clusters: int, xp: Any) -> Any:
    """Aggregate ``scaled`` rows by cluster code into a ``(G, k)`` array.

    JAX path uses :func:`jax.ops.segment_sum`; numpy path uses
    :func:`numpy.add.at`.  Both treat ``NaN`` entries as zero (NaN are
    masked out at the call site).
    """

    if xp is np:
        cleaned = np.where(np.isnan(scaled), 0.0, scaled)
        grouped = np.zeros((num_clusters, cleaned.shape[1]), dtype=cleaned.dtype)
        np.add.at(grouped, codes_np, cleaned)
        return grouped

    # JAX path.
    from jax.ops import segment_sum

    codes = xp.asarray(codes_np)
    cleaned = xp.where(xp.isnan(scaled), 0.0, scaled)
    return segment_sum(cleaned, codes, num_segments=num_clusters)


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

    def cluster_score_blocks(
        self,
        observation: Any,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
    ) -> Any:
        """Centered, ``1/sqrt(N)``-scaled per-row moment scores ``(N, k)``.

        Each row is its own i.i.d. unit under :class:`IIDSampling`, so
        ``blocks.T @ blocks`` (with PSD projection) is exactly
        :meth:`moment_covariance_estimator`'s output.
        """

        moments = _evaluate_moments(gi, theta, observation)
        xp = _array_namespace(moments)
        return _centered_scaled(moments, self.weights, centered, xp)

    def moment_covariance_estimator(
        self,
        observation: Any,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
    ) -> Any:
        """``hat Omega = scaled^T scaled`` where ``scaled_i = (g_i - bar g) / sqrt(N)``.

        Equivalent to ``(1/N) sum_i (g_i - bar g)(g_i - bar g)^T`` with
        per-column ``N_k`` accounting for NaN entries.  Returns a JAX
        array iff ``gi(theta, observation)`` returns one; otherwise
        returns a numpy array.
        """

        blocks = self.cluster_score_blocks(observation, theta, gi, centered=centered)
        xp = _array_namespace(blocks)
        omega = blocks.T @ blocks
        return _project_psd(omega, xp)


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

    def cluster_score_blocks(
        self,
        observation: Any,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
    ) -> Any:
        """Centered, ``1/sqrt(N)``-scaled per-cluster moment-sum scores ``(G, k)``.

        Each row is one cluster's i.i.d. contribution under the cluster
        bootstrap, so ``blocks.T @ blocks`` (with PSD projection) is
        exactly :meth:`moment_covariance_estimator`'s output.
        """

        moments = _evaluate_moments(gi, theta, observation)
        xp = _array_namespace(moments)

        # Host-side cluster-id normalisation (theta-independent).
        codes_np, num_clusters = _cluster_codes_numpy(self.cluster_ids)
        if codes_np.size != moments.shape[0]:
            raise ValueError(
                f"cluster_ids has {codes_np.size} entries; expected "
                f"{moments.shape[0]} (one per observation)."
            )

        scaled = _centered_scaled(moments, self.weights, centered, xp)
        return _group_sum(scaled, codes_np, num_clusters, xp)

    def moment_covariance_estimator(
        self,
        observation: Any,
        theta: Any,
        gi: Callable[[Any, Any], Any],
        *,
        centered: bool = True,
    ) -> Any:
        """Cluster-robust sandwich covariance of the moment vector.

        Returns a JAX array iff ``gi(theta, observation)`` returns one;
        otherwise returns a numpy array.  The cluster codes are
        resolved in numpy (host-side, one-time per call) because
        ``cluster_ids`` is a property of the design and is not a
        function of ``theta``; passing them through ``np.unique`` is
        not on any autodiff path.
        """

        blocks = self.cluster_score_blocks(observation, theta, gi, centered=centered)
        xp = _array_namespace(blocks)
        omega = blocks.T @ blocks
        return _project_psd(omega, xp)
