"""Sampling-design helpers for :class:`EmpiricalDGP`.

These objects describe the *dependence structure* of an observed
sample and provide the bootstrap-resampling recipe consistent with
that structure.  They are *internal helpers* for :class:`EmpiricalDGP`
and are **not** part of the public DGP Protocol -- consumers interact
with :class:`DataGeneratingProcess` instances, not with their internal
sampling-design objects.

Default is :class:`IIDSampling` (independent rows).
:class:`ClusteredSampling` provides cluster-block bootstrap for data
with intra-cluster correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SamplingDesign(Protocol):
    """A bootstrap-resampling recipe for an observed dataset.

    The single required method, :meth:`bootstrap_resample`, takes
    the observed dataset and produces a fresh realization consistent
    with the assumed dependence structure.
    """

    def bootstrap_resample(
        self,
        observation: Any,
        *,
        size: tuple[int, ...] | None,
        rng: np.random.Generator,
    ) -> Any:
        """Return a fresh realization sampled from ``observation``."""


@dataclass(frozen=True)
class IIDSampling:
    """Independent identically-distributed rows.

    The bootstrap is a multinomial resample of rows with replacement.
    Default sampling design for :class:`EmpiricalDGP`.
    """

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


@dataclass(frozen=True)
class ClusteredSampling:
    """Observations clustered by integer label; iid across clusters,
    arbitrary correlation within.

    The bootstrap is a *cluster (block) bootstrap*: resample clusters
    with replacement, concatenate their member rows verbatim.  Cluster
    sizes vary; the resulting matrix can have different total length
    from the original.

    Parameters
    ----------
    cluster_ids:
        Length-N integer or hashable array.  Each unique value
        identifies one cluster.  Cluster membership is fixed; the
        bootstrap resamples *whole clusters*, not within-cluster
        rows.
    """

    cluster_ids: Any

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
