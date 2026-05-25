"""Composition primitives: building DGPs from simpler DGPs.

The Protocol stays unchanged under composition -- a composite DGP
is still a DGP conforming to ``data`` + ``draw``.  Consumers do not
need to know about the composition.

Currently exported:

- :class:`TwoStageDGP` -- hierarchical two-stage sampling (clusters
  drawn from an outer DGP; within-cluster observations drawn from
  inner DGPs indexed by cluster characteristics).
- :func:`with_data` -- the rebinding operation, generalised to any
  DGP with a ``with_data`` method (a common pattern in this
  package's containers).

Recursive composition (3+ stages, clusters-within-clusters) is
supported by nesting: the ``inner`` of a :class:`TwoStageDGP` can
itself return :class:`TwoStageDGP` instances.

A ``bootstrap_dgp(...)`` constructor for resampling-based derived
DGPs is intentionally **not** implemented in this initial release.
The cluster-(block-)bootstrap of raw data is available via
:class:`~dgp_protocol.sampling.ClusteredSampling` on an
:class:`~dgp_protocol.empirical.EmpiricalDGP`; wild-bootstrap-of-
moment-errors variants are estimator-specific and belong in
consumer packages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocol import DataGeneratingProcess


@dataclass(frozen=True)
class TwoStageDGP:
    """Hierarchical two-stage DGP.

    Clusters are drawn from an outer DGP; within each cluster, rows
    are drawn from an inner DGP indexed by that cluster's
    characteristics.  Stitching is done by concatenating per-cluster
    realizations and labelling rows with their drawn cluster index.

    Parameters
    ----------
    outer:
        A DGP whose ``draw`` returns *cluster-level rows*.  One row
        per cluster.  The columns carry whatever cluster
        characteristics the inner DGP needs to condition on.
    inner:
        A callable ``cluster_chars -> DataGeneratingProcess`` mapping
        a single cluster's characteristics (one row of the outer
        draw) to the DGP describing that cluster's within-cluster
        observations.  When the inner is independent of cluster
        characteristics (purely iid within-cluster), the callable
        degenerates to ``lambda c: fixed_inner_dgp``.
    observation:
        Optional observed realization to expose via :attr:`data`.
        ``None`` means "no observed data yet".  The stitching of the
        observation is the caller's responsibility -- if you have
        cluster + observation data, pre-stitch into the flat-matrix
        form and pass via ``observation``.

    Returns
    -------
    A :class:`TwoStageDGP` conforming to the
    :class:`~dgp_protocol.protocol.DataGeneratingProcess` Protocol.

    Notes
    -----
    The current implementation returns ``draw`` realizations as a
    Python list of per-cluster numpy arrays.  Downstream consumers
    typically prefer a flat-matrix + cluster-id-array representation
    -- the choice is an open design point (see ``docs/design/dgp.org``
    in this repository's history).  Callers wanting a flat layout
    should post-process with :func:`numpy.vstack` and a
    cluster-id-array constructed from per-cluster lengths.
    """

    outer: DataGeneratingProcess
    inner: Callable[[Any], DataGeneratingProcess]
    observation: Any = field(default=None)

    @property
    def data(self) -> Any:
        return self.observation

    def draw(
        self,
        size: tuple[int, ...] | None = None,
        *,
        rng: np.random.Generator,
    ) -> Any:
        """Draw via two-stage simulation: clusters then within-cluster rows.

        Returns a list of per-cluster realizations; see the class
        docstring for the choice of return type.
        """

        cluster_rows = self.outer.draw(size=size, rng=rng)
        cluster_rows_arr = np.asarray(cluster_rows)
        # ``inner`` is called once per cluster row.  Each inner DGP
        # gets its own draw with the supplied rng (consumes state
        # in order).
        per_cluster: list[Any] = []
        for cluster_chars in cluster_rows_arr:
            inner_dgp = self.inner(cluster_chars)
            per_cluster.append(inner_dgp.draw(rng=rng))
        return per_cluster

    def with_data(self, observation: Any) -> TwoStageDGP:
        """Return a new TwoStageDGP bound to a different observed realization.

        Preserves the outer/inner structure.  The caller is
        responsible for ensuring the new observation is compatible
        with the composite's structural assumptions.
        """

        return TwoStageDGP(
            outer=self.outer,
            inner=self.inner,
            observation=observation,
        )


def with_data(dgp: DataGeneratingProcess, observation: Any) -> Any:
    """Return a new DGP bound to a different observed realization.

    Delegates to the DGP's own ``with_data`` method when it has one
    (the convention in this package's container types).  For DGPs
    without a ``with_data`` method, raises :class:`TypeError`.

    Notes
    -----
    This is a thin convenience.  Most users will call ``dgp.with_data(...)``
    directly on the concrete class.  This function exists so
    framework code can rebind any DGP-like object without needing to
    know the concrete type.
    """

    if hasattr(dgp, "with_data") and callable(dgp.with_data):
        return dgp.with_data(observation)
    raise TypeError(
        f"{type(dgp).__name__} does not expose a with_data method; "
        "cannot rebind its observed realization without an explicit "
        "constructor call."
    )
