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

Randomness:  A :class:`TwoStageDGP` owns its own RNG (set via the
``seed`` constructor kwarg, default system entropy).  On each draw
the composite spawns one independent child Generator per cluster
and passes it to the ``inner`` callable.  The composite's own seed
therefore drives *all* per-cluster within-cluster randomness
deterministically; the outer DGP's own RNG drives the cluster-
characteristic draw.  Seeding both the outer DGP and the composite
makes a top-level :meth:`TwoStageDGP.draw` fully reproducible.

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

from .distribution import DistributionalFeatures
from .protocol import DataGeneratingProcess


@dataclass(frozen=True)
class TwoStageDGP(DistributionalFeatures):
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
        A callable ``(cluster_chars, rng) -> DataGeneratingProcess``
        mapping a single cluster's characteristics (one row of the
        outer draw) plus a per-cluster Generator to the DGP describing
        that cluster's within-cluster observations.  The ``rng`` is
        spawned by the composite for each cluster; the callable
        typically installs it via ``inner_dgp.with_rng(rng)`` (or
        constructs the inner DGP with whatever seed convention the
        user prefers).  When the inner is independent of cluster
        characteristics, the callable degenerates to
        ``lambda chars, rng: fixed_inner_dgp.with_rng(rng)``.
    observation:
        Optional observed realization to expose via :attr:`data`.
        ``None`` means "no observed data yet".  The stitching of the
        observation is the caller's responsibility -- if you have
        cluster + observation data, pre-stitch into the flat-matrix
        form and pass via ``observation``.
    seed:
        Optional integer seed for the composite's own Generator,
        used to spawn the per-cluster child rngs that get passed to
        ``inner``.  Does *not* control the outer DGP's randomness
        (that belongs to the outer DGP's own seed).

    Returns
    -------
    A :class:`TwoStageDGP` conforming to the
    :class:`~dgp_protocol.protocol.DataGeneratingProcess` Protocol.

    Notes
    -----
    The current implementation returns ``draw`` realizations as a
    Python list of per-cluster numpy arrays.  Downstream consumers
    typically prefer a flat-matrix + cluster-id-array representation
    -- the choice is an open design point.  Callers wanting a flat
    layout should post-process with :func:`numpy.vstack` and a
    cluster-id-array constructed from per-cluster lengths.
    """

    outer: DataGeneratingProcess
    inner: Callable[[Any, np.random.Generator], DataGeneratingProcess]
    observation: Any = field(default=None)
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @property
    def data(self) -> Any:
        return self.observation

    def draw(self, size: tuple[int, ...] | None = None) -> Any:
        """Draw via two-stage simulation: clusters then within-cluster rows.

        Returns a list of per-cluster realizations; see the class
        docstring for the choice of return type.  One independent
        Generator is spawned per cluster from the composite's own
        Generator and passed to ``inner``.
        """

        cluster_rows = self.outer.draw(size=size)
        cluster_rows_arr = np.asarray(cluster_rows)
        per_cluster: list[Any] = []
        for cluster_chars in cluster_rows_arr:
            child_rng = self._rng.spawn(1)[0]
            inner_dgp = self.inner(cluster_chars, child_rng)
            per_cluster.append(inner_dgp.draw())
        return per_cluster

    # ---------------------------------------------------------------
    # Distributional features.  ``expect`` works (the mixin's MC
    # default), provided the user supplies a ``func`` that reduces the
    # heterogeneous per-cluster list returned by ``draw`` to a
    # consistent shape.  ``mean``/``var``/``cov`` do not have an
    # unambiguous shape for a list-of-arrays return, so we raise
    # NotImplementedError with a hint pointing back to ``expect``.
    # ---------------------------------------------------------------
    def mean(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError(
            "TwoStageDGP.mean: draws are lists of per-cluster arrays "
            "(heterogeneous shape).  Use .expect(func) with an "
            "explicit aggregator that flattens to a consistent shape, "
            "e.g.  ts.expect(lambda lst: np.vstack(lst).mean(axis=0))."
        )

    def var(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError(
            "TwoStageDGP.var: draws are lists of per-cluster arrays.  "
            "Use .expect(func) with an explicit aggregator."
        )

    def cov(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError(
            "TwoStageDGP.cov: draws are lists of per-cluster arrays.  "
            "Use .expect(func) with an explicit aggregator."
        )

    def with_data(self, observation: Any) -> TwoStageDGP:
        """Return a new TwoStageDGP bound to a different observed realization.

        Preserves the outer/inner structure; the child receives an
        *independent* Generator spawned from the parent's stream.
        The caller is responsible for ensuring the new observation
        is compatible with the composite's structural assumptions.
        """

        return self._rebuild(observation=observation, rng=self._rng.spawn(1)[0])

    def with_rng(self, rng: np.random.Generator) -> TwoStageDGP:
        """Return a new TwoStageDGP that uses ``rng`` as its Generator.

        Only the composite's own spawn-source changes; the outer DGP's
        Generator is unaffected.  See :meth:`with_data` for the
        bootstrap-fan-out idiom.
        """

        return self._rebuild(observation=self.observation, rng=rng)

    def _rebuild(self, *, observation: Any, rng: np.random.Generator) -> TwoStageDGP:
        """Construct a sibling with a specific ``rng`` installed."""

        new = TwoStageDGP(
            outer=self.outer,
            inner=self.inner,
            observation=observation,
        )
        object.__setattr__(new, "_rng", rng)
        return new


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
