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

Distributional features
-----------------------
``draw()`` on :class:`TwoStageDGP` returns a *list* of per-cluster
arrays -- a heterogeneous shape with no unambiguous "row" notion.
Per-observation marginal operations (:meth:`expect`, :meth:`mean`,
:meth:`var`, :meth:`cov`) therefore raise
:class:`NotImplementedError` pointing the user at the dataset-level
surface :attr:`sample_distribution`, which can accept any
user-supplied statistic of the whole realization.

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

import cloudpickle
import numpy as np

from .protocol import DataGeneratingProcess
from .sample_distribution import SampleDistribution


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

    # ------------------------------------------------------------------
    # P-side: refuse, point at sample_distribution.
    # ------------------------------------------------------------------
    def _refuse(self, op: str) -> None:
        raise NotImplementedError(
            f"TwoStageDGP.{op}: per-observation marginal operations have "
            f"no unambiguous shape on a list-of-per-cluster-arrays draw.  "
            f"Use dgp.sample_distribution.{op}(stat_func) (or "
            f"sample_distribution.expect with a flattening aggregator, "
            f"e.g.  lambda lst: np.vstack(lst).mean(axis=0)) for "
            f"dataset-level operations."
        )

    def expect(self, func: Any, **kwargs: Any) -> Any:
        del func, kwargs
        self._refuse("expect")

    def mean(self, **kwargs: Any) -> Any:
        del kwargs
        self._refuse("mean")

    def var(self, **kwargs: Any) -> Any:
        del kwargs
        self._refuse("var")

    def cov(self, **kwargs: Any) -> Any:
        del kwargs
        self._refuse("cov")

    # ------------------------------------------------------------------
    # D-side surface (works on any stat_func of the whole realization).
    # ------------------------------------------------------------------
    @property
    def sample_distribution(self) -> SampleDistribution:
        """Dataset-level distribution view (sampling distribution of statistics)."""

        return SampleDistribution(self)

    # ------------------------------------------------------------------
    # Lineage operations.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Pickle support.
    # ------------------------------------------------------------------
    def __reduce__(
        self,
    ) -> tuple[Callable[..., TwoStageDGP], tuple[Any, ...]]:
        """Pickle via cloudpickle for the ``inner`` callable.

        Stdlib pickle resolves callables by ``(module, qualname)``
        lookup, which fails for the lambda / closure-based ``inner``
        builders that are the natural idiom for TwoStageDGP.  This
        ``__reduce__`` pre-serialises ``inner`` to bytes via
        :mod:`cloudpickle`; ``outer`` is passed through to stdlib
        pickle (recursively using its own ``__reduce__`` if it has
        one) so a ParametricDGP outer with lambda generator round-
        trips correctly.
        """

        return (
            _reconstruct_two_stage_dgp,
            (
                self.outer,
                cloudpickle.dumps(self.inner),
                self.observation,
                self.seed,
                self._rng,
            ),
        )


def _reconstruct_two_stage_dgp(
    outer: DataGeneratingProcess,
    inner_bytes: bytes,
    observation: Any,
    seed: int | None,
    rng: np.random.Generator,
) -> TwoStageDGP:
    """Module-level reconstructor for :meth:`TwoStageDGP.__reduce__`."""

    new = TwoStageDGP(
        outer=outer,
        inner=cloudpickle.loads(inner_bytes),
        observation=observation,
        seed=seed,
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
