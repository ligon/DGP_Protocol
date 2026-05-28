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

Observed data
-------------
``TwoStageDGP`` *owns the stitch*.  Callers describe their observed
data as the pair ``(outer_observation, inner_observations)``:

- ``outer_observation`` -- ``(G, p_outer)`` matrix; one row per
  observed cluster, carrying the cluster characteristics the
  ``inner`` callable conditions on.  May be ``None`` when the
  inner DGP doesn't depend on per-cluster characteristics.
- ``inner_observations`` -- length-``G`` list of ``(n_c, p_inner)``
  arrays; element ``c`` is the within-cluster observed block for
  cluster ``c``.

The composite concatenates ``inner_observations`` into a flat
``(N, p_inner)`` matrix and remembers per-cluster sizes; the flat
matrix is what :attr:`data` returns (for consumers wanting a flat
layout) and the per-cluster sizes are what the analytic moment-
covariance computation in :meth:`_sd_cluster_score_blocks` uses to
form the cluster-robust sandwich.

Distributional features
-----------------------
``draw()`` on :class:`TwoStageDGP` returns a *list* of per-cluster
arrays -- a heterogeneous shape with no unambiguous "row" notion.
Per-observation marginal operations (:meth:`expect`, :meth:`mean`,
:meth:`var`, :meth:`cov`) therefore raise
:class:`NotImplementedError` pointing the user at the dataset-level
surface :attr:`sample_distribution`, which can accept any
user-supplied statistic of the whole realization.

Analytic moment-covariance
^^^^^^^^^^^^^^^^^^^^^^^^^^
When ``inner_observations`` is supplied, :class:`TwoStageDGP`
implements :meth:`_sd_cluster_score_blocks` so
``sample_distribution.moment_covariance(theta, gi)`` returns the
cluster-robust ``hat Omega`` analytically -- no Monte Carlo over
``draw()`` realizations.  The per-outer-cluster contribution is
obtained by asking each inner DGP for its
:meth:`_sd_within_cluster_block` (the empirical-inner case
delegates to a row-sum over the bound observed block; the
parametric-inner case with analytic ``expect`` delegates to
``n_c * mu_c(theta)``).  See the source for the centering and
``1/sqrt(N)`` conventions, which match
:class:`~dgp_protocol.sampling.ClusteredSampling`'s sandwich.

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

from .exceptions import AnalyticUnavailable
from .protocol import DataGeneratingProcess
from .sample_distribution import SampleDistribution
from .sampling import _array_namespace, _evaluate_moments


@dataclass(frozen=True)
class TwoStageDGP:
    """Hierarchical two-stage DGP.

    Clusters are drawn from an outer DGP; within each cluster, rows
    are drawn from an inner DGP indexed by that cluster's
    characteristics.  When ``inner_observations`` is supplied the
    composite stitches them into a flat ``(N, p)`` matrix and
    remembers per-cluster sizes; the flat matrix is exposed via
    :attr:`data` and the per-cluster mapping drives the analytic
    cluster-robust :meth:`_sd_cluster_score_blocks`.

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
    outer_observation:
        Optional ``(G, p_outer)`` matrix of observed cluster
        characteristics.  ``None`` (default) means "no outer data
        observed"; the analytic moment-covariance can still run when
        ``inner_observations`` is supplied (it then falls back to
        direct row-sums on the observed inner blocks rather than
        consulting the inner-DGP analytic surface).
    inner_observations:
        Optional length-``G`` list of within-cluster observed blocks
        ``[(n_c, p_inner) array, ...]``.  ``None`` (default) means
        "no observed data yet".  When supplied alongside
        ``outer_observation``, the lengths must agree:
        ``outer_observation.shape[0] == len(inner_observations)``.
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
    outer_observation: Any = field(default=None)
    inner_observations: list[Any] | None = field(default=None)
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)
    _stitched: Any = field(init=False, repr=False, compare=False)
    _cluster_sizes: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))
        # Validate consistency and pre-compute stitched data.
        if self.inner_observations is not None:
            inner_obs = list(self.inner_observations)
            cluster_sizes = tuple(int(np.asarray(o).shape[0]) for o in inner_obs)
            if self.outer_observation is not None:
                outer_arr = np.asarray(self.outer_observation)
                if outer_arr.shape[0] != len(cluster_sizes):
                    raise ValueError(
                        f"TwoStageDGP: outer_observation has "
                        f"{outer_arr.shape[0]} rows but "
                        f"inner_observations has {len(cluster_sizes)} "
                        f"entries; they must agree (one inner block "
                        f"per outer cluster row)."
                    )
            stitched = np.vstack([np.asarray(o) for o in inner_obs]) if inner_obs else None
            object.__setattr__(self, "_stitched", stitched)
            object.__setattr__(self, "_cluster_sizes", cluster_sizes)
        else:
            object.__setattr__(self, "_stitched", None)
            object.__setattr__(self, "_cluster_sizes", ())

    @property
    def data(self) -> Any:
        """The flat stitched ``(N, p)`` observation matrix, or ``None``."""

        return self._stitched

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
    # Analytic cluster-robust moment-covariance.
    # ------------------------------------------------------------------
    def _sd_cluster_score_blocks(
        self,
        theta: Any,
        gi: Any,
        *,
        centered: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Cluster-robust per-outer-cluster moment-score blocks ``(G, k)``.

        Available when ``inner_observations`` is supplied; raises
        :class:`AnalyticUnavailable` otherwise.

        For each observed outer cluster ``c``, asks the inner DGP for
        its uncentered, unscaled within-cluster moment sum ``S_c``
        (via :meth:`_sd_within_cluster_block` on the inner DGP, or by
        a direct row-sum on the bound block when the inner doesn't
        expose the hook).  Then centers blocks against the global
        mean ``g_bar = (Σ_c S_c) / N`` -- subtracting ``n_c · g_bar``
        from each block -- and scales by ``1/sqrt(N)`` so
        ``blocks.T @ blocks`` yields the cluster-robust ``hat Omega``.

        Notes
        -----
        Per-row centering vs per-block centering: this implementation
        centers each *block* by ``n_c · g_bar`` (i.e. the
        within-block sum minus the cluster-size-weighted global mean),
        which matches the v1 ManifoldGMM cluster-sandwich centering
        and the
        :class:`~dgp_protocol.sampling.ClusteredSampling`
        ``moment_covariance_estimator``.  Equivalent to subtracting
        ``g_bar`` from every row before summing the within-block
        rows.
        """

        del kwargs  # MC-control kwargs are not applicable on the analytic path.

        if self.inner_observations is None:
            raise AnalyticUnavailable(
                "TwoStageDGP._sd_cluster_score_blocks: inner_observations "
                "is None.  Construct with inner_observations=[...] (or "
                "rebind via .with_data([...])) to enable the analytic "
                "cluster-robust path; otherwise SampleDistribution falls "
                "back to Monte Carlo over draw()."
            )

        inner_obs = list(self.inner_observations)
        G = len(inner_obs)
        if G == 0:
            raise AnalyticUnavailable(
                "TwoStageDGP._sd_cluster_score_blocks: zero clusters."
            )

        # Build per-cluster raw moment sums S_c.  Prefer the inner DGP's
        # _sd_within_cluster_block hook; fall back to a direct row-sum on
        # the observed block if the hook isn't available or raises
        # AnalyticUnavailable.
        raw_sums: list[Any] = []
        n_per_cluster: list[int] = []
        xp_ref = None
        for c in range(G):
            block = np.asarray(inner_obs[c])
            n_c = int(block.shape[0])
            n_per_cluster.append(n_c)

            S_c = None
            if self.outer_observation is not None:
                chars = np.asarray(self.outer_observation)[c]
                inner_dgp_unbound = self.inner(chars, self._rng.spawn(1)[0])
                bound = (
                    inner_dgp_unbound.with_data(inner_obs[c])
                    if hasattr(inner_dgp_unbound, "with_data")
                    else inner_dgp_unbound
                )
                hook = getattr(bound, "_sd_within_cluster_block", None)
                if callable(hook):
                    try:
                        S_c = hook(theta, gi)
                    except AnalyticUnavailable:
                        S_c = None

            if S_c is None:
                # Direct row-sum fallback: equivalent to what the
                # EmpiricalDGP inner hook would do, and what we want
                # when no outer-observation / inner-hook is available.
                moments = _evaluate_moments(gi, theta, inner_obs[c])
                xp_local = _array_namespace(moments)
                S_c = xp_local.nansum(moments, axis=0)

            S_c_arr = np.asarray(S_c) if _array_namespace(S_c) is np else S_c
            # Coerce 0-d (scalar) to (1,); preserve higher-dim shapes.
            local_xp = _array_namespace(S_c_arr)
            if S_c_arr.ndim == 0:
                S_c_arr = S_c_arr.reshape(1)
            if xp_ref is None:
                xp_ref = local_xp
            raw_sums.append(S_c_arr)

        xp = xp_ref if xp_ref is not None else np
        blocks_raw = xp.stack([xp.asarray(s) for s in raw_sums])  # (G, k)

        n_arr = xp.asarray(n_per_cluster, dtype=blocks_raw.dtype).reshape(-1, 1)
        N = float(np.sum(n_per_cluster))
        if N <= 0:
            raise AnalyticUnavailable(
                "TwoStageDGP._sd_cluster_score_blocks: total N is zero."
            )

        if centered:
            total = blocks_raw.sum(axis=0)
            g_bar = total / N
            blocks_centered = blocks_raw - n_arr * g_bar.reshape(1, -1)
        else:
            blocks_centered = blocks_raw

        scaled = blocks_centered / xp.sqrt(xp.asarray(N))
        return scaled

    def _sd_within_cluster_block(
        self,
        theta: Any,
        gi: Any,
    ) -> Any:
        """Raw moment sum over the stitched observation (``Σ_c S_c``).

        Provided so a :class:`TwoStageDGP` itself can serve as the
        inner of a higher-order composition.  Equivalent to summing
        the per-cluster raw sums that :meth:`_sd_cluster_score_blocks`
        builds internally.
        """

        if self.inner_observations is None:
            raise AnalyticUnavailable(
                "TwoStageDGP._sd_within_cluster_block: inner_observations "
                "is None; bind one via .with_data([...]) before composing."
            )
        # Concatenate inner blocks and sum across rows; identical
        # answer to summing per-cluster S_c's by associativity.
        if self._stitched is None:
            raise AnalyticUnavailable(
                "TwoStageDGP._sd_within_cluster_block: stitched "
                "observation is empty."
            )
        moments = _evaluate_moments(gi, theta, self._stitched)
        xp = _array_namespace(moments)
        return xp.nansum(moments, axis=0)

    # ------------------------------------------------------------------
    # Lineage operations.
    # ------------------------------------------------------------------
    def with_data(
        self,
        inner_observations: list[Any] | None = None,
        *,
        outer_observation: Any | None = None,
        keep_outer_observation: bool = True,
    ) -> TwoStageDGP:
        """Return a new TwoStageDGP bound to a different observed realization.

        Positional ``inner_observations`` rebinds the per-cluster
        observed blocks; keyword ``outer_observation`` rebinds the
        observed cluster characteristics.  When ``outer_observation``
        is omitted, the existing one is preserved iff
        ``keep_outer_observation=True`` (the default); set
        ``keep_outer_observation=False`` to clear it explicitly.

        Preserves the outer/inner structure; the child receives an
        *independent* Generator spawned from the parent's stream.
        The caller is responsible for ensuring the new observation
        is compatible with the composite's structural assumptions
        (length match against ``outer_observation`` is enforced in
        ``__post_init__``).
        """

        new_outer_obs = (
            outer_observation
            if outer_observation is not None
            else (self.outer_observation if keep_outer_observation else None)
        )
        return self._rebuild(
            outer_observation=new_outer_obs,
            inner_observations=inner_observations,
            rng=self._rng.spawn(1)[0],
        )

    def with_rng(self, rng: np.random.Generator) -> TwoStageDGP:
        """Return a new TwoStageDGP that uses ``rng`` as its Generator.

        Only the composite's own spawn-source changes; the outer DGP's
        Generator is unaffected.  See :meth:`with_data` for the
        bootstrap-fan-out idiom.
        """

        return self._rebuild(
            outer_observation=self.outer_observation,
            inner_observations=self.inner_observations,
            rng=rng,
        )

    def _rebuild(
        self,
        *,
        outer_observation: Any,
        inner_observations: list[Any] | None,
        rng: np.random.Generator,
    ) -> TwoStageDGP:
        """Construct a sibling with a specific ``rng`` installed."""

        new = TwoStageDGP(
            outer=self.outer,
            inner=self.inner,
            outer_observation=outer_observation,
            inner_observations=inner_observations,
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
                self.outer_observation,
                self.inner_observations,
                self.seed,
                self._rng,
            ),
        )


def _reconstruct_two_stage_dgp(
    outer: DataGeneratingProcess,
    inner_bytes: bytes,
    outer_observation: Any,
    inner_observations: list[Any] | None,
    seed: int | None,
    rng: np.random.Generator,
) -> TwoStageDGP:
    """Module-level reconstructor for :meth:`TwoStageDGP.__reduce__`."""

    new = TwoStageDGP(
        outer=outer,
        inner=cloudpickle.loads(inner_bytes),
        outer_observation=outer_observation,
        inner_observations=inner_observations,
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
