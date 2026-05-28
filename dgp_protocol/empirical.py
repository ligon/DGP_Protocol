"""The :class:`EmpiricalDGP` container.

Wraps a bare observed dataset with an optional sampling design that
controls dependence structure.  Conforms to the
:class:`~dgp_protocol.protocol.DataGeneratingProcess` Protocol.

This is a *container*, not a model: the bootstrap-resampling logic
lives on the sampling-design object; the EmpiricalDGP is just a
Protocol-conformant adapter that exposes ``data`` and ``draw`` over
that machinery.

Randomness is owned by the DGP.  Pass ``seed`` at construction for
reproducibility; otherwise the Generator is seeded from system
entropy.  Use :meth:`EmpiricalDGP.with_rng` to inject a specific
Generator post-construction (e.g., a spawned child stream for a
parallel bootstrap worker).

Distributional features
-----------------------
Under :class:`~dgp_protocol.sampling.IIDSampling`, the per-observation
marginal distribution is the empirical distribution F̂ of the
observation matrix; :meth:`mean`, :meth:`var`, :meth:`cov`, and
:meth:`expect` are exact and computed in closed form.  Under
:class:`~dgp_protocol.sampling.ClusteredSampling` (and any other
non-iid design) rows are not iid; per-observation marginal moments
lose their analog-estimation interpretation and the methods raise
:class:`NotImplementedError` pointing the user at the dataset-level
surface :attr:`sample_distribution`.

Dataset-level operations (sampling distribution of statistics, the
cluster-robust moment-vector covariance for analog estimation) live
on :attr:`sample_distribution` and work for any sampling design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .exceptions import AnalyticUnavailable
from .sample_distribution import SampleDistribution
from .sampling import IIDSampling, SamplingDesign, _array_namespace, _evaluate_moments


@dataclass(frozen=True)
class EmpiricalDGP:
    """A Protocol-conformant wrapper around an observed dataset.

    Parameters
    ----------
    observation:
        The observed realization (any array-like; typically a
        :class:`numpy.ndarray` of shape ``(N, p)``).  Frozen for the
        lifetime of the DGP; use :meth:`with_data` to rebind.
    sampling:
        Sampling design controlling the bootstrap-resampling recipe.
        Default :class:`~dgp_protocol.sampling.IIDSampling` (rows are
        iid).  Use :class:`~dgp_protocol.sampling.ClusteredSampling`
        for cluster-correlated data.
    seed:
        Optional integer seed.  ``None`` (default) uses system entropy
        (draws are non-reproducible).  Pass an int for a reproducible
        Generator constructed via :func:`numpy.random.default_rng`.

    Examples
    --------
    >>> import numpy as np
    >>> from dgp_protocol import EmpiricalDGP, ClusteredSampling
    >>> obs = np.random.default_rng(0).standard_normal(size=(10, 3))
    >>> dgp = EmpiricalDGP(observation=obs, seed=1)
    >>> dgp.draw().shape
    (10, 3)
    >>> dgp.mean().shape   # exact analytic mean of each column
    (3,)
    """

    observation: Any
    sampling: SamplingDesign = field(default_factory=IIDSampling)
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # ``field(default_factory=IIDSampling)`` only fires when the
        # caller omits the kwarg; an explicit ``sampling=None`` bypasses
        # it and would crash downstream in
        # ``sampling.moment_covariance_estimator`` with an unhelpful
        # ``AttributeError`` (see issue #2 and ManifoldGMM PR #54
        # commit notes).  Coerce to the default so the type annotation
        # (``SamplingDesign``, not ``Optional[SamplingDesign]``) stays
        # honest.  ``object.__setattr__`` is required because the
        # dataclass is frozen.
        if self.sampling is None:
            object.__setattr__(self, "sampling", IIDSampling())
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @property
    def data(self) -> Any:
        """The frozen observed realization."""

        return self.observation

    def draw(self, size: tuple[int, ...] | None = None) -> Any:
        """Bootstrap-resample a fresh realization."""

        return self.sampling.bootstrap_resample(
            self.observation, size=size, rng=self._rng
        )

    # ------------------------------------------------------------------
    # P-side distributional features: analytic for IID, refuse for non-iid.
    # ------------------------------------------------------------------
    def _refuse_non_iid(self, op: str) -> None:
        if not isinstance(self.sampling, IIDSampling):
            raise NotImplementedError(
                f"EmpiricalDGP.{op}: per-observation marginal under "
                f"{type(self.sampling).__name__} loses the analog-"
                f"estimation interpretation (rows are not iid).  Use "
                f"dgp.sample_distribution for dataset-level operations "
                f"(including cluster-robust moment covariance), or "
                f"re-construct with sampling=IIDSampling() to get the "
                f"empirical marginal moment."
            )

    def expect(self, func: Any, **kwargs: Any) -> Any:
        """``E_F̂[func(X)]`` -- exact over rows of :attr:`observation`."""

        self._refuse_non_iid("expect")
        del kwargs  # No MC; convergence kwargs are not applicable.
        arr = np.asarray(self.observation)
        if arr.ndim == 0:
            return func(arr.item())
        vals = [func(row) for row in arr]
        # Aggregate via numpy for arrays / scalars; fall back to
        # numpy mean for the common case.
        first = vals[0]
        if isinstance(first, int | float | np.number) or isinstance(first, np.ndarray):
            return np.mean(np.stack([np.asarray(v) for v in vals]), axis=0)
        # Defer non-numpy aggregation to the shared aggregator.
        from ._mc import aggregate

        return aggregate(vals)[0]

    def mean(self, **kwargs: Any) -> Any:
        """Exact empirical mean over rows of :attr:`observation`."""

        self._refuse_non_iid("mean")
        del kwargs
        obs = np.asarray(self.observation)
        if obs.ndim <= 1:
            return float(obs.mean()) if obs.size else float("nan")
        return obs.mean(axis=0)

    def var(self, **kwargs: Any) -> Any:
        """Exact per-coordinate sample variance (ddof=1) of :attr:`observation`."""

        self._refuse_non_iid("var")
        del kwargs
        obs = np.asarray(self.observation, dtype=float)
        if obs.ndim <= 1:
            return float(obs.var(ddof=1)) if obs.size > 1 else float("nan")
        return obs.var(axis=0, ddof=1)

    def cov(self, **kwargs: Any) -> Any:
        """Exact sample covariance (ddof=1) of :attr:`observation`."""

        self._refuse_non_iid("cov")
        del kwargs
        obs = np.asarray(self.observation, dtype=float)
        if obs.ndim <= 1:
            return float(obs.var(ddof=1)) if obs.size > 1 else float("nan")
        return np.cov(obs, rowvar=False, ddof=1)

    # ------------------------------------------------------------------
    # D-side surface.
    # ------------------------------------------------------------------
    @property
    def sample_distribution(self) -> SampleDistribution:
        """Dataset-level distribution view (sampling distribution of statistics)."""

        return SampleDistribution(self)

    def _sd_cluster_score_blocks(
        self,
        theta: Any,
        gi: Any,
        *,
        centered: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Per-i.i.d.-unit centered moment-score blocks ``(G, k)``.

        Delegates to ``self.sampling.cluster_score_blocks`` when the
        sampling design exposes it (the built-in :class:`IIDSampling`
        and :class:`ClusteredSampling` both do).  Falls back to the
        legacy ``moment_covariance_estimator`` -> raise path for
        user-defined sampling designs that pre-date the blocks surface.
        """

        del kwargs  # MC-control kwargs are not applicable to the analytic path.
        blocks_method = getattr(self.sampling, "cluster_score_blocks", None)
        if callable(blocks_method):
            return blocks_method(self.observation, theta, gi, centered=centered)
        raise AnalyticUnavailable(
            f"EmpiricalDGP._sd_cluster_score_blocks: sampling design "
            f"{type(self.sampling).__name__} does not expose "
            f"cluster_score_blocks; falling back via _sd_moment_covariance."
        )

    def _sd_moment_covariance(
        self,
        theta: Any,
        gi: Any,
        **kwargs: Any,
    ) -> Any:
        """Analytic moment-vector covariance under the bound observation.

        Delegates to ``self.sampling.moment_covariance_estimator``, the
        closed-form formula corresponding to the sampling design (iid
        outer product for :class:`~dgp_protocol.IIDSampling`, cluster-
        robust sandwich for :class:`~dgp_protocol.ClusteredSampling`).
        No Monte Carlo.

        Honored kwargs: ``centered`` (default ``True``).  MC-control
        kwargs (``atol``, ``rtol``, ``max_its``, ``batch_size``) are
        silently ignored -- this is the analytic path, no convergence
        loop applies.

        Retained for compatibility with user-defined sampling designs
        that pre-date the :meth:`_sd_cluster_score_blocks` surface;
        :class:`~dgp_protocol.SampleDistribution.moment_covariance`
        prefers the blocks hook when both are present.
        """

        centered = kwargs.get("centered", True)
        return self.sampling.moment_covariance_estimator(
            self.observation, theta, gi, centered=centered
        )

    def _sd_within_cluster_block(
        self,
        theta: Any,
        gi: Any,
    ) -> Any:
        """Raw moment sum ``Σ_i g_i(θ, X_i)`` over the bound observation.

        The within-cluster primitive consumed by :class:`TwoStageDGP`
        when composing per-outer-cluster blocks: the composite asks
        each inner DGP for its uncentered, unscaled moment sum and
        does its own centering / scaling against the global ``N``.
        Distinct from :meth:`_sd_cluster_score_blocks` (which is the
        full ``(G, k)`` prepared-blocks surface consumed by
        :class:`SampleDistribution`).

        Raises :class:`AnalyticUnavailable` when ``self.observation``
        is ``None``.
        """

        if self.observation is None:
            raise AnalyticUnavailable(
                "EmpiricalDGP._sd_within_cluster_block: observation is "
                "None; bind one via .with_data(obs) before composing."
            )
        moments = _evaluate_moments(gi, theta, self.observation)
        xp = _array_namespace(moments)
        return xp.nansum(moments, axis=0)

    # ------------------------------------------------------------------
    # Lineage operations.
    # ------------------------------------------------------------------
    def with_data(self, observation: Any) -> EmpiricalDGP:
        """Return a new EmpiricalDGP bound to a different realization.

        Preserves the sampling-design structure; the child receives
        an *independent* Generator spawned from the parent's stream
        via :meth:`numpy.random.Generator.spawn`, so the lineage is
        deterministic (when the parent is seeded) but child draws do
        not consume the parent's randomness.  ``data`` on the
        original instance is unchanged.
        """

        return self._rebuild(observation=observation, rng=self._rng.spawn(1)[0])

    def with_rng(self, rng: np.random.Generator) -> EmpiricalDGP:
        """Return a new EmpiricalDGP that uses ``rng`` as its Generator.

        Useful for parallel-worker fan-out::

            children = [parent.with_rng(s) for s in parent._rng.spawn(N)]

        The new DGP shares all structural attributes with the parent;
        only the Generator differs.
        """

        return self._rebuild(observation=self.observation, rng=rng)

    def _rebuild(self, *, observation: Any, rng: np.random.Generator) -> EmpiricalDGP:
        """Construct a sibling with a specific ``rng`` installed."""

        new = EmpiricalDGP(observation=observation, sampling=self.sampling)
        object.__setattr__(new, "_rng", rng)
        return new
