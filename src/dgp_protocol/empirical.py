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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sampling import IIDSampling, SamplingDesign


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
    >>> # With clusters:
    >>> clusters = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    >>> cdgp = EmpiricalDGP(
    ...     observation=obs, sampling=ClusteredSampling(clusters), seed=1
    ... )
    >>> cdgp.draw().shape[1]                    # second axis preserved
    3
    """

    observation: Any
    sampling: SamplingDesign = field(default_factory=IIDSampling)
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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
