"""The :class:`EmpiricalDGP` container.

Wraps a bare observed dataset with an optional sampling design that
controls dependence structure.  Conforms to the
:class:`~dgp_protocol.protocol.DataGeneratingProcess` Protocol.

This is a *container*, not a model: the bootstrap-resampling logic
lives on the sampling-design object; the EmpiricalDGP is just a
Protocol-conformant adapter that exposes ``data`` and ``draw`` over
that machinery.
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

    Examples
    --------
    >>> import numpy as np
    >>> from dgp_protocol import EmpiricalDGP, ClusteredSampling
    >>> rng = np.random.default_rng(0)
    >>> obs = rng.standard_normal(size=(10, 3))
    >>> dgp = EmpiricalDGP(observation=obs)
    >>> dgp.draw(rng=rng).shape
    (10, 3)
    >>> # With clusters:
    >>> clusters = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    >>> cdgp = EmpiricalDGP(observation=obs, sampling=ClusteredSampling(clusters))
    >>> cdgp.draw(rng=rng).shape[1]                    # second axis preserved
    3
    """

    observation: Any
    sampling: SamplingDesign = field(default_factory=IIDSampling)

    @property
    def data(self) -> Any:
        """The frozen observed realization."""

        return self.observation

    def draw(
        self,
        size: tuple[int, ...] | None = None,
        *,
        rng: np.random.Generator,
    ) -> Any:
        """Bootstrap-resample a fresh realization."""

        return self.sampling.bootstrap_resample(self.observation, size=size, rng=rng)

    def with_data(self, observation: Any) -> EmpiricalDGP:
        """Return a new EmpiricalDGP bound to a different realization.

        Preserves the sampling-design structure; only the observed
        data is rebound.  ``data`` on the original instance is
        unchanged.
        """

        return EmpiricalDGP(observation=observation, sampling=self.sampling)
