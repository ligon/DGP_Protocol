"""The :class:`DataGeneratingProcess` Protocol.

A DGP is a probability distribution over datasets.  The dataset is a
(possibly very high-dimensional) random matrix; the *observed
realization* is one draw.  The Protocol exposes the observed
realization (frozen) and a way to obtain additional realizations.

Dependence structure -- iid, clustered, time-series, spatial -- is
*entirely internal* to each DGP implementation.  Two members on the
Protocol; everything else (sampling design, parametric-vs-empirical
nature, conditional moments, ...) is implementation-specific.

See the design note in ``docs/design/`` for the motivating
discussion.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DataGeneratingProcess(Protocol):
    """A probability distribution over datasets.

    Implementations must provide:

    - :attr:`data`: a frozen property returning the observed
      realization (the privileged one), or ``None`` for a DGP
      specified for Monte Carlo / power-study purposes without
      committing to a specific observation.
    - :meth:`draw`: returns a single fresh realization.  By default
      the realization has the same shape as ``data``; ``size`` is a
      hint to the DGP about per-realization dimensions for callers
      who want something different.  Implementations are free to
      ignore the hint.

    The Protocol is runtime-checkable; ``isinstance(obj,
    DataGeneratingProcess)`` returns ``True`` for any object that
    exposes both members.

    See Manski (1988), *Analog Estimation Methods in Econometrics*,
    for the conceptual framework that motivates this surface -- a
    DGP is the *stand-in distribution* against which an analog
    estimator is defined.
    """

    @property
    def data(self) -> Any:
        """The frozen observed realization (or ``None`` if not bound)."""

    def draw(
        self,
        size: tuple[int, ...] | None = None,
        *,
        rng: np.random.Generator,
    ) -> Any:
        """Return a single fresh realization."""
