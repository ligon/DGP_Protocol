"""The :class:`DataGeneratingProcess` Protocol.

A DGP is a probability distribution over datasets.  The dataset is a
(possibly very high-dimensional) random matrix; the *observed
realization* is one draw.  The Protocol exposes the observed
realization (frozen) and a way to obtain additional realizations.

The DGP **owns its own pseudorandom number generator**.  Callers do
not manage RNG state -- they instantiate a DGP (optionally with a
``seed`` for reproducibility), call :meth:`draw`, and that is it.
:meth:`draw` does *not* accept an ``rng`` argument.  Concrete
container types in this package expose a ``with_rng(rng)`` method
for post-construction Generator injection (used for parallel-worker
fan-out via :meth:`numpy.random.Generator.spawn`).

Dependence structure -- iid, clustered, time-series, spatial -- is
*entirely internal* to each DGP implementation.  Two members on the
Protocol; everything else (sampling design, parametric-vs-empirical
nature, conditional moments, RNG ownership, ...) is implementation-
specific.

See the design note in :file:`docs/design/dgp.org` in the
ManifoldGMM repo for the motivating discussion.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


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

    The DGP owns its own RNG; :meth:`draw` does not accept an ``rng``
    argument.  Concrete container types accept a ``seed`` constructor
    kwarg for reproducibility and expose a ``with_rng(rng)`` method
    for post-construction Generator injection.

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

    def draw(self, size: tuple[int, ...] | None = None) -> Any:
        """Return a single fresh realization."""
