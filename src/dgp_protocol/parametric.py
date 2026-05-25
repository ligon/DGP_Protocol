"""The :class:`ParametricDGP` container.

Wraps a user-supplied data-generating callable.  Conforms to the
:class:`~dgp_protocol.protocol.DataGeneratingProcess` Protocol.

This is a *container*, not a model: the generation logic lives in
the user's ``generator`` callable; ParametricDGP is just a
Protocol-conformant adapter.

Randomness is owned by the DGP.  Pass ``seed`` at construction for
reproducibility; otherwise the Generator is seeded from system
entropy.  Use :meth:`ParametricDGP.with_rng` to inject a specific
Generator post-construction (e.g., a spawned child stream for a
parallel worker).

Typical use cases:

- Monte Carlo / power studies (generate synthetic data with known
  ground truth, refit, validate inference).
- Specification checks ("if my model were exactly true at this
  ``theta``, what would I see?").
- Parametric bootstrap variants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParametricDGP:
    """A Protocol-conformant wrapper around a data-generating callable.

    Parameters
    ----------
    generator:
        Callable ``(rng, shape) -> realization``.  Should return a
        tabular object whose leading dimensions match ``shape``.
    default_shape:
        Per-realization shape used when :meth:`draw` is called without
        an explicit ``size``.
    observation:
        Optional observed realization to expose via :attr:`data`.
        ``None`` (default) means "no observed data yet" -- useful for
        Monte Carlo specifications that haven't been bound to a
        specific draw.
    seed:
        Optional integer seed.  ``None`` (default) uses system entropy
        (draws are non-reproducible across processes).  Pass an int
        for a reproducible Generator constructed via
        :func:`numpy.random.default_rng`.

    Examples
    --------
    >>> from dgp_protocol import ParametricDGP
    >>>
    >>> def standard_normal_matrix(rng, shape):
    ...     return rng.standard_normal(size=shape)
    >>>
    >>> dgp = ParametricDGP(
    ...     generator=standard_normal_matrix,
    ...     default_shape=(100, 3),
    ...     seed=0,
    ... )
    >>> dgp.draw().shape
    (100, 3)
    >>> dgp.draw(size=(50, 3)).shape
    (50, 3)
    """

    generator: Callable[[np.random.Generator, tuple[int, ...]], Any]
    default_shape: tuple[int, ...]
    observation: Any = field(default=None)
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass the __setattr__ guard to install
        # the internal Generator built from ``seed``.
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @property
    def data(self) -> Any:
        """The observed realization, or ``None`` if not bound."""

        return self.observation

    def draw(self, size: tuple[int, ...] | None = None) -> Any:
        """Generate a fresh realization via the supplied generator."""

        return self.generator(
            self._rng, size if size is not None else self.default_shape
        )

    def with_data(self, observation: Any) -> ParametricDGP:
        """Return a new ParametricDGP bound to a different realization.

        The child shares the generator and default_shape; it receives
        an *independent* Generator spawned from the parent's stream
        via :meth:`numpy.random.Generator.spawn`, so the lineage is
        deterministic (when the parent is seeded) but child draws do
        not consume the parent's randomness.
        """

        return self._rebuild(observation=observation, rng=self._rng.spawn(1)[0])

    def with_rng(self, rng: np.random.Generator) -> ParametricDGP:
        """Return a new ParametricDGP that uses ``rng`` as its Generator.

        Useful for parallel-worker fan-out::

            children = [parent.with_rng(s) for s in parent._rng.spawn(N)]

        The new DGP shares all structural attributes with the parent;
        only the Generator differs.
        """

        return self._rebuild(observation=self.observation, rng=rng)

    def _rebuild(self, *, observation: Any, rng: np.random.Generator) -> ParametricDGP:
        """Construct a sibling with a specific ``rng`` installed."""

        new = ParametricDGP(
            generator=self.generator,
            default_shape=self.default_shape,
            observation=observation,
        )
        object.__setattr__(new, "_rng", rng)
        return new
