"""The :class:`ParametricDGP` container.

Wraps a user-supplied data-generating callable.  Conforms to the
:class:`~dgp_protocol.protocol.DataGeneratingProcess` Protocol.

This is a *container*, not a model: the generation logic lives in
the user's ``generator`` callable; ParametricDGP is just a
Protocol-conformant adapter.

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

    Examples
    --------
    >>> import numpy as np
    >>> from dgp_protocol import ParametricDGP
    >>>
    >>> def standard_normal_matrix(rng, shape):
    ...     return rng.standard_normal(size=shape)
    >>>
    >>> dgp = ParametricDGP(
    ...     generator=standard_normal_matrix,
    ...     default_shape=(100, 3),
    ... )
    >>> rng = np.random.default_rng(0)
    >>> dgp.draw(rng=rng).shape
    (100, 3)
    >>> dgp.draw(size=(50, 3), rng=rng).shape
    (50, 3)
    """

    generator: Callable[[np.random.Generator, tuple[int, ...]], Any]
    default_shape: tuple[int, ...]
    observation: Any = field(default=None)

    @property
    def data(self) -> Any:
        """The observed realization, or ``None`` if not bound."""

        return self.observation

    def draw(
        self,
        size: tuple[int, ...] | None = None,
        *,
        rng: np.random.Generator,
    ) -> Any:
        """Generate a fresh realization via the supplied generator."""

        return self.generator(rng, size if size is not None else self.default_shape)

    def with_data(self, observation: Any) -> ParametricDGP:
        """Return a new ParametricDGP bound to a specific observed realization.

        Preserves the generator and default_shape; only the observed
        data is rebound.
        """

        return ParametricDGP(
            generator=self.generator,
            default_shape=self.default_shape,
            observation=observation,
        )
