"""The :class:`ParametricDGP` container.

Wraps either a user-supplied data-generating callable or a
scipy.stats-style frozen distribution (or any duck-typed equivalent).
Conforms to the :class:`~dgp_protocol.protocol.DataGeneratingProcess`
Protocol.

This is a *container*, not a model: the generation logic lives in
the user's ``generator`` callable or the supplied ``distribution``;
ParametricDGP is just a Protocol-conformant adapter.

Randomness is owned by the DGP.  Pass ``seed`` at construction for
reproducibility; otherwise the Generator is seeded from system
entropy.  Use :meth:`ParametricDGP.with_rng` to inject a specific
Generator post-construction (e.g., a spawned child stream for a
parallel worker).

Analytic features
-----------------
When ``distribution`` is supplied, the DGP's
:class:`~dgp_protocol.distribution.DistributionalFeatures` methods
(``expect``, ``mean``, ``var``, ``cov``) preferentially delegate to
the distribution's analytic methods (e.g., ``scipy.stats.norm().mean()``
returns ``0.0`` directly rather than computing it via Monte Carlo).
The dispatch is duck-typed via attribute lookup; any object exposing
the relevant methods qualifies.

Typical use cases:

- Monte Carlo / power studies (generate synthetic data with known
  ground truth, refit, validate inference).
- Specification checks ("if my model were exactly true at this
  ``theta``, what would I see?").
- Parametric bootstrap variants.
- Wrapping a ``scipy.stats`` distribution to get the
  :class:`DataGeneratingProcess` interface with analytic moments
  available.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .distribution import DistributionalFeatures


@dataclass(frozen=True)
class ParametricDGP(DistributionalFeatures):
    """A Protocol-conformant wrapper around a parametric data-generating recipe.

    Exactly one of ``generator`` or ``distribution`` must be supplied.

    Parameters
    ----------
    generator:
        Callable ``(rng, shape) -> realization``.  Should return a
        tabular object whose leading dimensions match ``shape``.  Use
        this when the generation logic is custom (not directly
        representable by a scipy.stats-style distribution).
    distribution:
        A scipy.stats-style frozen distribution (or any duck-typed
        equivalent) exposing ``rvs(size, random_state)`` plus, where
        applicable, ``mean``, ``var``, ``cov``, ``expect``.  When
        supplied, :meth:`draw` uses ``distribution.rvs`` and the
        :class:`~dgp_protocol.distribution.DistributionalFeatures`
        methods preferentially delegate to the distribution's analytic
        methods.
    default_shape:
        Per-realization shape used when :meth:`draw` is called without
        an explicit ``size``.  For scipy.stats univariate
        distributions this becomes ``rvs(size=default_shape)``; the
        returned shape follows scipy's conventions.
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
    Custom generator:

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

    scipy.stats distribution (analytic features available):

    >>> import scipy.stats                                 # doctest: +SKIP
    >>> from dgp_protocol import ParametricDGP             # doctest: +SKIP
    >>> dgp = ParametricDGP(                               # doctest: +SKIP
    ...     distribution=scipy.stats.norm(loc=0, scale=2),
    ...     default_shape=(100,),
    ...     seed=0,
    ... )
    >>> dgp.mean()                                         # doctest: +SKIP
    0.0
    >>> dgp.var()                                          # doctest: +SKIP
    4.0
    """

    generator: Callable[[np.random.Generator, tuple[int, ...]], Any] | None = None
    default_shape: tuple[int, ...] = ()
    distribution: Any = field(default=None)
    observation: Any = field(default=None)
    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.generator is None and self.distribution is None:
            raise ValueError(
                "ParametricDGP requires exactly one of `generator` or "
                "`distribution`; neither was supplied."
            )
        if self.generator is not None and self.distribution is not None:
            raise ValueError(
                "ParametricDGP requires exactly one of `generator` or "
                "`distribution`; both were supplied.  If you want a "
                "custom draw mechanism but still want analytic features "
                "from a scipy.stats-style backend, subclass ParametricDGP."
            )
        # Frozen dataclass: bypass the __setattr__ guard to install
        # the internal Generator built from ``seed``.
        object.__setattr__(self, "_rng", np.random.default_rng(self.seed))

    @property
    def data(self) -> Any:
        """The observed realization, or ``None`` if not bound."""

        return self.observation

    def draw(self, size: tuple[int, ...] | None = None) -> Any:
        """Generate a fresh realization.

        Uses ``distribution.rvs`` when ``distribution`` is supplied
        (passing the DGP's owned :class:`numpy.random.Generator` as
        ``random_state``); otherwise calls the user's ``generator``.
        """

        shape = size if size is not None else self.default_shape
        if self.distribution is not None:
            return self.distribution.rvs(size=shape, random_state=self._rng)
        # Validated in __post_init__: generator is non-None here.
        assert self.generator is not None
        return self.generator(self._rng, shape)

    def with_data(self, observation: Any) -> ParametricDGP:
        """Return a new ParametricDGP bound to a different realization.

        The child shares the generator / distribution and
        default_shape; it receives an *independent* Generator spawned
        from the parent's stream via
        :meth:`numpy.random.Generator.spawn`, so the lineage is
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
            distribution=self.distribution,
            observation=observation,
        )
        object.__setattr__(new, "_rng", rng)
        return new
