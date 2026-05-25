"""DGP_Protocol: a minimal Protocol for data-generating processes.

See :class:`DataGeneratingProcess` for the Protocol itself.  Concrete
container wrappers (:class:`EmpiricalDGP`, :class:`ParametricDGP`) and
composition primitives (:class:`TwoStageDGP`, :func:`with_data`) are
re-exported here for convenience; users typically import them as
``from dgp_protocol import ...``.

See ``docs/design/`` (in this repository's source) and Manski (1988),
*Analog Estimation Methods in Econometrics*, for the conceptual
framework -- a DGP is the stand-in distribution against which
analog estimators are defined.
"""

from __future__ import annotations

from .composition import TwoStageDGP, with_data
from .empirical import EmpiricalDGP
from .parametric import ParametricDGP
from .protocol import DataGeneratingProcess
from .sampling import ClusteredSampling, IIDSampling, SamplingDesign

__all__ = [
    "ClusteredSampling",
    "DataGeneratingProcess",
    "EmpiricalDGP",
    "IIDSampling",
    "ParametricDGP",
    "SamplingDesign",
    "TwoStageDGP",
    "with_data",
]

__version__ = "0.1.0a0"
