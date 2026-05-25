"""DGP_Protocol: a minimal Protocol for data-generating processes.

See :class:`DataGeneratingProcess` for the Protocol itself.  Concrete
container wrappers (:class:`EmpiricalDGP`, :class:`ParametricDGP`)
and composition primitives (:class:`TwoStageDGP`, :func:`with_data`)
are re-exported here for convenience; users typically import them as
``from dgp_protocol import ...``.

Distributional features come in two parallel surfaces:

- :mod:`dgp_protocol.marginal` (P-side): free functions
  :func:`expect`, :func:`mean`, :func:`var`, :func:`cov` for the
  per-observation marginal distribution (the analog-estimation
  primitive in Manski's framework).  Re-exported at package level
  for convenience.
- :class:`SampleDistribution` (D-side): accessed via
  ``dgp.sample_distribution``; methods :meth:`expect`, :meth:`cov`,
  :meth:`moment_covariance` operate on the dataset-level sampling
  distribution.

See the design note at ``docs/design/dgp.org`` in the ManifoldGMM
repo for the conceptual framework -- a DGP is the stand-in
distribution against which analog estimators are defined (Manski,
1988, *Analog Estimation Methods in Econometrics*).
"""

from __future__ import annotations

from .composition import TwoStageDGP, with_data
from .empirical import EmpiricalDGP
from .exceptions import AnalyticUnavailable
from .marginal import cov, expect, mean, var
from .parametric import ParametricDGP
from .protocol import DataGeneratingProcess
from .sample_distribution import SampleDistribution
from .sampling import ClusteredSampling, IIDSampling, SamplingDesign
from .warnings import NumericalWarning

__all__ = [
    "AnalyticUnavailable",
    "ClusteredSampling",
    "DataGeneratingProcess",
    "EmpiricalDGP",
    "IIDSampling",
    "NumericalWarning",
    "ParametricDGP",
    "SampleDistribution",
    "SamplingDesign",
    "TwoStageDGP",
    "cov",
    "expect",
    "mean",
    "var",
    "with_data",
]

__version__ = "0.1.0a0"
