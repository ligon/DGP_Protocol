"""Warnings raised by ``dgp_protocol``.

A small module so consumer code can ``filterwarnings`` on package-specific
categories without importing implementation modules.

Currently only :class:`NumericalWarning` is defined; future numerical-
quality signals (e.g., MC convergence stalls, ill-conditioned sample
covariance from very-small Monte Carlo samples) should be raised under
the same category unless a distinct subclass is materially useful.
"""

from __future__ import annotations


class NumericalWarning(RuntimeWarning):
    """Numerical-quality concern raised by ``dgp_protocol``.

    Emitted when an adaptive computation could not achieve the requested
    accuracy within its iteration budget, or when backend-specific kwargs
    were silently ignored because the analytic backend was unavailable.

    Subclasses :class:`RuntimeWarning` so it is shown by default.
    """
