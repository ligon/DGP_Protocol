"""Public exception types raised by ``dgp_protocol``.

A small module so consumer code can ``except`` on package-specific
categories without importing implementation modules.
"""

from __future__ import annotations


class AnalyticUnavailable(NotImplementedError):
    """The requested distributional operation has no analytic form on this DGP.

    Raised by container methods (``expect``/``mean``/``var``/``cov``)
    when the DGP cannot produce the answer in closed form -- typically
    because the underlying ``distribution`` doesn't expose the relevant
    attribute, or because the DGP was constructed with a custom
    ``generator`` callable and no analytic backend.

    Free functions (``dgp_protocol.expect`` etc.) **catch**
    ``AnalyticUnavailable`` and fall back to adaptive Monte Carlo.
    Direct method callers receive the exception and can choose
    between calling the free function, supplying a different DGP,
    or computing the value differently.

    Subclasses :class:`NotImplementedError` so legacy ``except
    NotImplementedError`` handlers continue to work; new code should
    catch this specifically to distinguish "no analytic, try MC" from
    "DGP refuses the operation entirely".
    """
