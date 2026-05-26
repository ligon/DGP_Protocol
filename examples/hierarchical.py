"""Hierarchical two-stage DGP: students within schools.

Illustrative; not part of the public ``dgp_protocol`` API.

Model
-----
::

    mu_s   ~ N(0, SIGMA_MU**2),   s = 0..S-1   (school means)
    y_{s,i} = mu_s + e_{s,i},     i = 0..K-1   (student scores)
    e_{s,i} ~ N(0, SIGMA_E**2)                 (within-school noise)

A balanced hierarchical sample of ``S`` schools, each containing
``K`` students.  Each school carries its own mean ability ``mu_s``
that is shared across the school's students; within-school student
scores are iid Gaussian noise around ``mu_s``.

Composition mechanics
---------------------
The DGP is built as a :class:`~dgp_protocol.TwoStageDGP`:

- The **outer** DGP is a :class:`~dgp_protocol.ParametricDGP` whose
  ``.draw()`` returns a ``(S, 1)`` matrix of school characteristics
  (one row per school, single column = mean ability).
- The **inner** is a callable
  ``(school_chars, rng) -> ParametricDGP`` that builds a
  per-school DGP centered at the school's mean ability.  The
  composite passes a spawned child Generator to ``inner`` for each
  school; the inner DGP installs it via
  :meth:`~dgp_protocol.ParametricDGP.with_rng` so the composite's
  own seed drives all within-school randomness deterministically.

A single ``ts.draw()`` therefore returns a Python list of length
``S``, with element ``s`` being a ``(K, 1)`` numpy array of
student scores for school ``s``.  Per-school sample means are
clustered around the school-level ``mu_s`` (varies across schools);
within-school standard deviations are near ``SIGMA_E``.

Bound observation
-----------------
The DGP is bound to an observed realization (``OBSERVATION``,
a list of S arrays) drawn once at module load with an independent
``(OBS_OUTER_SEED, OBS_COMP_SEED)`` pair, so ``ts.data`` returns
a fixed realization a downstream consumer can estimate from.
Fresh draws via ``ts.draw()`` come from the DGP's own
``(OUTER_SEED, COMP_SEED)`` streams and are useful for Monte Carlo
studies.

Run directly::

    python examples/hierarchical.py

Or as a module::

    python -m examples.hierarchical
"""

from __future__ import annotations

# Allow running as a script (``python examples/hierarchical.py`` or
# ``%run examples/hierarchical.py``) by putting the repo root on
# ``sys.path`` so ``examples`` resolves as a package.  Also works
# unchanged under ``python -m examples.hierarchical``.
if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "examples"

import numpy as np

from dgp_protocol import (
    DataGeneratingProcess,
    ParametricDGP,
    TwoStageDGP,
)

# ---------------------------------------------------------------------------
# Parameters.
# ---------------------------------------------------------------------------
S: int = 8  # number of schools
K: int = 12  # students per school
SIGMA_MU: float = 1.0  # school-mean std dev
SIGMA_E: float = 0.5  # within-school noise std dev

# ---------------------------------------------------------------------------
# Seeds.  Four streams (two outer, two composite) so the bound
# observation and the main DGP's draw stream are fully independent.
# ---------------------------------------------------------------------------
OUTER_SEED: int = 11  # main DGP's outer (school-chars) rng
COMP_SEED: int = 0  # main DGP's composite (within-school spawn) rng
OBS_OUTER_SEED: int = 2030  # bound-observation outer rng
OBS_COMP_SEED: int = 2031  # bound-observation composite rng


# ---------------------------------------------------------------------------
# Outer DGP: school characteristics (one row per school).
# ---------------------------------------------------------------------------
def school_chars(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Draw the per-school characteristic vector.

    Returns a ``(S, 1)`` matrix; column 0 is the school's mean
    ability ``mu_s``.
    """

    del shape  # fixed at (S, 1) by construction
    return SIGMA_MU * rng.standard_normal(size=(S, 1))


# ---------------------------------------------------------------------------
# Inner builder: per-school DGP.
# ---------------------------------------------------------------------------
def school_inner_dgp(chars: np.ndarray, rng: np.random.Generator) -> ParametricDGP:
    """Build a per-school inner DGP given its characteristics and rng.

    Each school's inner DGP draws ``(K, 1)`` student scores from
    ``N(mu_s, SIGMA_E**2)`` where ``mu_s = chars[0]``.  The composite
    DGP spawns ``rng`` per school; we install it on the inner DGP via
    ``with_rng`` so the composite's seed deterministically drives
    within-school randomness.
    """

    mu_s = float(chars[0])
    return ParametricDGP(
        generator=lambda rng_in, sh: mu_s + SIGMA_E * rng_in.standard_normal(sh),
        default_shape=(K, 1),
    ).with_rng(rng)


# ---------------------------------------------------------------------------
# Construction helper: lets us build matched (independent-seed) DGPs
# for the bound observation and for the main draw stream.
# ---------------------------------------------------------------------------
def _build(outer_seed: int, comp_seed: int, observation: object = None) -> TwoStageDGP:
    outer = ParametricDGP(generator=school_chars, default_shape=(S, 1), seed=outer_seed)
    return TwoStageDGP(
        outer=outer, inner=school_inner_dgp, observation=observation, seed=comp_seed
    )


# Materialize the bound observation from an independent build.
OBSERVATION: list = _build(OBS_OUTER_SEED, OBS_COMP_SEED).draw()

# Main DGP: independent seeds, observation pre-bound.
ts = _build(OUTER_SEED, COMP_SEED, observation=OBSERVATION)


# ---------------------------------------------------------------------------
# Demo.
# ---------------------------------------------------------------------------
def _demo() -> None:
    assert isinstance(ts, DataGeneratingProcess)
    assert ts.data is not None
    assert isinstance(ts.data, list)
    assert len(ts.data) == S
    assert all(school.shape == (K, 1) for school in ts.data)

    print(f"hierarchical: S = {S} schools, K = {K} students/school")
    print(
        f"truth:        SIGMA_MU = {SIGMA_MU:.2f} (school-mean std),"
        f"  SIGMA_E = {SIGMA_E:.2f} (within-school std)"
    )
    print()

    # ----- Bound observation: per-school sample means -----
    print(f"observed:  {len(ts.data)} schools, each with shape={ts.data[0].shape}")
    print("  per-school sample means (across the K student scores):")
    for s, school in enumerate(ts.data):
        m = float(school.mean())
        sd = float(school.std(ddof=1))
        print(f"    school {s}: mean = {m:+.3f},  within-school std = {sd:.3f}")
    grand_mean = float(np.vstack(ts.data).mean())
    print(f"  grand mean across all S*K = {S * K} observations: {grand_mean:+.4f}")
    print(
        "  (per-school means scatter ~ N(0, SIGMA_MU**2);"
        " within-school stds scatter ~ SIGMA_E)"
    )
    print()

    # ----- Fresh draw: same structure, different values -----
    fresh = ts.draw()
    print(f"fresh draw: {len(fresh)} schools, each {fresh[0].shape}")
    print("  per-school sample means:")
    for s, school in enumerate(fresh):
        print(f"    school {s}: mean = {float(school.mean()):+.3f}")
    print()

    # ----- Determinism: two constructions with identical seeds agree -----
    ts_a = _build(OUTER_SEED, COMP_SEED)  # no observation needed for this check
    ts_b = _build(OUTER_SEED, COMP_SEED)
    realization_a = ts_a.draw()
    realization_b = ts_b.draw()
    all_equal = all(
        np.array_equal(ra, rb)
        for ra, rb in zip(realization_a, realization_b, strict=True)
    )
    print(f"determinism: same (OUTER_SEED, COMP_SEED) -> identical draws: {all_equal}")


if __name__ == "__main__":
    _demo()
