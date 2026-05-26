"""EmpiricalDGP + sample_distribution: bootstrap-distribution showcase.

Illustrative; not part of the public ``dgp_protocol`` API.

This example covers the territory the other example files don't:

- :class:`~dgp_protocol.EmpiricalDGP` -- the bootstrap / empirical-
  distribution wrapper, with both
  :class:`~dgp_protocol.IIDSampling` (default) and
  :class:`~dgp_protocol.ClusteredSampling` (cluster-block bootstrap).
- The **P-side analytic shortcut**: under ``IIDSampling``,
  ``empirical_dgp.mean()`` / ``.var()`` / ``.cov()`` return the
  moments of F̂ in closed form -- no Monte Carlo at all.
- The **D-side surface** via ``dgp.sample_distribution``: Monte
  Carlo expectations / covariances of arbitrary statistics under
  the DGP's distribution over datasets.  This includes
  ``moment_covariance(theta, gi)``, the analog-estimation entry
  point ManifoldGMM's ``omega_hat`` is designed to consume.
- The **parallel-fan-out idiom**: ``[parent.with_rng(s) for s in
  parent._rng.spawn(N)]`` -- spawn N independent child Generators
  for parallel bootstrap workers.

Synthesised observation
-----------------------
We synthesise an observation with deliberate cluster structure:
``N_CLUSTERS`` clusters of ``ROWS_PER_CLUSTER`` rows each, where
each cluster carries a cluster-level random offset on top of
within-cluster iid noise.  This means rows within a cluster are
positively correlated (sharing the offset); rows across clusters
are independent.  The cluster-block bootstrap preserves that
within-cluster correlation; the iid bootstrap destroys it.  So the
two ``sample_distribution.moment_covariance`` values below differ
in a way that mirrors the classical iid-vs-cluster-robust
variance contrast.

Run directly::

    python examples/empirical_bootstrap.py

Or as a module::

    python -m examples.empirical_bootstrap
"""

from __future__ import annotations

# Allow running as a script (``python examples/empirical_bootstrap.py``
# or ``%run examples/empirical_bootstrap.py``) by putting the repo
# root on ``sys.path`` so ``examples`` resolves as a package.  Also
# works under ``python -m examples.empirical_bootstrap``.
if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "examples"

import numpy as np

from dgp_protocol import (
    ClusteredSampling,
    DataGeneratingProcess,
    EmpiricalDGP,
)

# ---------------------------------------------------------------------------
# Synthesise an observation with cluster structure.
# ---------------------------------------------------------------------------
N_CLUSTERS: int = 20
ROWS_PER_CLUSTER: int = 10
N: int = N_CLUSTERS * ROWS_PER_CLUSTER  # 200 rows
P: int = 2  # columns

SAMPLE_SEED: int = 2029
CLUSTER_OFFSET_SIGMA: float = 0.8
WITHIN_NOISE_SIGMA: float = 0.6

_rng = np.random.default_rng(SAMPLE_SEED)
_cluster_offsets = CLUSTER_OFFSET_SIGMA * _rng.standard_normal(size=(N_CLUSTERS, P))
_within_noise = WITHIN_NOISE_SIGMA * _rng.standard_normal(size=(N, P))
CLUSTER_IDS: np.ndarray = np.repeat(np.arange(N_CLUSTERS), ROWS_PER_CLUSTER)
OBSERVATION: np.ndarray = _cluster_offsets[CLUSTER_IDS] + _within_noise

# ---------------------------------------------------------------------------
# Two parallel DGPs from the same observation.
# ---------------------------------------------------------------------------
IID_SEED: int = 1
CLUSTERED_SEED: int = 1

iid_dgp = EmpiricalDGP(observation=OBSERVATION, seed=IID_SEED)
clustered_dgp = EmpiricalDGP(
    observation=OBSERVATION,
    sampling=ClusteredSampling(cluster_ids=CLUSTER_IDS),
    seed=CLUSTERED_SEED,
)


# ---------------------------------------------------------------------------
# Demo.
# ---------------------------------------------------------------------------
def _demo() -> None:
    """Walk through the P-side analytic path, the D-side MC path, and
    the parallel-fan-out idiom."""

    assert isinstance(iid_dgp, DataGeneratingProcess)
    assert isinstance(clustered_dgp, DataGeneratingProcess)

    print(
        f"observation: N = {N} ({N_CLUSTERS} clusters x "
        f"{ROWS_PER_CLUSTER} rows), p = {P} columns"
    )
    print()

    # ----- P-side: analytic moments of F̂ (exact, no MC) -----
    print("P-side (per-observation marginal of F̂, IID sampling):")
    print(f"  mean = {iid_dgp.mean()}     # exact analytic, no MC")
    print(f"  var  = {iid_dgp.var()}     # exact analytic, ddof=1")
    print(f"  cov  = {iid_dgp.cov().flatten()}" f"    # 2x2 sample cov flattened")
    print()

    # ----- D-side: sampling distribution of the sample mean -----
    # E_DGP[X̄] for an EmpiricalDGP under iid bootstrap converges to
    # the observation's column mean (the bootstrap is unbiased for the
    # sample mean).
    print("D-side (sampling distribution of the sample mean, via MC):")
    sample_mean_under_iid = iid_dgp.sample_distribution.expect(
        lambda r: r.mean(axis=0),
        atol=0.02,
        rtol=0.0,
        max_its=2000,
        batch_size=100,
    )
    sample_mean_under_cluster = clustered_dgp.sample_distribution.expect(
        lambda r: r.mean(axis=0),
        atol=0.02,
        rtol=0.0,
        max_its=2000,
        batch_size=100,
    )
    print(f"  E_DGP[X̄] under iid bootstrap:     {sample_mean_under_iid}")
    print(f"  E_DGP[X̄] under cluster bootstrap: {sample_mean_under_cluster}")
    print(f"  (data X̄ for comparison:           {OBSERVATION.mean(axis=0)})")
    print()

    # ----- D-side: moment-vector covariance under iid vs cluster bootstrap -----
    # gi(theta, X) = X - theta returns the moment-condition matrix
    # (X - theta) which, under the true theta = E[X], has zero mean
    # and covariance equal to Cov(X).  The cluster-robust variance
    # of g_bar_N is larger when there is positive intra-cluster
    # correlation.
    theta_true = OBSERVATION.mean(axis=0)

    def gi(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        return X - theta

    print("D-side (moment_covariance for gi(theta, X) = X - theta):")
    omega_iid = iid_dgp.sample_distribution.moment_covariance(
        theta=theta_true,
        gi=gi,
        atol=0.05,
        rtol=0.0,
        max_its=2000,
        batch_size=100,
    )
    omega_cluster = clustered_dgp.sample_distribution.moment_covariance(
        theta=theta_true,
        gi=gi,
        atol=0.05,
        rtol=0.0,
        max_its=2000,
        batch_size=100,
    )
    omega_iid_arr = np.asarray(omega_iid)
    omega_cluster_arr = np.asarray(omega_cluster)
    print("  omega_hat under iid bootstrap:")
    print(f"    {omega_iid_arr.flatten()}")
    print("  omega_hat under cluster bootstrap:")
    print(f"    {omega_cluster_arr.flatten()}")
    print(
        f"  cluster/iid ratio (trace): "
        f"{np.trace(omega_cluster_arr) / np.trace(omega_iid_arr):.3f}x"
    )
    print("  -- cluster bootstrap preserves within-cluster correlation,")
    print("     so omega_hat is larger when there is intra-cluster correlation.")
    print()

    # ----- Parallel fan-out via with_rng + spawn -----
    # The canonical pattern for parallel bootstrap workers: spawn N
    # independent child Generators off the parent's stream, then
    # rebind each child onto a sibling DGP via with_rng.  Each
    # sibling shares structural attributes with the parent but draws
    # from an independent stream.
    N_WORKERS = 4
    parent = EmpiricalDGP(observation=OBSERVATION, seed=42)
    worker_rngs = parent._rng.spawn(N_WORKERS)
    workers = [parent.with_rng(r) for r in worker_rngs]
    print(f"Parallel fan-out: spawned {N_WORKERS} workers via with_rng + spawn")
    print("  first-draw column-means per worker:")
    for k, w in enumerate(workers):
        first_draw = w.draw()
        print(f"    worker[{k}]: {first_draw.mean(axis=0)}")
    print(
        "  (parent's stream is unchanged; spawn advances a separate "
        "counter, not draw state.)"
    )


if __name__ == "__main__":
    _demo()
