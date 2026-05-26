"""Parity tests for ``SamplingDesign.moment_covariance_estimator``.

The formula is a direct port of ManifoldGMM's
``MomentRestriction.omega_hat`` (centered + per-moment scaled by
sqrt(N) + group-summed for cluster, iid for iid, weighted mean for
centering when ``weights`` is set, PSD-projection at the end).
These tests pin numerical parity against an inline reimplementation
of that formula so any future drift is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from dgp_protocol import ClusteredSampling, IIDSampling


# ---------------------------------------------------------------------------
# Reference implementation: a faithful inline port of ManifoldGMM's
# ``MomentRestriction.omega_hat`` (numpy backend) for parity checks.
# ---------------------------------------------------------------------------
def _reference_omega(
    moments: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    centered: bool = True,
) -> np.ndarray:
    """Inline replica of MomentRestriction.omega_hat (numpy path)."""

    arr = np.asarray(moments, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]

    # Per-column non-NaN counts.
    counts = (~np.isnan(arr)).sum(axis=0).astype(float)
    safe_counts = np.where(counts > 0, counts, 1.0)
    scale = np.sqrt(safe_counts)

    if centered:
        if weights is None:
            mean = np.nanmean(arr, axis=0)
        else:
            w = np.asarray(weights, dtype=float).reshape(n, 1)
            mean = np.nansum(w * arr, axis=0) / n
        centered_arr = arr - mean.reshape(1, -1)
    else:
        centered_arr = arr

    scaled = centered_arr / scale.reshape(1, -1)

    if cluster_ids is None:
        omega = scaled.T @ scaled
    else:
        cleaned = np.where(np.isnan(scaled), 0.0, scaled)
        _, codes = np.unique(cluster_ids, return_inverse=True)
        codes = np.asarray(codes, dtype=np.int64)
        num_clusters = int(codes.max()) + 1
        grouped = np.zeros((num_clusters, cleaned.shape[1]), dtype=cleaned.dtype)
        np.add.at(grouped, codes, cleaned)
        omega = grouped.T @ grouped

    # PSD projection: symmetrise + eigenvalue-clip.
    sym = 0.5 * (omega + omega.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    return eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
@pytest.fixture
def panel():
    """A small N=60, k=3 moment matrix with positive intra-cluster
    correlation (20 clusters of 3 rows each)."""

    rng = np.random.default_rng(2030)
    n_clusters = 20
    rows_per_cluster = 3
    n = n_clusters * rows_per_cluster
    k = 3
    cluster_offsets = 0.6 * rng.standard_normal(size=(n_clusters, k))
    within = 0.4 * rng.standard_normal(size=(n, k))
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    moments = cluster_offsets[cluster_ids] + within
    return moments, cluster_ids


def _identity_gi(theta, obs):
    """Use the panel itself as the moment matrix (theta is ignored)."""

    return obs


# ---------------------------------------------------------------------------
# IIDSampling parity.
# ---------------------------------------------------------------------------
def test_iid_centered_matches_reference(panel) -> None:
    moments, _ = panel
    expected = _reference_omega(moments, centered=True)
    got = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=_identity_gi, centered=True
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_iid_uncentered_matches_reference(panel) -> None:
    moments, _ = panel
    expected = _reference_omega(moments, centered=False)
    got = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=_identity_gi, centered=False
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_iid_with_weights_matches_reference(panel) -> None:
    """Weights enter via the weighted mean used for centering."""

    moments, _ = panel
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.5, 1.5, size=moments.shape[0])
    expected = _reference_omega(moments, weights=weights, centered=True)
    got = IIDSampling(weights=weights).moment_covariance_estimator(
        moments, theta=None, gi=_identity_gi, centered=True
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# ClusteredSampling parity.
# ---------------------------------------------------------------------------
def test_clustered_centered_matches_reference(panel) -> None:
    moments, cluster_ids = panel
    expected = _reference_omega(moments, cluster_ids=cluster_ids, centered=True)
    got = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
        moments, theta=None, gi=_identity_gi, centered=True
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_clustered_uncentered_matches_reference(panel) -> None:
    moments, cluster_ids = panel
    expected = _reference_omega(moments, cluster_ids=cluster_ids, centered=False)
    got = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
        moments, theta=None, gi=_identity_gi, centered=False
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_clustered_with_weights_matches_reference(panel) -> None:
    moments, cluster_ids = panel
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.5, 1.5, size=moments.shape[0])
    expected = _reference_omega(
        moments, cluster_ids=cluster_ids, weights=weights, centered=True
    )
    got = ClusteredSampling(
        cluster_ids=cluster_ids, weights=weights
    ).moment_covariance_estimator(moments, theta=None, gi=_identity_gi, centered=True)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_singleton_clusters_collapse_to_iid(panel) -> None:
    """When every cluster has size one, cluster-sandwich = iid outer product."""

    moments, _ = panel
    singletons = np.arange(moments.shape[0])
    iid_result = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=_identity_gi, centered=True
    )
    cluster_result = ClusteredSampling(
        cluster_ids=singletons
    ).moment_covariance_estimator(moments, theta=None, gi=_identity_gi, centered=True)
    np.testing.assert_allclose(cluster_result, iid_result, atol=1e-12)


# ---------------------------------------------------------------------------
# Properties of the output.
# ---------------------------------------------------------------------------
def test_output_is_symmetric_and_psd(panel) -> None:
    moments, cluster_ids = panel
    for design in (
        IIDSampling(),
        ClusteredSampling(cluster_ids=cluster_ids),
    ):
        omega = design.moment_covariance_estimator(
            moments, theta=None, gi=_identity_gi, centered=True
        )
        # Symmetric to machine precision (output of PSD projection).
        np.testing.assert_allclose(omega, omega.T, atol=1e-12)
        # All eigenvalues non-negative.
        eigs = np.linalg.eigvalsh(omega)
        assert eigs.min() >= -1e-10


def test_cluster_ids_wrong_length_raises(panel) -> None:
    moments, _ = panel
    bad_ids = np.array([0, 1, 2])  # length 3 != 60
    with pytest.raises(ValueError, match="cluster_ids has 3"):
        ClusteredSampling(cluster_ids=bad_ids).moment_covariance_estimator(
            moments, theta=None, gi=_identity_gi, centered=True
        )


# ---------------------------------------------------------------------------
# gi as a real function (not identity).
# ---------------------------------------------------------------------------
def test_gi_as_actual_moment_function(panel) -> None:
    """``gi(theta, X) = X - theta`` is the canonical location-moment form."""

    moments, cluster_ids = panel
    theta = moments.mean(axis=0)  # centers gi(theta, X) at zero

    def gi(theta, X):
        return X - theta

    expected = _reference_omega(moments - theta, cluster_ids=cluster_ids, centered=True)
    got = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
        moments, theta=theta, gi=gi, centered=True
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)
