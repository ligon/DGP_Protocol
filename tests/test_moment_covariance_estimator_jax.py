"""JAX-path tests for ``SamplingDesign.moment_covariance_estimator``.

The numpy parity tests live in ``test_moment_covariance_estimator.py``.
This module adds the JAX-array dispatch path:

- :class:`IIDSampling` and :class:`ClusteredSampling` should accept a
  ``gi`` returning a JAX array and run the entire ``Omega`` computation
  in ``jnp`` (rather than coercing to numpy at the boundary).
- The result must agree with the numpy path bit-for-bit (modulo float
  rounding tolerated by ``np.testing.assert_allclose``).
- The result must be JAX-traceable so consumers can differentiate
  through ``theta`` (e.g., ManifoldGMM's CUE gradient).

JAX is an optional dependency.  Skipped when ``jax`` is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# JAX defaults to float32; downstream consumers (e.g. ManifoldGMM) enable
# x64 globally on import.  Match that convention so numerical-parity
# tests can assert tight tolerances against the float64 numpy path.
jax.config.update("jax_enable_x64", True)  # noqa: E402

from dgp_protocol import ClusteredSampling, IIDSampling  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (mirror the numpy test module's panel; reused intentionally).
# ---------------------------------------------------------------------------
@pytest.fixture
def panel():
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


def _gi_identity_numpy(theta, obs):
    """Return the observation as the moment matrix (theta ignored)."""

    return obs


def _gi_identity_jax(theta, obs):
    """JAX-array variant of the identity moment function."""

    return jnp.asarray(obs)


# ---------------------------------------------------------------------------
# JAX vs numpy parity.
# ---------------------------------------------------------------------------
def test_iid_jax_matches_numpy(panel):
    """JAX dispatch path produces the same Omega as the numpy path."""

    moments, _ = panel
    np_omega = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_numpy, centered=True
    )
    jax_omega = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_jax, centered=True
    )
    # The JAX path returns a JAX array; cast for comparison.
    assert isinstance(jax_omega, jax.Array)
    np.testing.assert_allclose(np.asarray(jax_omega), np_omega, atol=1e-10)


def test_clustered_jax_matches_numpy(panel):
    moments, cluster_ids = panel
    np_omega = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_numpy, centered=True
    )
    jax_omega = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_jax, centered=True
    )
    assert isinstance(jax_omega, jax.Array)
    np.testing.assert_allclose(np.asarray(jax_omega), np_omega, atol=1e-10)


def test_iid_jax_with_weights_matches_numpy(panel):
    moments, _ = panel
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.5, 1.5, size=moments.shape[0])
    np_omega = IIDSampling(weights=weights).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_numpy, centered=True
    )
    jax_omega = IIDSampling(weights=weights).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_jax, centered=True
    )
    np.testing.assert_allclose(np.asarray(jax_omega), np_omega, atol=1e-10)


def test_clustered_jax_with_weights_matches_numpy(panel):
    moments, cluster_ids = panel
    rng = np.random.default_rng(7)
    weights = rng.uniform(0.5, 1.5, size=moments.shape[0])
    np_omega = ClusteredSampling(
        cluster_ids=cluster_ids, weights=weights
    ).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_numpy, centered=True
    )
    jax_omega = ClusteredSampling(
        cluster_ids=cluster_ids, weights=weights
    ).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_jax, centered=True
    )
    np.testing.assert_allclose(np.asarray(jax_omega), np_omega, atol=1e-10)


def test_iid_jax_uncentered_matches_numpy(panel):
    moments, _ = panel
    np_omega = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_numpy, centered=False
    )
    jax_omega = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_jax, centered=False
    )
    np.testing.assert_allclose(np.asarray(jax_omega), np_omega, atol=1e-10)


def test_clustered_jax_uncentered_matches_numpy(panel):
    moments, cluster_ids = panel
    np_omega = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_numpy, centered=False
    )
    jax_omega = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
        moments, theta=None, gi=_gi_identity_jax, centered=False
    )
    np.testing.assert_allclose(np.asarray(jax_omega), np_omega, atol=1e-10)


# ---------------------------------------------------------------------------
# Properties of the JAX-output.
# ---------------------------------------------------------------------------
def test_jax_output_dtype_and_psd(panel):
    """Result is a JAX float array; symmetric; non-negative eigenvalues."""

    moments, cluster_ids = panel
    for design in (
        IIDSampling(),
        ClusteredSampling(cluster_ids=cluster_ids),
    ):
        omega = design.moment_covariance_estimator(
            moments, theta=None, gi=_gi_identity_jax, centered=True
        )
        assert isinstance(omega, jax.Array)
        # Symmetric to machine precision (output of PSD projection).
        np.testing.assert_allclose(np.asarray(omega), np.asarray(omega).T, atol=1e-10)
        eigs = np.linalg.eigvalsh(np.asarray(omega))
        assert eigs.min() >= -1e-10


# ---------------------------------------------------------------------------
# Autodiff through theta.
# ---------------------------------------------------------------------------
def test_iid_jax_differentiable_through_theta(panel):
    """JAX autodiff sees a non-zero gradient w.r.t. theta in a location-moment.

    For ``gi(theta, X) = X - theta``, ``Omega(theta)`` is *constant in
    theta* (only the mean shifts, which is subtracted by centering).
    A more discriminating test uses a moment that genuinely depends on
    theta: ``gi(theta, X) = theta * X``.  Then
    ``Omega(theta) = theta^2 * Omega(1)``, so
    ``dOmega/dtheta = 2 * theta * Omega(1)``, a known closed form.
    """

    moments, _ = panel
    obs_jax = jnp.asarray(moments)

    def omega_scalar(theta_scalar):
        # theta_scalar: shape ()
        # gi multiplies every entry by theta -> moments scale linearly
        def gi(theta, X):
            return theta * X

        omega = IIDSampling().moment_covariance_estimator(
            obs_jax, theta=theta_scalar, gi=gi, centered=True
        )
        # Sum the diagonal for a scalar objective to differentiate.
        return jnp.trace(omega)

    # Closed-form gradient: trace(Omega(theta)) = theta^2 * C
    # where C = trace(Omega(1)), so d/dtheta = 2 * theta * C.
    grad_fn = jax.grad(omega_scalar)

    theta_value = jnp.array(2.0)
    grad_at_theta = float(grad_fn(theta_value))

    # Compute closed form via the numpy reference path.
    def gi_one(theta, X):
        return X  # theta=1 case

    omega_at_one = IIDSampling().moment_covariance_estimator(
        moments, theta=None, gi=gi_one, centered=True
    )
    C = float(np.trace(omega_at_one))
    expected = 2.0 * float(theta_value) * C
    np.testing.assert_allclose(grad_at_theta, expected, rtol=1e-5)


def test_clustered_jax_differentiable_through_theta(panel):
    """Same as above but with ClusteredSampling -- segment_sum must autodiff."""

    moments, cluster_ids = panel
    obs_jax = jnp.asarray(moments)

    def omega_scalar(theta_scalar):
        def gi(theta, X):
            return theta * X

        omega = ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
            obs_jax, theta=theta_scalar, gi=gi, centered=True
        )
        return jnp.trace(omega)

    grad_fn = jax.grad(omega_scalar)
    theta_value = jnp.array(2.0)
    grad_at_theta = float(grad_fn(theta_value))

    def gi_one(theta, X):
        return X

    omega_at_one = ClusteredSampling(
        cluster_ids=cluster_ids
    ).moment_covariance_estimator(moments, theta=None, gi=gi_one, centered=True)
    C = float(np.trace(omega_at_one))
    expected = 2.0 * float(theta_value) * C
    np.testing.assert_allclose(grad_at_theta, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# jax.jit wrapping.
# ---------------------------------------------------------------------------
def test_iid_jax_jit_compatible(panel):
    """The IID Omega computation is jax.jit-compatible."""

    moments, _ = panel
    obs_jax = jnp.asarray(moments)

    def compute_omega(theta_scalar):
        def gi(theta, X):
            return theta * X

        return IIDSampling().moment_covariance_estimator(
            obs_jax, theta=theta_scalar, gi=gi, centered=True
        )

    jit_omega = jax.jit(compute_omega)
    result = jit_omega(jnp.array(1.5))
    assert isinstance(result, jax.Array)
    assert result.shape == (3, 3)


def test_clustered_jax_jit_compatible(panel):
    """The clustered Omega computation is jax.jit-compatible."""

    moments, cluster_ids = panel
    obs_jax = jnp.asarray(moments)

    def compute_omega(theta_scalar):
        def gi(theta, X):
            return theta * X

        return ClusteredSampling(cluster_ids=cluster_ids).moment_covariance_estimator(
            obs_jax, theta=theta_scalar, gi=gi, centered=True
        )

    jit_omega = jax.jit(compute_omega)
    result = jit_omega(jnp.array(1.5))
    assert isinstance(result, jax.Array)
    assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# Cluster-ids length mismatch raises (same error as numpy path).
# ---------------------------------------------------------------------------
def test_clustered_jax_cluster_ids_wrong_length_raises(panel):
    moments, _ = panel
    bad_ids = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="cluster_ids has 3"):
        ClusteredSampling(cluster_ids=bad_ids).moment_covariance_estimator(
            moments, theta=None, gi=_gi_identity_jax, centered=True
        )
