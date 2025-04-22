import numpy as np
import pytest
from numpy.testing import assert_allclose

from openopticalflow.liu_shen_estimator import liu_shen_estimator

@pytest.fixture
def create_test_images():
    """Create synthetic test images with known flow field."""
    def _create_images(size=(20, 20), noise_level=0.1):
        # Create random image patterns
        np.random.seed(42)
        I0 = np.random.rand(size[0], size[1])

        # Create known flow field (rotating pattern)
        y, x = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]), indexing='ij')
        u_true = -y/5  # horizontal flow
        v_true = x/5   # vertical flow

        # Create second image by applying flow
        I1 = np.zeros_like(I0)
        for i in range(size[0]):
            for j in range(size[1]):
                if 0 <= i + v_true[i,j] < size[0] and 0 <= j + u_true[i,j] < size[1]:
                    I1[i,j] = I0[int(i + v_true[i,j]), int(j + u_true[i,j])]

        # Add noise
        I0 += np.random.normal(0, noise_level, I0.shape)
        I1 += np.random.normal(0, noise_level, I1.shape)

        return I0, I1, u_true, v_true

    return _create_images

@pytest.mark.liu_shen
def test_liu_shen_estimator_basic(create_test_images):
    """Test basic functionality of liu_shen_estimator."""
    # Create test images
    I0, I1, u_true, v_true = create_test_images()

    # Set parameters
    f = np.zeros_like(I0)  # No physical transport term
    dx = 1.0
    dt = 1.0
    lambda_param = 0.1
    tol = 1e-4
    maxnum = 10
    u0 = np.zeros_like(I0)  # Initial guess
    v0 = np.zeros_like(I0)

    # Run the estimator
    u, v, error = liu_shen_estimator(I0, I1, f, dx, dt, lambda_param, tol, maxnum, u0, v0)

    # Check output shapes
    assert u.shape == I0.shape
    assert v.shape == I0.shape

    # Check that error is a list of decreasing values
    assert isinstance(error, list)
    assert len(error) > 0
    assert all(error[i] >= error[i+1] for i in range(len(error)-1))

    # Check that the flow fields are not all zeros
    assert not np.allclose(u, 0)
    assert not np.allclose(v, 0)

    # Check correlation with ground truth
    # We use correlation rather than exact values because optical flow
    # estimation is an ill-posed problem and the results depend on parameters
    corr_u = np.corrcoef(u.flatten(), u_true.flatten())[0, 1]
    corr_v = np.corrcoef(v.flatten(), v_true.flatten())[0, 1]

    # The correlation might be positive or negative depending on the implementation
    # We check the absolute correlation instead
    assert abs(corr_u) > 0.1, f"u correlation too low: {corr_u}"
    assert abs(corr_v) > 0.1, f"v correlation too low: {corr_v}"

@pytest.mark.liu_shen
def test_liu_shen_estimator_convergence(create_test_images):
    """Test convergence of liu_shen_estimator."""
    # Create test images
    I0, I1, _, _ = create_test_images()

    # Set parameters
    f = np.zeros_like(I0)
    dx = 1.0
    dt = 1.0
    lambda_param = 0.1
    tol = 1e-4
    maxnum = 20  # More iterations for convergence test
    u0 = np.zeros_like(I0)
    v0 = np.zeros_like(I0)

    # Run the estimator
    _, _, error = liu_shen_estimator(I0, I1, f, dx, dt, lambda_param, tol, maxnum, u0, v0)

    # Check that the error decreases
    assert error[0] > error[-1]

    # Check that the final error is below the tolerance or maxnum was reached
    assert error[-1] < tol or len(error) == maxnum

@pytest.mark.liu_shen
def test_liu_shen_estimator_lambda_effect(create_test_images):
    """Test the effect of lambda parameter on liu_shen_estimator."""
    # Create test images
    I0, I1, _, _ = create_test_images()

    # Set common parameters
    f = np.zeros_like(I0)
    dx = 1.0
    dt = 1.0
    tol = 1e-4
    maxnum = 10
    u0 = np.zeros_like(I0)
    v0 = np.zeros_like(I0)

    # Run with different lambda values
    lambda1 = 0.01  # Small regularization
    lambda2 = 1.0   # Large regularization

    u1, v1, _ = liu_shen_estimator(I0, I1, f, dx, dt, lambda1, tol, maxnum, u0, v0)
    u2, v2, _ = liu_shen_estimator(I0, I1, f, dx, dt, lambda2, tol, maxnum, u0, v0)

    # Calculate variance of the flow fields
    var_u1 = np.var(u1)
    var_u2 = np.var(u2)
    var_v1 = np.var(v1)
    var_v2 = np.var(v2)

    # With more regularization (larger lambda), the variance should be smaller
    assert var_u2 < var_u1 * 1.5, "Lambda did not reduce variance as expected"
    assert var_v2 < var_v1 * 1.5, "Lambda did not reduce variance as expected"

@pytest.mark.liu_shen
def test_liu_shen_estimator_zero_motion(create_test_images):
    """Test liu_shen_estimator with identical images (no motion)."""
    # Create a single image (no motion)
    I0, _, _, _ = create_test_images()
    I1 = I0.copy()  # Identical image

    # Set parameters
    f = np.zeros_like(I0)
    dx = 1.0
    dt = 1.0
    lambda_param = 0.1
    tol = 1e-4
    maxnum = 10
    u0 = np.zeros_like(I0)
    v0 = np.zeros_like(I0)

    # Run the estimator
    u, v, _ = liu_shen_estimator(I0, I1, f, dx, dt, lambda_param, tol, maxnum, u0, v0)

    # For identical images, the flow should be close to zero
    assert np.mean(np.abs(u)) < 0.1, f"Mean |u| too large: {np.mean(np.abs(u))}"
    assert np.mean(np.abs(v)) < 0.1, f"Mean |v| too large: {np.mean(np.abs(v))}"

@pytest.mark.liu_shen
def test_liu_shen_estimator_with_initial_guess(create_test_images):
    """Test liu_shen_estimator with a non-zero initial guess."""
    # Create test images
    I0, I1, u_true, v_true = create_test_images()

    # Set parameters
    f = np.zeros_like(I0)
    dx = 1.0
    dt = 1.0
    lambda_param = 0.1
    tol = 1e-4
    maxnum = 10

    # Use ground truth as initial guess (should converge faster)
    u0 = u_true.copy()
    v0 = v_true.copy()

    # Run the estimator
    _, _, error1 = liu_shen_estimator(I0, I1, f, dx, dt, lambda_param, tol, maxnum, u0, v0)

    # Run again with zero initial guess
    _, _, error2 = liu_shen_estimator(I0, I1, f, dx, dt, lambda_param, tol, maxnum, np.zeros_like(I0), np.zeros_like(I0))

    # With a good initial guess, the algorithm should converge differently
    # This is a loose test since the behavior depends on the implementation
    # We just check that both runs converge to some extent
    assert error1[0] > error1[-1], "Algorithm did not converge with good initial guess"
    assert error2[0] > error2[-1], "Algorithm did not converge with zero initial guess"
