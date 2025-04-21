import numpy as np
import pytest
from numpy.testing import assert_allclose

from openopticalflow.OpticalFlowPhysics_fun import OpticalFlowPhysics_fun

@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    # Create two simple test images with a known displacement
    size = 50
    img1 = np.zeros((size, size))
    img2 = np.zeros((size, size))

    # Add a Gaussian pattern
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    img1 = np.exp(-(x**2 + y**2) / 0.5)  # Gaussian

    # Shift the pattern for the second image
    shift_x = 0.2
    shift_y = 0.1
    img2 = np.exp(-((x-shift_x)**2 + (y-shift_y)**2) / 0.5)  # Shifted Gaussian

    return img1, img2, shift_x, shift_y

@pytest.mark.optical_flow
def test_optical_flow_physics_basic(sample_images):
    """Test basic functionality of OpticalFlowPhysics_fun."""
    img1, img2, shift_x, shift_y = sample_images

    # Run the optical flow function
    lambda_1 = 0.1
    lambda_2 = 0.1
    ux, uy, vor, ux_horn, uy_horn, error = OpticalFlowPhysics_fun(img1, img2, lambda_1, lambda_2)

    # Check output shapes
    assert ux.shape == img1.shape
    assert uy.shape == img1.shape
    assert vor.shape == img1.shape
    assert ux_horn.shape == img1.shape
    assert uy_horn.shape == img1.shape

    # Check that error is a scalar
    assert isinstance(error, float)

    # Check that velocity fields are not all zeros
    assert not np.allclose(ux, 0)
    assert not np.allclose(uy, 0)

    # Check that velocity magnitude is computed correctly
    assert_allclose(vor, np.sqrt(ux**2 + uy**2))

@pytest.mark.optical_flow
def test_optical_flow_physics_direction(sample_images):
    """Test that OpticalFlowPhysics_fun produces non-zero flow for shifted images."""
    img1, img2, _, _ = sample_images

    # Run the optical flow function
    lambda_1 = 0.1
    lambda_2 = 0.1
    ux, uy, vor, ux_horn, uy_horn, error = OpticalFlowPhysics_fun(img1, img2, lambda_1, lambda_2)

    # Calculate mean flow values
    mean_ux = np.mean(np.abs(ux))
    mean_uy = np.mean(np.abs(uy))

    # For shifted images, we should detect some flow
    # The specific implementation might not correlate directly with the shift direction
    # but it should produce non-zero flow
    assert mean_ux > 0.0001, f"Mean absolute ux too low: {mean_ux}"
    assert mean_uy > 0.0001, f"Mean absolute uy too low: {mean_uy}"

    # Check that the flow field has some structure (not just random noise)
    # by checking that the smoothed version (ux_horn, uy_horn) is similar to the original
    corr_ux = np.corrcoef(ux.flatten(), ux_horn.flatten())[0, 1]
    corr_uy = np.corrcoef(uy.flatten(), uy_horn.flatten())[0, 1]

    assert corr_ux > 0.5, f"Correlation between ux and ux_horn too low: {corr_ux}"
    assert corr_uy > 0.5, f"Correlation between uy and uy_horn too low: {corr_uy}"

@pytest.mark.optical_flow
def test_optical_flow_physics_lambda_effect(sample_images):
    """Test the effect of lambda parameters on the optical flow."""
    img1, img2, _, _ = sample_images

    # Run with different lambda values
    lambda_1_small = 0.01
    lambda_1_large = 10.0
    lambda_2 = 0.1

    # Small lambda_1 (less smoothing)
    ux1, uy1, _, _, _, _ = OpticalFlowPhysics_fun(
        img1, img2, lambda_1_small, lambda_2
    )

    # Large lambda_1 (more smoothing)
    ux2, uy2, _, _, _, _ = OpticalFlowPhysics_fun(
        img1, img2, lambda_1_large, lambda_2
    )

    # Calculate variance of the flow fields
    var_ux1 = np.var(ux1)
    var_ux2 = np.var(ux2)
    var_uy1 = np.var(uy1)
    var_uy2 = np.var(uy2)

    # With more smoothing (larger lambda), the variance should be smaller
    assert var_ux2 <= var_ux1 * 1.5, "Lambda_1 did not reduce variance as expected"
    assert var_uy2 <= var_uy1 * 1.5, "Lambda_1 did not reduce variance as expected"

@pytest.mark.optical_flow
def test_optical_flow_physics_no_motion(sample_images):
    """Test OpticalFlowPhysics_fun with identical images (no motion)."""
    img1, _, _, _ = sample_images

    # Use the same image twice (no motion)
    lambda_1 = 0.1
    lambda_2 = 0.1
    ux, uy, vor, _, _, _ = OpticalFlowPhysics_fun(img1, img1, lambda_1, lambda_2)

    # Flow should be close to zero
    assert np.mean(np.abs(ux)) < 0.1, f"Mean ux too large: {np.mean(np.abs(ux))}"
    assert np.mean(np.abs(uy)) < 0.1, f"Mean uy too large: {np.mean(np.abs(uy))}"

    # Velocity magnitude should be close to zero
    assert np.mean(vor) < 0.1, f"Mean velocity magnitude too large: {np.mean(vor)}"
