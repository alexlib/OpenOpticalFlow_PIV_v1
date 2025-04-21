import numpy as np
import pytest
from numpy.testing import assert_allclose

from openopticalflow.preprocessing import (
    pre_processing,
    correction_illumination,
    shift_image_refine
)

@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    # Create two simple test images
    size = 50
    img1 = np.zeros((size, size))
    img2 = np.zeros((size, size))
    
    # Add some patterns
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    img1 = np.exp(-(x**2 + y**2) / 0.5)  # Gaussian
    img2 = np.exp(-((x-0.2)**2 + (y-0.1)**2) / 0.5)  # Shifted Gaussian
    
    return img1, img2

@pytest.fixture
def sample_velocity_field():
    """Create sample velocity field for testing."""
    size = 20
    ux = np.ones((size, size)) * 0.2
    uy = np.ones((size, size)) * 0.1
    return ux, uy

@pytest.mark.preprocessing
def test_pre_processing_no_scaling(sample_images):
    """Test pre_processing without scaling."""
    img1, img2 = sample_images
    
    # Process with no scaling
    processed1, processed2 = pre_processing(img1, img2, scale_im=1.0, size_filter=0.5)
    
    # Check shapes are preserved
    assert processed1.shape == img1.shape
    assert processed2.shape == img2.shape
    
    # Check that filtering was applied (values should be different but close)
    assert not np.array_equal(processed1, img1)
    assert not np.array_equal(processed2, img2)
    
    # Check that the images are still similar (correlation)
    corr1 = np.corrcoef(img1.flatten(), processed1.flatten())[0, 1]
    corr2 = np.corrcoef(img2.flatten(), processed2.flatten())[0, 1]
    assert corr1 > 0.9, f"Correlation too low: {corr1}"
    assert corr2 > 0.9, f"Correlation too low: {corr2}"

@pytest.mark.preprocessing
def test_pre_processing_with_scaling(sample_images):
    """Test pre_processing with scaling."""
    img1, img2 = sample_images
    
    # Process with scaling
    scale_factor = 0.5
    processed1, processed2 = pre_processing(img1, img2, scale_im=scale_factor, size_filter=0.5)
    
    # Check shapes are scaled correctly
    expected_shape = (int(img1.shape[0] * scale_factor), int(img1.shape[1] * scale_factor))
    assert processed1.shape == expected_shape
    assert processed2.shape == expected_shape

@pytest.mark.preprocessing
def test_correction_illumination(sample_images):
    """Test correction_illumination function."""
    img1, img2 = sample_images
    
    # Add illumination differences
    img1_bright = img1 + 0.2
    img2_dark = img2 - 0.1
    
    # Apply illumination correction
    window = [0, img1.shape[1], 0, img1.shape[0]]  # Full image
    size_avg = 5
    corrected1, corrected2 = correction_illumination(img1_bright, img2_dark, window, size_avg)
    
    # Check shapes are preserved
    assert corrected1.shape == img1.shape
    assert corrected2.shape == img2.shape
    
    # Check that correction was applied
    assert not np.array_equal(corrected1, img1_bright)
    assert not np.array_equal(corrected2, img2_dark)
    
    # Check that mean values are closer after correction
    diff_before = abs(np.mean(img1_bright) - np.mean(img2_dark))
    diff_after = abs(np.mean(corrected1) - np.mean(corrected2))
    assert diff_after < diff_before, "Illumination correction did not reduce mean difference"

@pytest.mark.preprocessing
def test_correction_illumination_no_correction(sample_images):
    """Test correction_illumination with size_average=0 (no correction)."""
    img1, img2 = sample_images
    
    # Apply with size_average=0 (should return original images)
    window = [0, img1.shape[1], 0, img1.shape[0]]
    corrected1, corrected2 = correction_illumination(img1, img2, window, 0)
    
    # Check that no correction was applied
    assert_allclose(corrected1, img1)
    assert_allclose(corrected2, img2)

@pytest.mark.preprocessing
def test_shift_image_refine(sample_images, sample_velocity_field):
    """Test shift_image_refine function."""
    img1, img2 = sample_images
    ux, uy = sample_velocity_field
    
    # Apply image shifting
    shifted, ux_interp, uy_interp = shift_image_refine(ux, uy, img1, img2)
    
    # Check shapes
    assert shifted.shape == img1.shape
    assert ux_interp.shape == img1.shape
    assert uy_interp.shape == img1.shape
    
    # Check that shifting was applied (image should be different)
    assert not np.array_equal(shifted, img1)
    
    # Check that interpolated velocity fields have expected values
    assert np.isclose(np.mean(ux_interp), np.mean(ux), rtol=0.1)
    assert np.isclose(np.mean(uy_interp), np.mean(uy), rtol=0.1)

@pytest.mark.preprocessing
def test_shift_image_refine_same_size(sample_images):
    """Test shift_image_refine with velocity fields of same size as images."""
    img1, img2 = sample_images
    
    # Create velocity fields of same size as images
    ux = np.ones_like(img1) * 0.2
    uy = np.ones_like(img1) * 0.1
    
    # Apply image shifting
    shifted, ux_interp, uy_interp = shift_image_refine(ux, uy, img1, img2)
    
    # Check that velocity fields were not interpolated (should be identical)
    assert_allclose(ux_interp, ux)
    assert_allclose(uy_interp, uy)
