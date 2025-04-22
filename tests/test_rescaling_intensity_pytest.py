import numpy as np
import pytest
from numpy.testing import assert_allclose

from openopticalflow.rescaling_intensity import rescaling_intensity

@pytest.mark.preprocessing
def test_rescaling_intensity_normal_images():
    """Test rescaling_intensity with normal images."""
    # Create test images with different intensity ranges
    img1 = np.array([[1, 2, 3], [4, 5, 6]])
    img2 = np.array([[10, 20, 30], [40, 50, 60]])
    
    # Set max intensity value
    max_intensity = 1.0
    
    # Rescale images
    rescaled1, rescaled2 = rescaling_intensity(img1, img2, max_intensity)
    
    # Check that output shapes match input shapes
    assert rescaled1.shape == img1.shape
    assert rescaled2.shape == img2.shape
    
    # Check that outputs are float arrays
    assert rescaled1.dtype == np.float64
    assert rescaled2.dtype == np.float64
    
    # Check that values are in the range [0, max_intensity]
    assert np.min(rescaled1) >= 0
    assert np.max(rescaled1) <= max_intensity
    assert np.min(rescaled2) >= 0
    assert np.max(rescaled2) <= max_intensity
    
    # Check that the maximum value is exactly max_intensity
    assert np.max(rescaled1) == max_intensity
    assert np.max(rescaled2) == max_intensity
    
    # Check that the minimum value is exactly 0
    assert np.min(rescaled1) == 0
    assert np.min(rescaled2) == 0
    
    # Check that the relative ordering of values is preserved
    assert np.all(np.diff(rescaled1.flatten()) > 0)
    assert np.all(np.diff(rescaled2.flatten()) > 0)

@pytest.mark.preprocessing
def test_rescaling_intensity_constant_images():
    """Test rescaling_intensity with constant-valued images."""
    # Create constant-valued images
    img1 = np.ones((3, 3)) * 5
    img2 = np.ones((3, 3)) * 10
    
    # Set max intensity value
    max_intensity = 2.0
    
    # Rescale images
    rescaled1, rescaled2 = rescaling_intensity(img1, img2, max_intensity)
    
    # For constant images, the function should set values to half the max intensity
    expected_value = max_intensity / 2
    assert_allclose(rescaled1, expected_value)
    assert_allclose(rescaled2, expected_value)

@pytest.mark.preprocessing
def test_rescaling_intensity_different_shapes():
    """Test rescaling_intensity with images of different shapes."""
    # Create images with different shapes
    img1 = np.array([[1, 2, 3], [4, 5, 6]])
    img2 = np.array([[10, 20], [30, 40], [50, 60]])
    
    # Set max intensity value
    max_intensity = 1.0
    
    # Rescale images
    rescaled1, rescaled2 = rescaling_intensity(img1, img2, max_intensity)
    
    # Check that output shapes match input shapes
    assert rescaled1.shape == img1.shape
    assert rescaled2.shape == img2.shape
    
    # Check that values are in the range [0, max_intensity]
    assert np.min(rescaled1) >= 0
    assert np.max(rescaled1) <= max_intensity
    assert np.min(rescaled2) >= 0
    assert np.max(rescaled2) <= max_intensity

@pytest.mark.preprocessing
def test_rescaling_intensity_negative_values():
    """Test rescaling_intensity with images containing negative values."""
    # Create images with negative values
    img1 = np.array([[-1, 0, 1], [2, 3, 4]])
    img2 = np.array([[-10, -5, 0], [5, 10, 15]])
    
    # Set max intensity value
    max_intensity = 1.0
    
    # Rescale images
    rescaled1, rescaled2 = rescaling_intensity(img1, img2, max_intensity)
    
    # Check that values are in the range [0, max_intensity]
    assert np.min(rescaled1) >= 0
    assert np.max(rescaled1) <= max_intensity
    assert np.min(rescaled2) >= 0
    assert np.max(rescaled2) <= max_intensity
    
    # Check specific values
    # For img1: range is [-1, 4], so -1 maps to 0 and 4 maps to max_intensity
    assert_allclose(rescaled1[0, 0], 0.0)  # -1 maps to 0
    assert_allclose(rescaled1[1, 2], max_intensity)  # 4 maps to max_intensity
    
    # For img2: range is [-10, 15], so -10 maps to 0 and 15 maps to max_intensity
    assert_allclose(rescaled2[0, 0], 0.0)  # -10 maps to 0
    assert_allclose(rescaled2[1, 2], max_intensity)  # 15 maps to max_intensity

@pytest.mark.preprocessing
def test_rescaling_intensity_preserves_input():
    """Test that rescaling_intensity does not modify the input arrays."""
    # Create test images
    img1 = np.array([[1, 2, 3], [4, 5, 6]])
    img2 = np.array([[10, 20, 30], [40, 50, 60]])
    
    # Make copies for comparison
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    # Rescale images
    rescaling_intensity(img1, img2, 1.0)
    
    # Check that inputs were not modified
    assert_allclose(img1, img1_copy)
    assert_allclose(img2, img2_copy)
