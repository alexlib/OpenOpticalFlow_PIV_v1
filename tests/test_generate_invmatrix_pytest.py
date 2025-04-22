import numpy as np
import pytest
from numpy.testing import assert_allclose

from openopticalflow.generate_invmatrix import generate_invmatrix

@pytest.mark.liu_shen
def test_generate_invmatrix_basic():
    """Test basic functionality of generate_invmatrix."""
    # Create a simple test image
    size = 20
    i = np.ones((size, size))

    # Set parameters
    alpha = 0.1
    h = 1.0

    # Generate inverse matrix
    b11, b12, b22 = generate_invmatrix(i, alpha, h)

    # Check output shapes
    assert b11.shape == i.shape
    assert b12.shape == i.shape
    assert b22.shape == i.shape

    # For a constant image, b11 and b22 should be equal and b12 should be zero
    assert_allclose(b11, b22)
    assert_allclose(b12, 0.0)

@pytest.mark.liu_shen
def test_generate_invmatrix_gradient():
    """Test generate_invmatrix with a gradient image."""
    # Create a gradient test image
    size = 20
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    i = x + y  # Linear gradient

    # Set parameters
    alpha = 0.1
    h = 1.0

    # Generate inverse matrix
    b11, b12, b22 = generate_invmatrix(i, alpha, h)

    # Check output shapes
    assert b11.shape == i.shape
    assert b12.shape == i.shape
    assert b22.shape == i.shape

    # For a gradient image, the determinant should be non-zero
    # and the inverse matrix should be well-defined
    det = b11 * b22 - b12 * b12
    assert np.all(det > 0)

@pytest.mark.liu_shen
def test_generate_invmatrix_zero_image():
    """Test generate_invmatrix with a zero image."""
    # Create a zero image
    size = 20
    i = np.zeros((size, size))

    # Set parameters
    alpha = 0.1
    h = 1.0

    # Generate inverse matrix
    b11, b12, b22 = generate_invmatrix(i, alpha, h)

    # Check output shapes
    assert b11.shape == i.shape
    assert b12.shape == i.shape
    assert b22.shape == i.shape

    # For a zero image, the determinant is invalid everywhere
    # The function should handle this case by setting b11 and b22 to a consistent value
    # and b12 to zero
    assert_allclose(b11, b22)  # b11 and b22 should be equal
    assert_allclose(b12, 0.0)  # b12 should be zero

@pytest.mark.liu_shen
def test_generate_invmatrix_alpha_effect():
    """Test the effect of alpha parameter on generate_invmatrix."""
    # Create a test image
    size = 20
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    i = np.exp(-(x**2 + y**2) / 0.2)  # Gaussian

    # Set parameters
    h = 1.0
    alpha1 = 0.1
    alpha2 = 1.0  # Larger alpha

    # Generate inverse matrices with different alpha values
    b11_1, _, b22_1 = generate_invmatrix(i, alpha1, h)
    b11_2, _, b22_2 = generate_invmatrix(i, alpha2, h)

    # With larger alpha, the regularization term has more influence
    # This should result in different inverse matrix components
    assert not np.allclose(b11_1, b11_2)
    assert not np.allclose(b22_1, b22_2)

@pytest.mark.liu_shen
def test_generate_invmatrix_h_effect():
    """Test the effect of h parameter on generate_invmatrix."""
    # Create a test image
    size = 20
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    i = np.exp(-(x**2 + y**2) / 0.2)  # Gaussian

    # Set parameters
    alpha = 0.1
    h1 = 1.0
    h2 = 0.5  # Smaller h (finer grid)

    # Generate inverse matrices with different h values
    b11_1, _, b22_1 = generate_invmatrix(i, alpha, h1)
    b11_2, _, b22_2 = generate_invmatrix(i, alpha, h2)

    # With different h values, the derivatives are scaled differently
    # This should result in different inverse matrix components
    assert not np.allclose(b11_1, b11_2)
    assert not np.allclose(b22_1, b22_2)

@pytest.mark.liu_shen
def test_generate_invmatrix_analytical():
    """Test generate_invmatrix against an analytical solution."""
    # For a simple case where we can calculate the expected result
    # Create a constant image with a single non-zero pixel
    size = 5
    i = np.zeros((size, size))
    i[2, 2] = 1.0  # Single non-zero pixel at the center

    # Set parameters
    alpha = 0.0  # No regularization for simplicity
    h = 1.0

    # Generate inverse matrix
    b11, b12, b22 = generate_invmatrix(i, alpha, h)

    # For this specific case, we expect:
    # - Non-zero values only at and around the center pixel
    # - Symmetry between b11 and b22 due to the symmetric pattern
    # - b12 should be zero at the center due to symmetry

    # Check symmetry
    assert_allclose(b11[2, 2], b22[2, 2], rtol=1e-5)
    assert_allclose(b12[2, 2], 0.0, atol=1e-5)

    # Check that values away from the center are close to zero or handled specially
    assert np.all(b11[0, :] == b11[0, 0])
    assert np.all(b22[0, :] == b22[0, 0])
    assert_allclose(b12[0, :], 0.0)
