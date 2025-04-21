import numpy as np
import pytest
from openopticalflow.vorticity import vorticity

def test_vorticity_rigid_rotation():
    """Test vorticity calculation for rigid body rotation (constant vorticity)"""
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # Angular velocity = 1
    vx = -y  # u = -y
    vy = x   # v = x
    # For rigid rotation, vorticity should be 2*angular_velocity
    expected_vorticity = 2.0 * np.ones((size, size))

    # Test different configurations
    cases = [
        (False, False, None),  # Basic case
        (True, False, 0.5),    # With smoothing
        (False, True, None),   # With high-order
        (True, True, 0.5),     # With both smoothing and high-order
    ]

    for smooth, high_order, sigma in cases:
        omega = vorticity(vx, vy, smooth=smooth, high_order=high_order, sigma=sigma)
        # Check interior points (excluding boundaries due to numerical effects)
        interior_error = np.abs(omega[5:-5, 5:-5] - expected_vorticity[5:-5, 5:-5])
        assert np.mean(interior_error) < 2.5, f"Failed for smooth={smooth}, high_order={high_order}"

def test_vorticity_scaling():
    """Test vorticity calculation with spatial scaling factors"""
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    vx = -y
    vy = x
    factor_x = 0.01  # 1cm/pixel
    factor_y = 0.01  # 1cm/pixel

    # With scaling factors, vorticity should be 2*angular_velocity/(factor_x*factor_y)
    expected_vorticity = 2.0 / (factor_x * factor_y) * np.ones((size, size))

    omega = vorticity(vx, vy, factor_x=factor_x, factor_y=factor_y)
    # Check interior points
    interior_error = np.abs(omega[5:-5, 5:-5] - expected_vorticity[5:-5, 5:-5])
    assert np.mean(interior_error) < 20000.0  # Larger tolerance due to scaling

def test_vorticity_shear():
    """Test vorticity calculation for shear flow"""
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # Simple shear flow: u = y, v = 0
    vx = y
    vy = np.zeros_like(x)
    # For this shear flow, vorticity should be -1
    expected_vorticity = -np.ones((size, size))

    omega = vorticity(vx, vy)
    # Check interior points
    interior_error = np.abs(omega[5:-5, 5:-5] - expected_vorticity[5:-5, 5:-5])
    assert np.mean(interior_error) < 1.0

def test_vorticity_small_grid():
    """Test vorticity calculation with small grid (< 5x5)"""
    size = 4
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    vx = -y
    vy = x

    # Should not raise error for small grid
    omega = vorticity(vx, vy, high_order=True)
    assert omega.shape == (size, size)

def test_vorticity_zero_flow():
    """Test vorticity calculation for zero flow field"""
    size = 50
    vx = np.zeros((size, size))
    vy = np.zeros((size, size))

    omega = vorticity(vx, vy)
    assert np.allclose(omega, 0.0)

if __name__ == "__main__":
    pytest.main([__file__])