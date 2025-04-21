"""
Common fixtures and configuration for pytest tests.
"""
import os
import sys
import pytest
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_addoption(parser):
    parser.addoption(
        "--visual", action="store_true", default=False,
        help="Run tests that generate visual outputs"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "visual: mark test as generating visual output")

@pytest.fixture
def create_velocity_field():
    """Create a velocity field for testing vorticity calculations."""
    def _create_field(size=50, field_type="rigid_rotation"):
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))

        if field_type == "rigid_rotation":
            # Rigid body rotation (vortex)
            vx = -y
            vy = x
        elif field_type == "shear":
            # Shear flow
            vx = y
            vy = np.zeros_like(x)
        elif field_type == "stagnation":
            # Stagnation point flow
            vx = x
            vy = -y
        elif field_type == "random":
            # Random flow
            np.random.seed(42)
            vx = np.random.randn(size, size)
            vy = np.random.randn(size, size)
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        return x, y, vx, vy

    return _create_field

@pytest.fixture
def create_test_images():
    """Create synthetic test images with known flow field."""
    def _create_images(size=(50, 50), noise_level=0.1):
        # Create random image patterns
        np.random.seed(42)
        I1 = np.random.rand(size[0], size[1])

        # Create known flow field (rotating pattern)
        y, x = np.mgrid[-size[0]//2:size[0]//2, -size[1]//2:size[1]//2]
        u_true = -y/size[0]  # horizontal flow
        v_true = x/size[1]   # vertical flow

        # Create second image by applying flow
        I2 = np.zeros_like(I1)
        for i in range(size[0]):
            for j in range(size[1]):
                if 0 <= i + v_true[i,j] < size[0] and 0 <= j + u_true[i,j] < size[1]:
                    I2[i,j] = I1[int(i + v_true[i,j]), int(j + u_true[i,j])]

        # Add noise
        I1 += np.random.normal(0, noise_level, I1.shape)
        I2 += np.random.normal(0, noise_level, I2.shape)

        # Calculate derivatives
        Ix = np.zeros_like(I1)
        Iy = np.zeros_like(I1)
        It = I2 - I1

        # Central differences for spatial derivatives
        Ix[:, 1:-1] = (I1[:, 2:] - I1[:, :-2])/2
        Iy[1:-1, :] = (I1[2:, :] - I1[:-2, :])/2

        return Ix, Iy, It, u_true, v_true, I1, I2

    return _create_images

@pytest.fixture
def results_dir():
    """Create and return the results directory."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
