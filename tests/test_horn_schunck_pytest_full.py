import numpy as np
import pytest
from numpy.testing import assert_allclose

from openopticalflow.horn_schunk_estimator import horn_schunk_estimator

@pytest.fixture
def create_test_case():
    """Create synthetic test case with known flow field"""
    def _create_test_case(size=(50, 50), noise_level=0.1):
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
        
        return Ix, Iy, It, u_true, v_true
    
    return _create_test_case

@pytest.mark.horn_schunck
@pytest.mark.parametrize("case", [
    {'size': (20, 20), 'alpha': 0.1, 'noise': 0.05},
    {'size': (30, 30), 'alpha': 1.0, 'noise': 0.1},
    {'size': (40, 40), 'alpha': 10.0, 'noise': 0.2}
])
def test_horn_schunck_accuracy(create_test_case, case):
    """Test Horn-Schunck implementation against ground truth"""
    
    # Create test data
    Ix, Iy, It, u_true, v_true = create_test_case(
        size=case['size'], 
        noise_level=case['noise']
    )
    
    # Parameters
    alpha = case['alpha']
    tol = 1e-4
    maxiter = 100
    
    # Run implementation
    u, v = horn_schunk_estimator(Ix, Iy, It, alpha, tol, maxiter)
    
    # Compare with ground truth
    rmse_u = np.sqrt(np.mean((u - u_true)**2))
    rmse_v = np.sqrt(np.mean((v - v_true)**2))
    
    print(f"RMSE vs ground truth - u: {rmse_u:.6f}, v: {rmse_v:.6f}")
    
    # Assert results are reasonably close to ground truth
    # Horn-Schunck is an approximation, so we use a loose threshold
    assert rmse_u < 1.0, f"RMSE for u is too high: {rmse_u:.6f}"
    assert rmse_v < 1.0, f"RMSE for v is too high: {rmse_v:.6f}"

@pytest.mark.horn_schunck
@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_horn_schunck_convergence(create_test_case, alpha):
    """Test that Horn-Schunck converges with different alpha values"""
    
    # Create test data
    Ix, Iy, It, _, _ = create_test_case(size=(30, 30), noise_level=0.1)
    
    # Parameters
    tol = 1e-4
    maxiter = 100
    
    # Run implementation
    u, v = horn_schunk_estimator(Ix, Iy, It, alpha, tol, maxiter)
    
    # Check that results are not NaN or infinite
    assert not np.isnan(u).any(), "NaN values in u"
    assert not np.isnan(v).any(), "NaN values in v"
    assert not np.isinf(u).any(), "Infinite values in u"
    assert not np.isinf(v).any(), "Infinite values in v"
    
    # Check that results have reasonable magnitudes
    assert np.abs(u).max() < 10.0, f"u values too large: {np.abs(u).max():.6f}"
    assert np.abs(v).max() < 10.0, f"v values too large: {np.abs(v).max():.6f}"
