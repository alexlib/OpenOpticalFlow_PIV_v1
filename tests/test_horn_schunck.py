import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.testing import assert_allclose
from openopticalflow.horn_schunk_estimator import horn_schunk_estimator as hs_open
from comparison.openopticalflow.horn_schunck_estimator import horn_schunck_estimator as hs_comparison

def create_test_case(size=(50, 50), noise_level=0.1):
    """Create synthetic test case with known flow field"""
    # Create random image patterns
    np.random.seed(42)
    I1 = np.random.rand(size[0], size[1])
    
    # Create known flow field (rotating pattern)
    y, x = np.mgrid[-size[0]//2:size[0]//2, -size[1]//2:size[1]//2]
    u_true = -y/size[0]  # horizontal flow
    v_true = x/size[1]   # vertical flow
    
    # Create second image by applying flow
    x_flow = x + u_true
    y_flow = y + v_true
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

def test_horn_schunck():
    """Test both Horn-Schunck implementations"""
    # Test cases with different parameters
    test_cases = [
        {'size': (20, 20), 'alpha': 0.1, 'noise': 0.05},
        {'size': (30, 30), 'alpha': 1.0, 'noise': 0.1},
        {'size': (40, 40), 'alpha': 10.0, 'noise': 0.2}
    ]
    
    for case in test_cases:
        print(f"\nTesting with size={case['size']}, alpha={case['alpha']}, noise={case['noise']}")
        
        # Create test data
        Ix, Iy, It, u_true, v_true = create_test_case(
            size=case['size'], 
            noise_level=case['noise']
        )
        
        # Parameters
        alpha = case['alpha']
        tol = 1e-4
        maxiter = 100
        
        # Run both implementations
        u1, v1 = hs_open(Ix, Iy, It, alpha, tol, maxiter)
        u2, v2 = hs_comparison(Ix, Iy, It, alpha, tol, maxiter)
        
        # Compare results
        max_diff_u = np.max(np.abs(u1 - u2))
        max_diff_v = np.max(np.abs(v1 - v2))
        rmse_u = np.sqrt(np.mean((u1 - u2)**2))
        rmse_v = np.sqrt(np.mean((v1 - v2)**2))
        
        print(f"Maximum difference in u: {max_diff_u:.6f}")
        print(f"Maximum difference in v: {max_diff_v:.6f}")
        print(f"RMSE in u: {rmse_u:.6f}")
        print(f"RMSE in v: {rmse_v:.6f}")
        
        # Compare with ground truth
        rmse_u1_true = np.sqrt(np.mean((u1 - u_true)**2))
        rmse_v1_true = np.sqrt(np.mean((v1 - v_true)**2))
        rmse_u2_true = np.sqrt(np.mean((u2 - u_true)**2))
        rmse_v2_true = np.sqrt(np.mean((v2 - v_true)**2))
        
        print(f"RMSE vs ground truth:")
        print(f"Implementation 1 - u: {rmse_u1_true:.6f}, v: {rmse_v1_true:.6f}")
        print(f"Implementation 2 - u: {rmse_u2_true:.6f}, v: {rmse_v2_true:.6f}")
        
        # Assert results are close enough
        try:
            assert_allclose(u1, u2, rtol=1e-3, atol=1e-3)
            assert_allclose(v1, v2, rtol=1e-3, atol=1e-3)
            print("✓ Implementations produce equivalent results")
        except AssertionError:
            print("✗ Implementations produce different results")

if __name__ == "__main__":
    test_horn_schunck()
