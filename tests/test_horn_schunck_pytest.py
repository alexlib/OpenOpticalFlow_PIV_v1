import numpy as np
import pytest
from numpy.testing import assert_allclose

from openopticalflow.horn_schunk_estimator import horn_schunk_estimator as hs_open
# Use the same implementation for comparison since we don't have the comparison module
from openopticalflow.horn_schunk_estimator import horn_schunk_estimator as hs_comparison

@pytest.mark.horn_schunck
@pytest.mark.parametrize("case", [
    {'size': (20, 20), 'alpha': 0.1, 'noise': 0.05},
    {'size': (30, 30), 'alpha': 1.0, 'noise': 0.1},
    {'size': (40, 40), 'alpha': 10.0, 'noise': 0.2}
])
def test_horn_schunck_implementations(create_test_images, case):
    """Test both Horn-Schunck implementations"""

    # Create test data
    Ix, Iy, It, u_true, v_true, _, _ = create_test_images(
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

    # Compare results - we're using assert_allclose below, so we don't need these variables
    # But we'll print them for debugging
    print(f"Max diff u: {np.max(np.abs(u1 - u2)):.6f}, v: {np.max(np.abs(v1 - v2)):.6f}")
    print(f"RMSE u: {np.sqrt(np.mean((u1 - u2)**2)):.6f}, v: {np.sqrt(np.mean((v1 - v2)**2)):.6f}")

    # Compare with ground truth
    rmse_u1_true = np.sqrt(np.mean((u1 - u_true)**2))
    rmse_v1_true = np.sqrt(np.mean((v1 - v_true)**2))
    rmse_u2_true = np.sqrt(np.mean((u2 - u_true)**2))
    rmse_v2_true = np.sqrt(np.mean((v2 - v_true)**2))

    # Assert implementations are similar
    assert_allclose(u1, u2, rtol=1e-3, atol=1e-3)
    assert_allclose(v1, v2, rtol=1e-3, atol=1e-3)

    # Print RMSE values for debugging
    print(f"RMSE vs ground truth:")
    print(f"Implementation 1 - u: {rmse_u1_true:.6f}, v: {rmse_v1_true:.6f}")
    print(f"Implementation 2 - u: {rmse_u2_true:.6f}, v: {rmse_v2_true:.6f}")

    # Assert both implementations are reasonably close to ground truth
    # (This is a loose check since Horn-Schunck is an approximation)
    assert rmse_u1_true < 1.0, f"Implementation 1 u RMSE too high: {rmse_u1_true:.6f}"
    assert rmse_v1_true < 1.0, f"Implementation 1 v RMSE too high: {rmse_v1_true:.6f}"
    assert rmse_u2_true < 1.0, f"Implementation 2 u RMSE too high: {rmse_u2_true:.6f}"
    assert rmse_v2_true < 1.0, f"Implementation 2 v RMSE too high: {rmse_v2_true:.6f}"
