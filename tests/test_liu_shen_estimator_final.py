import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.liu_shen_estimator import liu_shen_estimator as liu_shen_open
# Only use the main implementation
from openopticalflow.liu_shen_estimator import liu_shen_estimator

def create_test_case(size=(50, 50), noise_level=0.1):
    """Create synthetic test case with known flow field"""
    # Create random image patterns
    np.random.seed(42)
    I0 = np.random.rand(size[0], size[1])

    # Create known flow field (rotating pattern)
    y, x = np.mgrid[-size[0]//2:size[0]//2, -size[1]//2:size[1]//2]
    u_true = -y/size[0]  # horizontal flow
    v_true = x/size[1]   # vertical flow

    # Create second image by applying flow
    x_flow = x + u_true
    y_flow = y + v_true
    I1 = np.zeros_like(I0)
    for i in range(size[0]):
        for j in range(size[1]):
            if 0 <= i + v_true[i,j] < size[0] and 0 <= j + u_true[i,j] < size[1]:
                I1[i,j] = I0[int(i + v_true[i,j]), int(j + u_true[i,j])]

    # Add noise
    I0 += np.random.normal(0, noise_level, I0.shape)
    I1 += np.random.normal(0, noise_level, I1.shape)

    return I0, I1, u_true, v_true

def test_liu_shen_estimator_final():
    """Final test for Liu-Shen estimator implementations"""

    # Create test case
    size = (30, 30)
    noise = 0.05
    I0, I1, u_true, v_true = create_test_case(size, noise)

    # Parameters
    dx = 1
    dt = 1
    lambda_param = 0.1
    tol = 1e-4
    maxnum = 20
    f = np.zeros_like(I0)  # No physical transport term
    u0 = np.zeros_like(I0)  # Initial guess
    v0 = np.zeros_like(I0)

    print("Testing Liu-Shen estimator implementations:")

    # Time the open implementation
    start = time()
    u_open, v_open, error_open = liu_shen_open(I0, I1, f, dx, dt, lambda_param, tol, maxnum, u0, v0)
    time_open = time() - start

    # No comparison implementation anymore

    # Calculate error against ground truth
    u_error_open = np.sqrt(np.mean((u_open - u_true)**2))
    v_error_open = np.sqrt(np.mean((v_open - v_true)**2))

    # Print results
    print(f"RMSE against ground truth:")
    print(f"  Open: u={u_error_open:.6f}, v={v_error_open:.6f}")

    print(f"Execution times:")
    print(f"  Open implementation: {time_open:.6f} seconds")

    print(f"Convergence:")
    print(f"  Open: {len(error_open)} iterations, final error={error_open[-1]:.8f}")

    # Plot results
    plt.figure(figsize=(15, 10))

    # Original images and ground truth
    plt.subplot(331)
    plt.imshow(I0, cmap='gray')
    plt.colorbar()
    plt.title('First Image')

    plt.subplot(332)
    plt.imshow(I1, cmap='gray')
    plt.colorbar()
    plt.title('Second Image')

    plt.subplot(333)
    X, Y = np.meshgrid(np.arange(0, size[1], 2), np.arange(0, size[0], 2))
    plt.quiver(X, Y, u_true[::2, ::2], v_true[::2, ::2])
    plt.title('Ground Truth Flow')
    plt.axis('equal')

    # Flow fields
    plt.subplot(334)
    plt.imshow(u_open, cmap='RdBu_r')
    plt.colorbar()
    plt.title('u (Open)')

    plt.subplot(335)
    plt.imshow(v_open, cmap='RdBu_r')
    plt.colorbar()
    plt.title('v (Open)')

    plt.subplot(336)
    plt.quiver(X, Y, u_open[::2, ::2], v_open[::2, ::2])
    plt.title('Flow (Open)')
    plt.axis('equal')

    # Differences
    plt.subplot(337)
    plt.imshow(np.abs(u_open - u_true), cmap='viridis')
    plt.colorbar()
    plt.title('u Error vs Ground Truth')

    plt.subplot(338)
    plt.imshow(np.abs(v_open - v_true), cmap='viridis')
    plt.colorbar()
    plt.title('v Error vs Ground Truth')

    # Convergence
    plt.subplot(339)
    plt.semilogy(error_open, 'b-', label='Implementation')
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Convergence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/liu_shen_final.png')
    plt.close()

    print("Test completed. See results/liu_shen_final.png for visualization.")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run test
    test_liu_shen_estimator_final()
