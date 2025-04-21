import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.horn_schunk_estimator import horn_schunk_estimator as horn_schunck_open
from comparison.openopticalflow.horn_schunck_estimator import horn_schunck_estimator as horn_schunck_comparison

def create_test_case(size=(50, 50), noise_level=0.1):
    """Create synthetic test case with known flow field"""
    # Create random image patterns
    np.random.seed(42)
    img1 = np.random.rand(size[0], size[1])
    
    # Create known flow field (rotating pattern)
    y, x = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
    center_y, center_x = size[0] // 2, size[1] // 2
    u_true = -(y - center_y) / 10  # horizontal flow
    v_true = (x - center_x) / 10   # vertical flow
    
    # Create second image by applying flow
    img2 = np.zeros_like(img1)
    for i in range(size[0]):
        for j in range(size[1]):
            # Get source coordinates
            src_i = int(i - v_true[i, j])
            src_j = int(j - u_true[i, j])
            
            # Check if source is within bounds
            if 0 <= src_i < size[0] and 0 <= src_j < size[1]:
                img2[i, j] = img1[src_i, src_j]
    
    # Add noise
    img1 += np.random.normal(0, noise_level, img1.shape)
    img2 += np.random.normal(0, noise_level, img2.shape)
    
    # Clip to valid range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # Calculate derivatives
    Ix = np.zeros_like(img1)
    Iy = np.zeros_like(img1)
    It = np.zeros_like(img1)
    
    # Simple central differences for interior points
    Ix[1:-1, 1:-1] = (img1[1:-1, 2:] - img1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (img1[2:, 1:-1] - img1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = img2[1:-1, 1:-1] - img1[1:-1, 1:-1]
    
    return Ix, Iy, It, u_true, v_true

def test_horn_schunck_estimator():
    """Test and compare implementations of Horn-Schunck estimator"""
    
    # Create test cases
    test_cases = []
    
    # Test case 1: Small image, low noise
    size = (30, 30)
    noise = 0.05
    Ix, Iy, It, u_true, v_true = create_test_case(size, noise)
    test_cases.append(("Small image", Ix, Iy, It, u_true, v_true))
    
    # Test case 2: Larger image
    size = (50, 50)
    noise = 0.05
    Ix, Iy, It, u_true, v_true = create_test_case(size, noise)
    test_cases.append(("Larger image", Ix, Iy, It, u_true, v_true))
    
    # Parameters for Horn-Schunck
    lambda_values = [0.1, 1.0, 10.0]
    tol = 1e-4
    maxiter = 100
    
    # Run tests
    for name, Ix, Iy, It, u_true, v_true in test_cases:
        print(f"\nTesting {name}:")
        
        for lambda_ in lambda_values:
            print(f"\n  Lambda = {lambda_}:")
            
            # Time the open implementation
            start = time()
            u_open, v_open = horn_schunck_open(Ix, Iy, It, lambda_, tol, maxiter)
            time_open = time() - start
            
            # Time the comparison implementation
            start = time()
            u_comp, v_comp = horn_schunck_comparison(Ix, Iy, It, lambda_, tol, maxiter)
            time_comp = time() - start
            
            # Calculate differences between implementations
            u_diff = np.abs(u_open - u_comp).max()
            v_diff = np.abs(v_open - v_comp).max()
            
            # Calculate error metrics against ground truth
            u_error_open = np.sqrt(np.mean((u_open - u_true)**2))
            v_error_open = np.sqrt(np.mean((v_open - v_true)**2))
            u_error_comp = np.sqrt(np.mean((u_comp - u_true)**2))
            v_error_comp = np.sqrt(np.mean((v_comp - v_true)**2))
            
            # Print results
            print(f"    Maximum differences between implementations:")
            print(f"      u: {u_diff:.8f}")
            print(f"      v: {v_diff:.8f}")
            
            print(f"    RMSE against ground truth:")
            print(f"      Open: u = {u_error_open:.6f}, v = {v_error_open:.6f}")
            print(f"      Comparison: u = {u_error_comp:.6f}, v = {v_error_comp:.6f}")
            
            print(f"    Execution times:")
            print(f"      Open implementation: {time_open:.6f} seconds")
            print(f"      Comparison implementation: {time_comp:.6f} seconds")
            
            # Plot results
            plt.figure(figsize=(15, 10))
            
            # True flow
            plt.subplot(231)
            plt.quiver(u_true, v_true)
            plt.title('True Flow')
            plt.axis('equal')
            
            # Open implementation
            plt.subplot(232)
            plt.quiver(u_open, v_open)
            plt.title(f'Open Implementation\nRMSE: {np.sqrt(u_error_open**2 + v_error_open**2):.6f}')
            plt.axis('equal')
            
            # Comparison implementation
            plt.subplot(233)
            plt.quiver(u_comp, v_comp)
            plt.title(f'Comparison Implementation\nRMSE: {np.sqrt(u_error_comp**2 + v_error_comp**2):.6f}')
            plt.axis('equal')
            
            # Error maps
            plt.subplot(234)
            plt.imshow(np.sqrt((u_true - u_open)**2 + (v_true - v_open)**2), cmap='viridis')
            plt.colorbar()
            plt.title('Error Map (Open)')
            
            plt.subplot(235)
            plt.imshow(np.sqrt((u_true - u_comp)**2 + (v_true - v_comp)**2), cmap='viridis')
            plt.colorbar()
            plt.title('Error Map (Comparison)')
            
            # Difference between implementations
            plt.subplot(236)
            plt.imshow(np.sqrt((u_open - u_comp)**2 + (v_open - v_comp)**2), cmap='viridis')
            plt.colorbar()
            plt.title(f'Implementation Difference\nMax: {max(u_diff, v_diff):.8f}')
            
            plt.tight_layout()
            plt.savefig(f'results/horn_schunck_{name.replace(" ", "_").lower()}_lambda_{lambda_}.png')
            plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_horn_schunck_estimator()
