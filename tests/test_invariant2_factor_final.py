import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.invariant2_factor import invariant2_factor as invariant2_factor_open
from openopticalflow.invariant2_factor import invariant2_factor_loop
from openopticalflow.invariant2_factor import invariant2_factor_vectorized
from comparison.openopticalflow.invariant2_factor import invariant2_factor as invariant2_factor_comparison

def test_invariant2_factor_implementations():
    """Test and compare all implementations of invariant2_factor"""
    
    # Create test cases
    test_cases = []
    
    # Test case 1: Rigid body rotation (vortex)
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    vx = -y
    vy = x
    test_cases.append(("Rigid body rotation", vx, vy))
    
    # Test case 2: Shear flow
    vx = y
    vy = np.zeros_like(x)
    test_cases.append(("Shear flow", vx, vy))
    
    # Test case 3: Stagnation point flow
    vx = x
    vy = -y
    test_cases.append(("Stagnation point flow", vx, vy))
    
    # Test case 4: Random flow
    np.random.seed(42)
    vx = np.random.randn(size, size)
    vy = np.random.randn(size, size)
    test_cases.append(("Random flow", vx, vy))
    
    # Set conversion factors
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel
    
    # Run tests
    for name, vx, vy in test_cases:
        print(f"\nTesting {name}:")
        
        # Time the implementations
        start = time()
        qq_open = invariant2_factor_open(vx, vy, factor_x, factor_y)
        time_open = time() - start
        
        start = time()
        qq_loop = invariant2_factor_loop(vx, vy, factor_x, factor_y)
        time_loop = time() - start
        
        start = time()
        qq_vectorized = invariant2_factor_vectorized(vx, vy, factor_x, factor_y)
        time_vectorized = time() - start
        
        start = time()
        qq_comparison = invariant2_factor_comparison(vx, vy, factor_x, factor_y)
        time_comparison = time() - start
        
        # Calculate differences
        diff_open_loop = np.abs(qq_open - qq_loop).max()
        diff_open_vectorized = np.abs(qq_open - qq_vectorized).max()
        diff_loop_vectorized = np.abs(qq_loop - qq_vectorized).max()
        diff_comparison_vectorized = np.abs(qq_comparison - qq_vectorized).max()
        
        # Check if there's a scaling factor between comparison and vectorized
        # Find the ratio between non-zero elements
        mask = (np.abs(qq_vectorized) > 1e-10) & (np.abs(qq_comparison) > 1e-10)
        if np.any(mask):
            ratios = qq_comparison[mask] / qq_vectorized[mask]
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            print(f"Ratio between comparison and vectorized: mean={mean_ratio:.4f}, std={std_ratio:.4f}")
            
            # Check if applying the ratio reduces the difference
            qq_comparison_scaled = qq_comparison / mean_ratio
            diff_scaled = np.abs(qq_comparison_scaled - qq_vectorized).max()
            print(f"Max difference after scaling: {diff_scaled:.8f}")
        
        # Print results
        print(f"Maximum differences:")
        print(f"  Open vs Loop: {diff_open_loop:.8f}")
        print(f"  Open vs Vectorized: {diff_open_vectorized:.8f}")
        print(f"  Loop vs Vectorized: {diff_loop_vectorized:.8f}")
        print(f"  Comparison vs Vectorized: {diff_comparison_vectorized:.8f}")
        
        print(f"Execution times:")
        print(f"  Open implementation: {time_open:.6f} seconds")
        print(f"  Loop implementation: {time_loop:.6f} seconds")
        print(f"  Vectorized implementation: {time_vectorized:.6f} seconds")
        print(f"  Comparison implementation: {time_comparison:.6f} seconds")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Create a 3x2 grid of plots
        plt.subplot(321)
        plt.quiver(x[::3, ::3], y[::3, ::3], vx[::3, ::3], vy[::3, ::3])
        plt.title(f'Velocity Field: {name}')
        plt.axis('equal')
        
        plt.subplot(322)
        plt.imshow(qq_vectorized, cmap='RdBu_r')
        plt.colorbar(label='Q-criterion')
        plt.title('Vectorized Implementation')
        
        plt.subplot(323)
        plt.imshow(qq_loop, cmap='RdBu_r')
        plt.colorbar(label='Q-criterion')
        plt.title('Loop Implementation')
        
        plt.subplot(324)
        plt.imshow(qq_comparison, cmap='RdBu_r')
        plt.colorbar(label='Q-criterion')
        plt.title('Comparison Implementation')
        
        # Plot differences
        plt.subplot(325)
        diff = np.abs(qq_vectorized - qq_comparison)
        plt.imshow(diff, cmap='viridis')
        plt.colorbar(label='Absolute Difference')
        plt.title('Vectorized vs Comparison')
        
        # Plot scaled comparison
        if 'mean_ratio' in locals():
            plt.subplot(326)
            plt.imshow(qq_comparison_scaled, cmap='RdBu_r')
            plt.colorbar(label='Q-criterion')
            plt.title(f'Comparison (Scaled by {mean_ratio:.4f})')
        
        plt.tight_layout()
        plt.savefig(f'results/invariant2_factor_final_{name.replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_invariant2_factor_implementations()
