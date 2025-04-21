import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.vorticity_factor import vorticity_factor as vorticity_open
from comparison.openopticalflow.vorticity_factor import vorticity as vorticity_comparison

def test_vorticity_factor():
    """Test and compare implementations of vorticity calculation"""
    
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
        vort_open = vorticity_open(vx, vy, factor_x, factor_y)
        time_open = time() - start
        
        start = time()
        vort_comp = vorticity_comparison(vx, vy, factor_x, factor_y)
        time_comp = time() - start
        
        # Calculate differences
        diff = np.abs(vort_open - vort_comp).max()
        
        # Print results
        print(f"Maximum difference: {diff:.8f}")
        print(f"Execution times:")
        print(f"  Open implementation: {time_open:.6f} seconds")
        print(f"  Comparison implementation: {time_comp:.6f} seconds")
        
        # Calculate statistics
        vort_open_mean = np.mean(vort_open)
        vort_comp_mean = np.mean(vort_comp)
        
        print(f"Mean values:")
        print(f"  Open: {vort_open_mean:.6f}")
        print(f"  Comparison: {vort_comp_mean:.6f}")
        
        # Calculate theoretical vorticity for rigid body rotation
        if name == "Rigid body rotation":
            theoretical = 2 * np.ones_like(vx)  # Vorticity is 2 for rigid body rotation
            error_open = np.abs(vort_open - theoretical).mean()
            error_comp = np.abs(vort_comp - theoretical).mean()
            print(f"Mean error from theoretical value:")
            print(f"  Open: {error_open:.6f}")
            print(f"  Comparison: {error_comp:.6f}")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.quiver(x[::3, ::3], y[::3, ::3], vx[::3, ::3], vy[::3, ::3])
        plt.title(f'Velocity Field: {name}')
        plt.axis('equal')
        
        plt.subplot(142)
        plt.imshow(vort_open, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Vorticity (Open)')
        
        plt.subplot(143)
        plt.imshow(vort_comp, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Vorticity (Comparison)')
        
        plt.subplot(144)
        plt.imshow(np.abs(vort_open - vort_comp), cmap='viridis')
        plt.colorbar()
        plt.title('Absolute Difference')
        
        plt.tight_layout()
        plt.savefig(f'results/vorticity_{name.replace(" ", "_").lower()}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_vorticity_factor()
