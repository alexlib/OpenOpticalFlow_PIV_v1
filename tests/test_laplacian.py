import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.laplacian import laplacian as laplacian_open
from comparison.openopticalflow.laplacian import laplacian as laplacian_comparison

def test_laplacian():
    """Test and compare implementations of laplacian function"""
    
    # Create test cases
    test_cases = []
    
    # Test case 1: Constant image
    size = 50
    img = np.ones((size, size))
    test_cases.append(("Constant", img))
    
    # Test case 2: Gradient image
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    img = x + y
    test_cases.append(("Gradient", img))
    
    # Test case 3: Gaussian bump
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    img = np.exp(-(x**2 + y**2))
    test_cases.append(("Gaussian", img))
    
    # Test case 4: Checkerboard pattern
    img = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            img[i, j] = (i + j) % 2
    test_cases.append(("Checkerboard", img))
    
    # Set step size
    h = 1
    
    # Run tests
    for name, img in test_cases:
        print(f"\nTesting {name}:")
        
        # Time the implementations
        start = time()
        lap_open = laplacian_open(img, h)
        time_open = time() - start
        
        start = time()
        lap_comp = laplacian_comparison(img, h)
        time_comp = time() - start
        
        # Calculate differences
        diff = np.abs(lap_open - lap_comp).max()
        
        # Print results
        print(f"Maximum difference: {diff:.8f}")
        print(f"Execution times:")
        print(f"  Open implementation: {time_open:.6f} seconds")
        print(f"  Comparison implementation: {time_comp:.6f} seconds")
        
        # Calculate statistics
        lap_open_mean = np.mean(lap_open)
        lap_comp_mean = np.mean(lap_comp)
        
        print(f"Mean values:")
        print(f"  Open: {lap_open_mean:.6f}")
        print(f"  Comparison: {lap_comp_mean:.6f}")
        
        # Plot results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.title(f'Original Image: {name}')
        
        plt.subplot(142)
        plt.imshow(lap_open, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Laplacian (Open)')
        
        plt.subplot(143)
        plt.imshow(lap_comp, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Laplacian (Comparison)')
        
        plt.subplot(144)
        plt.imshow(np.abs(lap_open - lap_comp), cmap='viridis')
        plt.colorbar()
        plt.title('Absolute Difference')
        
        plt.tight_layout()
        plt.savefig(f'results/laplacian_{name.lower()}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_laplacian()
