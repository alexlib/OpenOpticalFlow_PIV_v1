import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementation
from openopticalflow.correction_illumination import correction_illumination

def test_correction_illumination():
    """Test and compare both implementations of correction_illumination"""
    
    # Create test cases
    test_cases = []
    
    # Test case 1: Uniform brightness difference
    size = 100
    img1 = np.ones((size, size)) * 100  # Bright image
    img2 = np.ones((size, size)) * 50   # Darker image
    window = [0, size, 0, size]  # Full image
    test_cases.append(("Uniform brightness", img1, img2, window, 0))
    
    # Test case 2: Gradient brightness
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    img1 = 100 * (x + y)  # Gradient image
    img2 = 50 * (x + y)   # Darker gradient
    test_cases.append(("Gradient brightness", img1, img2, window, 0))
    
    # Test case 3: Local illumination changes
    img1 = 100 * np.ones((size, size))
    img2 = 100 * np.ones((size, size))
    # Add a bright spot to img2
    center = size // 2
    radius = size // 10
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2:
                img2[i, j] = 150
    test_cases.append(("Local illumination", img1, img2, window, 10))
    
    # Test case 4: Random noise
    np.random.seed(42)
    img1 = 100 + 10 * np.random.randn(size, size)
    img2 = 50 + 10 * np.random.randn(size, size)
    test_cases.append(("Random noise", img1, img2, window, 5))
    
    # Run tests
    for name, img1, img2, window, size_avg in test_cases:
        print(f"\nTesting {name}:")
        
        # Time the implementation
        start = time()
        i1_open, i2_open = correction_illumination(img1, img2, window, size_avg)
        time_open = time() - start
        
        # Print results
        print(f"Execution time:")
        print(f"  Open implementation: {time_open:.6f} seconds")
        
        # Calculate statistics
        i1_open_mean = np.mean(i1_open)
        i2_open_mean = np.mean(i2_open)
        
        print(f"Mean values:")
        print(f"  Open: i1={i1_open_mean:.2f}, i2={i2_open_mean:.2f}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Original images
        plt.subplot(331)
        plt.imshow(img1, cmap='gray')
        plt.colorbar()
        plt.title('Original Image 1')
        
        plt.subplot(332)
        plt.imshow(img2, cmap='gray')
        plt.colorbar()
        plt.title('Original Image 2')
        
        plt.subplot(333)
        plt.imshow(img1 - img2, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Original Difference')
        
        # Open implementation
        plt.subplot(334)
        plt.imshow(i1_open, cmap='gray')
        plt.colorbar()
        plt.title('Open Implementation - Image 1')
        
        plt.subplot(335)
        plt.imshow(i2_open, cmap='gray')
        plt.colorbar()
        plt.title('Open Implementation - Image 2')
        
        plt.subplot(336)
        plt.imshow(i1_open - i2_open, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Open Implementation - Difference')
        
        plt.tight_layout()
        plt.savefig(f'results/correction_illumination_{name.replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_correction_illumination()
