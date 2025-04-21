import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.rescaling_intensity import rescaling_intensity as rescaling_intensity_open
from comparison.openopticalflow.rescaling_intensity import rescaling_intensity as rescaling_intensity_comparison

def test_rescaling_intensity():
    """Test and compare implementations of rescaling_intensity function"""
    
    # Create test cases
    test_cases = []
    
    # Test case 1: Random images with different ranges
    np.random.seed(42)
    img1 = np.random.rand(100, 100) * 100  # Range [0, 100]
    img2 = np.random.rand(100, 100) * 200  # Range [0, 200]
    test_cases.append(("Random images", img1, img2))
    
    # Test case 2: Constant images
    img1 = np.ones((50, 50)) * 10
    img2 = np.ones((50, 50)) * 20
    test_cases.append(("Constant images", img1, img2))
    
    # Test case 3: Images with negative values
    img1 = np.random.rand(50, 50) * 200 - 100  # Range [-100, 100]
    img2 = np.random.rand(50, 50) * 100 - 50   # Range [-50, 50]
    test_cases.append(("Images with negative values", img1, img2))
    
    # Test case 4: Images with different shapes
    img1 = np.random.rand(40, 60) * 100
    img2 = np.random.rand(40, 60) * 150
    test_cases.append(("Different shapes", img1, img2))
    
    # Set max intensity value
    max_intensity = 255
    
    # Run tests
    for name, img1, img2 in test_cases:
        print(f"\nTesting {name}:")
        
        # Apply both implementations
        img1_open, img2_open = rescaling_intensity_open(img1.copy(), img2.copy(), max_intensity)
        img1_comp, img2_comp = rescaling_intensity_comparison(img1.copy(), img2.copy(), max_intensity)
        
        # Calculate differences
        diff1 = np.abs(img1_open - img1_comp).max()
        diff2 = np.abs(img2_open - img2_comp).max()
        
        # Print results
        print(f"Maximum differences:")
        print(f"  Image 1: {diff1:.8f}")
        print(f"  Image 2: {diff2:.8f}")
        
        # Check output types
        print(f"Output types:")
        print(f"  Open: {img1_open.dtype}, {img2_open.dtype}")
        print(f"  Comparison: {img1_comp.dtype}, {img2_comp.dtype}")
        
        # Check output ranges
        print(f"Output ranges:")
        print(f"  Open: [{img1_open.min():.2f}, {img1_open.max():.2f}], [{img2_open.min():.2f}, {img2_open.max():.2f}]")
        print(f"  Comparison: [{img1_comp.min():.2f}, {img1_comp.max():.2f}], [{img2_comp.min():.2f}, {img2_comp.max():.2f}]")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Original images
        plt.subplot(331)
        plt.imshow(img1, cmap='gray')
        plt.colorbar()
        plt.title(f'Original Image 1\nRange: [{img1.min():.2f}, {img1.max():.2f}]')
        
        plt.subplot(332)
        plt.imshow(img2, cmap='gray')
        plt.colorbar()
        plt.title(f'Original Image 2\nRange: [{img2.min():.2f}, {img2.max():.2f}]')
        
        # Open implementation
        plt.subplot(334)
        plt.imshow(img1_open, cmap='gray')
        plt.colorbar()
        plt.title(f'Open Implementation - Image 1\nRange: [{img1_open.min():.2f}, {img1_open.max():.2f}]')
        
        plt.subplot(335)
        plt.imshow(img2_open, cmap='gray')
        plt.colorbar()
        plt.title(f'Open Implementation - Image 2\nRange: [{img2_open.min():.2f}, {img2_open.max():.2f}]')
        
        # Comparison implementation
        plt.subplot(337)
        plt.imshow(img1_comp, cmap='gray')
        plt.colorbar()
        plt.title(f'Comparison Implementation - Image 1\nRange: [{img1_comp.min():.2f}, {img1_comp.max():.2f}]')
        
        plt.subplot(338)
        plt.imshow(img2_comp, cmap='gray')
        plt.colorbar()
        plt.title(f'Comparison Implementation - Image 2\nRange: [{img2_comp.min():.2f}, {img2_comp.max():.2f}]')
        
        # Differences
        plt.subplot(336)
        plt.imshow(np.abs(img1_open - img1_comp), cmap='viridis')
        plt.colorbar()
        plt.title(f'Difference - Image 1\nMax: {diff1:.8f}')
        
        plt.subplot(339)
        plt.imshow(np.abs(img2_open - img2_comp), cmap='viridis')
        plt.colorbar()
        plt.title(f'Difference - Image 2\nMax: {diff2:.8f}')
        
        plt.tight_layout()
        plt.savefig(f'results/rescaling_intensity_{name.replace(" ", "_").lower()}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_rescaling_intensity()
