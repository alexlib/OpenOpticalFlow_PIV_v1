import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1 as shift_image_open
from comparison.openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1 as shift_image_comparison

def create_test_case(size=(50, 50), noise_level=0.1):
    """Create synthetic test case with known flow field"""
    # Create random image patterns
    np.random.seed(42)
    img1 = np.random.rand(size[0], size[1]) * 255
    
    # Create known flow field (rotating pattern)
    y, x = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
    center_y, center_x = size[0] // 2, size[1] // 2
    u = -(y - center_y) / 10  # horizontal flow
    v = (x - center_x) / 10   # vertical flow
    
    # Create second image by applying flow
    img2 = np.zeros_like(img1)
    for i in range(size[0]):
        for j in range(size[1]):
            # Get source coordinates
            src_i = int(i - v[i, j])
            src_j = int(j - u[i, j])
            
            # Check if source is within bounds
            if 0 <= src_i < size[0] and 0 <= src_j < size[1]:
                img2[i, j] = img1[src_i, src_j]
    
    # Add noise
    img1 += np.random.normal(0, noise_level * 255, img1.shape)
    img2 += np.random.normal(0, noise_level * 255, img2.shape)
    
    # Clip to valid range
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    
    return img1, img2, u, v

def test_shift_image_fun_refine_1():
    """Test and compare implementations of shift_image_fun_refine_1 function"""
    
    # Create test cases
    test_cases = []
    
    # Test case 1: Small image, low noise
    size = (30, 30)
    noise = 0.05
    img1, img2, u, v = create_test_case(size, noise)
    test_cases.append(("Small image", img1, img2, u, v))
    
    # Test case 2: Larger image
    size = (50, 50)
    noise = 0.05
    img1, img2, u, v = create_test_case(size, noise)
    test_cases.append(("Larger image", img1, img2, u, v))
    
    # Run tests
    for name, img1, img2, u, v in test_cases:
        print(f"\nTesting {name}:")
        
        # Convert to uint8 for consistency
        img1_uint8 = img1.astype(np.uint8)
        img2_uint8 = img2.astype(np.uint8)
        
        # Time the open implementation
        start = time()
        try:
            img1_shift_open, ux_open, uy_open = shift_image_open(u, v, img1_uint8, img2_uint8)
            time_open = time() - start
            open_success = True
        except Exception as e:
            print(f"Open implementation failed: {e}")
            time_open = time() - start
            open_success = False
        
        # Time the comparison implementation
        start = time()
        try:
            img1_shift_comp, ux_comp, uy_comp = shift_image_comparison(u, v, img1_uint8, img2_uint8)
            time_comp = time() - start
            comp_success = True
        except Exception as e:
            print(f"Comparison implementation failed: {e}")
            time_comp = time() - start
            comp_success = False
        
        # Print timing results
        print(f"Execution times:")
        print(f"  Open implementation: {time_open:.6f} seconds")
        print(f"  Comparison implementation: {time_comp:.6f} seconds")
        
        # If both implementations succeeded, compare results
        if open_success and comp_success:
            # Calculate differences
            img_diff = np.abs(img1_shift_open.astype(float) - img1_shift_comp.astype(float)).max()
            ux_diff = np.abs(ux_open - ux_comp).max()
            uy_diff = np.abs(uy_open - uy_comp).max()
            
            # Print differences
            print(f"Maximum differences:")
            print(f"  Shifted image: {img_diff:.2f}")
            print(f"  ux: {ux_diff:.6f}")
            print(f"  uy: {uy_diff:.6f}")
            
            # Calculate similarity metrics
            if np.std(img1_shift_open) > 0 and np.std(img1_shift_comp) > 0:
                correlation = np.corrcoef(img1_shift_open.flatten(), img1_shift_comp.flatten())[0, 1]
                print(f"Image correlation: {correlation:.6f}")
            
            # Plot results
            plt.figure(figsize=(15, 10))
            
            # Original images
            plt.subplot(331)
            plt.imshow(img1_uint8, cmap='gray')
            plt.colorbar()
            plt.title('Original Image 1')
            
            plt.subplot(332)
            plt.imshow(img2_uint8, cmap='gray')
            plt.colorbar()
            plt.title('Original Image 2')
            
            # Flow field
            plt.subplot(333)
            plt.quiver(u[::2, ::2], v[::2, ::2])
            plt.title('Flow Field')
            plt.axis('equal')
            
            # Open implementation
            plt.subplot(334)
            plt.imshow(img1_shift_open, cmap='gray')
            plt.colorbar()
            plt.title('Shifted Image (Open)')
            
            plt.subplot(335)
            plt.imshow(ux_open, cmap='RdBu_r')
            plt.colorbar()
            plt.title('ux (Open)')
            
            plt.subplot(336)
            plt.imshow(uy_open, cmap='RdBu_r')
            plt.colorbar()
            plt.title('uy (Open)')
            
            # Comparison implementation
            plt.subplot(337)
            plt.imshow(img1_shift_comp, cmap='gray')
            plt.colorbar()
            plt.title('Shifted Image (Comparison)')
            
            plt.subplot(338)
            plt.imshow(ux_comp, cmap='RdBu_r')
            plt.colorbar()
            plt.title('ux (Comparison)')
            
            plt.subplot(339)
            plt.imshow(uy_comp, cmap='RdBu_r')
            plt.colorbar()
            plt.title('uy (Comparison)')
            
            plt.tight_layout()
            plt.savefig(f'results/shift_image_{name.replace(" ", "_").lower()}.png')
            plt.close()
            
            # Plot differences
            plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(np.abs(img1_shift_open.astype(float) - img1_shift_comp.astype(float)), cmap='viridis')
            plt.colorbar()
            plt.title(f'Image Difference\nMax: {img_diff:.2f}')
            
            plt.subplot(132)
            plt.imshow(np.abs(ux_open - ux_comp), cmap='viridis')
            plt.colorbar()
            plt.title(f'ux Difference\nMax: {ux_diff:.6f}')
            
            plt.subplot(133)
            plt.imshow(np.abs(uy_open - uy_comp), cmap='viridis')
            plt.colorbar()
            plt.title(f'uy Difference\nMax: {uy_diff:.6f}')
            
            plt.tight_layout()
            plt.savefig(f'results/shift_image_{name.replace(" ", "_").lower()}_diff.png')
            plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_shift_image_fun_refine_1()
