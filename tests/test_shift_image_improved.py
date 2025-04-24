import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1
from openopticalflow.shift_image_fun_refine_1_improved import shift_image_fun_refine_1 as shift_image_improved

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

def test_shift_image_improved():
    """Test and compare all implementations of shift_image_fun_refine_1 function"""
    
    # Create test case
    size = (50, 50)
    noise = 0.05
    img1, img2, u, v = create_test_case(size, noise)
    
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
    
    # Time the improved implementation
    start = time()
    try:
        img1_shift_improved, ux_improved, uy_improved = shift_image_improved(u, v, img1_uint8, img2_uint8)
        time_improved = time() - start
        improved_success = True
    except Exception as e:
        print(f"Improved implementation failed: {e}")
        time_improved = time() - start
        improved_success = False
    
    # Print timing results
    print(f"Execution times:")
    print(f"  Open implementation: {time_open:.6f} seconds")
    print(f"  Comparison implementation: {time_comp:.6f} seconds")
    print(f"  Improved implementation: {time_improved:.6f} seconds")
    
    # Calculate image quality metrics
    if open_success and comp_success and improved_success:
        # Calculate MSE with original second image (which should be the target)
        mse_open = np.mean((img1_shift_open.astype(float) - img2_uint8.astype(float))**2)
        mse_comp = np.mean((img1_shift_comp.astype(float) - img2_uint8.astype(float))**2)
        mse_improved = np.mean((img1_shift_improved.astype(float) - img2_uint8.astype(float))**2)
        
        print(f"Mean Squared Error with target image:")
        print(f"  Open implementation: {mse_open:.2f}")
        print(f"  Comparison implementation: {mse_comp:.2f}")
        print(f"  Improved implementation: {mse_improved:.2f}")
        
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
        plt.title('Original Image 2 (Target)')
        
        # Flow field
        plt.subplot(333)
        plt.quiver(u[::2, ::2], v[::2, ::2])
        plt.title('Flow Field')
        plt.axis('equal')
        
        # Open implementation
        plt.subplot(334)
        plt.imshow(img1_shift_open, cmap='gray')
        plt.colorbar()
        plt.title(f'Open Implementation\nMSE: {mse_open:.2f}')
        
        # Comparison implementation
        plt.subplot(335)
        plt.imshow(img1_shift_comp, cmap='gray')
        plt.colorbar()
        plt.title(f'Comparison Implementation\nMSE: {mse_comp:.2f}')
        
        # Improved implementation
        plt.subplot(336)
        plt.imshow(img1_shift_improved, cmap='gray')
        plt.colorbar()
        plt.title(f'Improved Implementation\nMSE: {mse_improved:.2f}')
        
        # Differences with target
        plt.subplot(337)
        plt.imshow(np.abs(img1_shift_open.astype(float) - img2_uint8.astype(float)), cmap='viridis')
        plt.colorbar()
        plt.title('Open - Difference with Target')
        
        plt.subplot(338)
        plt.imshow(np.abs(img1_shift_comp.astype(float) - img2_uint8.astype(float)), cmap='viridis')
        plt.colorbar()
        plt.title('Comparison - Difference with Target')
        
        plt.subplot(339)
        plt.imshow(np.abs(img1_shift_improved.astype(float) - img2_uint8.astype(float)), cmap='viridis')
        plt.colorbar()
        plt.title('Improved - Difference with Target')
        
        plt.tight_layout()
        plt.savefig('results/shift_image_improved_comparison.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_shift_image_improved()
