import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1 as shift_image_open
# Only use the main implementation
from openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1

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

        # No comparison implementation anymore
        comp_success = False

        # Print timing results
        print(f"Execution times:")
        print(f"  Open implementation: {time_open:.6f} seconds")

        # If open implementation succeeded, show results
        if open_success:

            # Plot results
            plt.figure(figsize=(12, 8))

            # Original images
            plt.subplot(231)
            plt.imshow(img1_uint8, cmap='gray')
            plt.colorbar()
            plt.title('Original Image 1')

            plt.subplot(232)
            plt.imshow(img2_uint8, cmap='gray')
            plt.colorbar()
            plt.title('Original Image 2')

            # Flow field
            plt.subplot(233)
            plt.quiver(u[::2, ::2], v[::2, ::2])
            plt.title('Flow Field')
            plt.axis('equal')

            # Implementation results
            plt.subplot(234)
            plt.imshow(img1_shift_open, cmap='gray')
            plt.colorbar()
            plt.title('Shifted Image')

            plt.subplot(235)
            plt.imshow(ux_open, cmap='RdBu_r')
            plt.colorbar()
            plt.title('ux')

            plt.subplot(236)
            plt.imshow(uy_open, cmap='RdBu_r')
            plt.colorbar()
            plt.title('uy')

            plt.tight_layout()
            plt.savefig(f'results/shift_image_{name.replace(" ", "_").lower()}.png')
            plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run tests
    test_shift_image_fun_refine_1()
