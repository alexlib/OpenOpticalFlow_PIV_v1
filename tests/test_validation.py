import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from skimage import io
from typing import Tuple, Optional, List, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openopticalflow.OpticalFlowPhysics_fun import OpticalFlowPhysics_fun
from openopticalflow.pre_processing_a import pre_processing_a
from openopticalflow.correction_illumination import correction_illumination
from openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1

def create_synthetic_case(size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Create a synthetic test case with random Gaussian blobs.
    
    Args:
        size (Tuple[int, int]): Size of the image to create.
        
    Returns:
        np.ndarray: Synthetic image with random patterns.
    """
    # Create a random pattern
    np.random.seed(42)  # for reproducibility
    img = np.zeros(size)

    # Add random Gaussian blobs
    for _ in range(50):
        x = np.random.randint(0, size[1])
        y = np.random.randint(0, size[0])
        sigma = np.random.uniform(2, 5)
        amplitude = np.random.uniform(0.5, 1.0)

        y_grid, x_grid = np.ogrid[-y:size[0]-y, -x:size[1]-x]
        r2 = x_grid*x_grid + y_grid*y_grid
        img += amplitude * np.exp(-r2/(2.*sigma**2))

    # Normalize to 0-255 range
    img = ((img - img.min()) * (255.0 / (img.max() - img.min()))).astype(np.uint8)

    return img

def apply_known_shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Apply known shift to image using scipy.ndimage.shift.
    
    Args:
        image (np.ndarray): Input image.
        dx (float): Shift in x-direction.
        dy (float): Shift in y-direction.
        
    Returns:
        np.ndarray: Shifted image.
    """
    shifted = shift(image, (dy, dx), mode='reflect', order=3)
    return shifted

def compute_error_metrics(ux: np.ndarray, uy: np.ndarray, true_dx: float, true_dy: float) -> Tuple[float, float]:
    """
    Compute error metrics between estimated and true displacement.
    
    Args:
        ux (np.ndarray): Estimated x-displacement field.
        uy (np.ndarray): Estimated y-displacement field.
        true_dx (float): True x-displacement.
        true_dy (float): True y-displacement.
        
    Returns:
        Tuple[float, float]: RMSE and MAE error metrics.
    """
    error_x = ux - true_dx
    error_y = uy - true_dy

    rmse = np.sqrt(np.mean(error_x**2 + error_y**2))
    mae = np.mean(np.abs(error_x) + np.abs(error_y))

    return rmse, mae

def plot_flow_field(img1: np.ndarray, img2: np.ndarray, ux: np.ndarray, uy: np.ndarray, 
                   true_dx: Optional[float] = None, true_dy: Optional[float] = None) -> None:
    """
    Plot original and shifted images along with the flow field.
    
    Args:
        img1 (np.ndarray): Original image.
        img2 (np.ndarray): Shifted/second image.
        ux (np.ndarray): x-displacement field.
        uy (np.ndarray): y-displacement field.
        true_dx (Optional[float]): True x-displacement (if known).
        true_dy (Optional[float]): True y-displacement (if known).
    """
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(131)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Image')

    # Shifted image
    plt.subplot(132)
    plt.imshow(img2, cmap='gray')
    plt.title('Shifted/Second Image')

    # Flow field with normalized vectors
    plt.subplot(133)
    plt.imshow(img1, cmap='gray', alpha=0.5)

    # Subsample the flow field for better visualization
    sy, sx = ux.shape
    step = max(sx, sy) // 25  # Show about 25 vectors in largest dimension

    y, x = np.mgrid[0:sy:step, 0:sx:step]
    u = ux[::step, ::step]
    v = uy[::step, ::step]

    # Normalize vectors
    magnitude = np.sqrt(u**2 + v**2)
    if magnitude.max() > 0:
        u = u / magnitude.max()
        v = v / magnitude.max()

    plt.quiver(x, y, u, v, color='r', scale=15, width=0.003)
    
    if true_dx is not None and true_dy is not None:
        plt.title(f'Flow Field (normalized)\nTrue: dx={true_dx}, dy={true_dy}\nEst: dx={np.mean(ux):.2f}, dy={np.mean(uy):.2f}')
    else:
        plt.title(f'Flow Field (normalized)\nEst: dx={np.mean(ux):.2f}, dy={np.mean(uy):.2f}')

    plt.tight_layout()

def run_test_case(img1_path: str, img2_path: Optional[str] = None, 
                 true_dx: Optional[float] = None, true_dy: Optional[float] = None, 
                 create_synthetic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run optical flow test case for given image pair.
    
    Args:
        img1_path (str): Path to first image.
        img2_path (Optional[str]): Path to second image (if not synthetic).
        true_dx (Optional[float]): True x-displacement (if known).
        true_dy (Optional[float]): True y-displacement (if known).
        create_synthetic (bool): Whether to create synthetic shifted image.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Estimated displacement fields (ux, uy) and vorticity.
    """
    # Read images
    img1 = io.imread(img1_path)

    if create_synthetic:
        # Create shifted image using known displacement
        img2 = apply_known_shift(img1, true_dx, true_dy)
        # Save shifted image for verification
        base_name = img1_path.rsplit('.', 1)[0]
        io.imsave(f'{base_name}_shifted.tif', img2)
    else:
        img2 = io.imread(img2_path)

    # Parameters for optical flow computation
    lambda_1 = 20.0    # Horn-Schunck regularization
    lambda_2 = 2000.0  # Liu-Shen regularization
    scale_im = 1.0     # No downscaling
    size_filter = 4    # Gaussian filter size
    size_average = 0   # No illumination correction
    no_iteration = 3   # Number of refinement iterations

    # Store originals
    i1_original = img1.astype(float)
    i2_original = img2.astype(float)

    # Initial pre-processing
    window_shifting = np.array([1, i1_original.shape[1], 1, i1_original.shape[0]])
    i1, i2 = correction_illumination(i1_original, i2_original, window_shifting, size_average)
    i1, i2 = pre_processing_a(i1, i2, scale_im, size_filter)

    # Initial optical flow
    ux0, uy0, vor, ux_horn, uy_horn, error1 = OpticalFlowPhysics_fun(i1, i2, lambda_1, lambda_2)

    # Convert to uint8 for shift operation
    im1 = i1_original.astype(np.uint8)
    im2 = i2_original.astype(np.uint8)

    ux_corr = ux0.copy()
    uy_corr = uy0.copy()

    # Iterative refinement
    for k in range(no_iteration):
        # Shift image based on current estimate
        im1_shift, uxi, uyi = shift_image_fun_refine_1(ux_corr, uy_corr, im1, im2)

        # Convert to float for processing
        i1 = im1_shift.astype(float)
        i2 = im2.astype(float)

        # Additional pre-processing with smaller filter
        size_filter_1 = 2
        i1, i2 = pre_processing_a(i1, i2, 1, size_filter_1)

        # Calculate correction
        dux, duy, vor, dux_horn, duy_horn, error2 = OpticalFlowPhysics_fun(
            i1, i2, lambda_1, lambda_2)

        # Update flow fields
        ux_corr = uxi + dux
        uy_corr = uyi + duy

    # Final velocity field
    ux = ux_corr
    uy = uy_corr

    # Plot results with normalized vectors
    plot_flow_field(img1, img2, ux, uy, true_dx, true_dy)

    if true_dx is not None and true_dy is not None:
        # Compute error metrics
        rmse, mae = compute_error_metrics(ux, uy, true_dx, true_dy)
        print(f"True displacement: dx={true_dx}, dy={true_dy}")
        print(f"Mean estimated displacement: dx={np.mean(ux):.3f}, dy={np.mean(uy):.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
    else:
        print(f"Mean displacement: dx={np.mean(ux):.3f}, dy={np.mean(uy):.3f}")

    return ux, uy, vor

def test_validation():
    """Run validation tests on sample images."""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Test Case 1: Create synthetic test case
    print("\nTest Case 1: Synthetic Test Case")
    print("---------------------------------")
    
    # Create synthetic image
    synthetic_img = create_synthetic_case(size=(200, 200))
    io.imsave('results/synthetic_test_case.tif', synthetic_img)
    
    # Run test with synthetic shift
    ux1, uy1, vor1 = run_test_case(
        img1_path='results/synthetic_test_case.tif',
        img2_path=None,
        true_dx=2.5,
        true_dy=1.5,
        create_synthetic=True
    )
    
    # Test Case 2: Use existing image pair if available
    try:
        print("\nTest Case 2: Real Image Pair")
        print("---------------------------")
        ux2, uy2, vor2 = run_test_case(
            img1_path='img/frame_001.tif',
            img2_path='img/frame_002.tif',
            create_synthetic=False
        )
        
        # Additional visualization for real case
        plt.figure(figsize=(10, 5))
        
        # Plot vorticity field
        plt.subplot(121)
        vor_normalized = vor2 / np.max(np.abs(vor2)) if np.max(np.abs(vor2)) > 0 else vor2
        plt.imshow(vor_normalized, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Normalized Vorticity')
        plt.title('Vorticity Field')
        
        # Plot velocity magnitude
        plt.subplot(122)
        vel_mag = np.sqrt(ux2**2 + uy2**2)
        vel_mag_normalized = vel_mag / np.max(vel_mag) if np.max(vel_mag) > 0 else vel_mag
        plt.imshow(vel_mag_normalized, cmap='viridis')
        plt.colorbar(label='Normalized Velocity Magnitude')
        plt.title('Velocity Magnitude')
        
        plt.tight_layout()
        plt.savefig('results/validation_real_case.png')
        
    except FileNotFoundError:
        print("Real image pair not found. Skipping Test Case 2.")
    
    plt.savefig('results/validation_synthetic_case.png')
    plt.close('all')

if __name__ == "__main__":
    test_validation()
