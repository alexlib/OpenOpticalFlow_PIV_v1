import numpy as np
from scipy import ndimage
from typing import Tuple

def shift_image_fun_refine_1(ux: np.ndarray, uy: np.ndarray, Im1: np.ndarray, Im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shift and refine image based on velocity fields.
    
    This function shifts the first image (Im1) according to the velocity field (ux, uy)
    and returns the shifted image along with interpolated velocity fields.
    
    Args:
        ux (np.ndarray): Velocity field in x-direction
        uy (np.ndarray): Velocity field in y-direction
        Im1 (np.ndarray): First input image
        Im2 (np.ndarray): Second input image
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Shifted image from Im1 based on velocity field
            - Interpolated velocity field in x-direction
            - Interpolated velocity field in y-direction
    """
    # Convert inputs to float
    Im1 = Im1.astype(float)
    Im2 = Im2.astype(float)
    ux = ux.astype(float)
    uy = uy.astype(float)
    
    # Get dimensions
    m1, n1 = Im1.shape
    m0, n0 = ux.shape
    
    # Resize velocity field to match image dimensions if needed
    if ux.shape != Im1.shape:
        zoom_factor_y = m1 / m0
        zoom_factor_x = n1 / n0
        uxI = ndimage.zoom(ux, (zoom_factor_y, zoom_factor_x), order=1)
        uyI = ndimage.zoom(uy, (zoom_factor_y, zoom_factor_x), order=1)
    else:
        uxI = ux.copy()
        uyI = uy.copy()
    
    # Create coordinate arrays for the output image
    y_coords, x_coords = np.mgrid[0:m1, 0:n1]
    
    # Calculate shifted coordinates
    # For backward mapping: where in the source image (Im1) to get each pixel in the output
    x_src = x_coords - uxI
    y_src = y_coords - uyI
    
    # Clip coordinates to valid range
    x_src = np.clip(x_src, 0, n1-1)
    y_src = np.clip(y_src, 0, m1-1)
    
    # Use map_coordinates for smooth interpolation
    coords = np.vstack((y_src.flatten(), x_src.flatten()))
    Im1_shift = ndimage.map_coordinates(Im1, coords, order=1).reshape(Im1.shape)
    
    # Apply Gaussian smoothing to reduce artifacts
    Im1_shift = ndimage.gaussian_filter(Im1_shift, sigma=0.5)
    
    # Ensure output is in the same range as input
    if np.issubdtype(Im1.dtype, np.integer):
        Im1_shift = np.clip(Im1_shift, 0, 255).astype(Im1.dtype)
    
    return Im1_shift, uxI, uyI

# Example usage
if __name__ == "__main__":
    # Create sample data
    size = (100, 100)
    Im1 = np.random.rand(size[0], size[1]) * 255
    Im2 = np.random.rand(size[0], size[1]) * 255
    
    # Create sample flow field (rotating pattern)
    y, x = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
    center_y, center_x = size[0] // 2, size[1] // 2
    ux = -(y - center_y) / 10  # horizontal flow
    uy = (x - center_x) / 10   # vertical flow
    
    # Apply shift
    Im1_shift, uxI, uyI = shift_image_fun_refine_1(ux, uy, Im1, Im2)
    
    # Display results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(Im1, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(132)
    plt.imshow(Im1_shift, cmap='gray')
    plt.title('Shifted Image')
    
    plt.subplot(133)
    plt.quiver(uxI[::5, ::5], uyI[::5, ::5])
    plt.title('Flow Field')
    
    plt.tight_layout()
    plt.show()
