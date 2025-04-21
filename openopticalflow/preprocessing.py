import numpy as np
from scipy.ndimage import gaussian_filter, zoom, map_coordinates
from typing import Tuple, List, Optional

def pre_processing(Im1: np.ndarray, Im2: np.ndarray, scale_im: float, size_filter: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-process images by resizing and applying a Gaussian filter.

    Args:
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.
        scale_im (float): Scale factor for resizing images.
        size_filter (float): Size of the Gaussian filter for removing random noise.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Pre-processed images Im1 and Im2.
    """
    # Convert to float if needed
    Im1 = Im1.astype(np.float64)
    Im2 = Im2.astype(np.float64)

    # Resize images
    if scale_im != 1:
        # Use scipy.ndimage.zoom for resizing
        Im1 = zoom(Im1, scale_im, order=1)  # order=1 for bilinear interpolation
        Im2 = zoom(Im2, scale_im, order=1)

    # Apply Gaussian filter to images
    # For compatibility with the comparison implementation, calculate sigma
    # based on the filter size
    sigma = size_filter * 0.62 if size_filter > 1 else size_filter
    Im1 = gaussian_filter(Im1, sigma=sigma)
    Im2 = gaussian_filter(Im2, sigma=sigma)

    return Im1, Im2

def correction_illumination(Im1: np.ndarray, Im2: np.ndarray, 
                           window_shifting: List[int], size_average: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct illumination differences between images.

    This function applies local mean subtraction to correct for illumination
    differences between two images, which can improve optical flow estimation.

    Args:
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.
        window_shifting (List[int]): [x1, x2, y1, y2] defining correction window.
        size_average (int): Size of averaging window. If <= 0, no correction is applied.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Illumination-corrected images Im1 and Im2.
    """
    # If size_average is 0 or negative, return original images
    if size_average <= 0:
        return Im1, Im2
    
    # Extract window boundaries
    x1, x2, y1, y2 = window_shifting
    
    # Calculate local means using a uniform kernel
    kernel = np.ones((size_average, size_average)) / (size_average ** 2)
    Im1_mean = gaussian_filter(Im1, sigma=size_average/2)
    Im2_mean = gaussian_filter(Im2, sigma=size_average/2)
    
    # Correct intensities by subtracting local mean and adding global mean
    Im1_corr = Im1 - Im1_mean + np.mean(Im1_mean)
    Im2_corr = Im2 - Im2_mean + np.mean(Im2_mean)
    
    return Im1_corr, Im2_corr

def shift_image_refine(ux: np.ndarray, uy: np.ndarray, Im1: np.ndarray, Im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shift image based on velocity field.

    This function shifts the first image (Im1) according to the velocity field (ux, uy)
    and returns the shifted image along with interpolated velocity fields.

    Args:
        ux (np.ndarray): Velocity field in x-direction.
        uy (np.ndarray): Velocity field in y-direction.
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Shifted image from Im1 based on velocity field
            - Interpolated velocity field in x-direction
            - Interpolated velocity field in y-direction
    """
    # Get dimensions
    m, n = Im1.shape
    m_flow, n_flow = ux.shape
    
    # Resize velocity fields to match image dimensions if needed
    if ux.shape != Im1.shape:
        zoom_factor_y = m / m_flow
        zoom_factor_x = n / n_flow
        ux_interp = zoom(ux, (zoom_factor_y, zoom_factor_x), order=1)
        uy_interp = zoom(uy, (zoom_factor_y, zoom_factor_x), order=1)
    else:
        ux_interp = ux.copy()
        uy_interp = uy.copy()
    
    # Create coordinate arrays for the output image
    y_coords, x_coords = np.mgrid[0:m, 0:n]
    
    # Calculate shifted coordinates
    # For backward mapping: where in the source image (Im1) to get each pixel in the output
    x_src = x_coords - ux_interp
    y_src = y_coords - uy_interp
    
    # Clip coordinates to valid range
    x_src = np.clip(x_src, 0, n-1)
    y_src = np.clip(y_src, 0, m-1)
    
    # Use map_coordinates for smooth interpolation
    coords = np.vstack((y_src.flatten(), x_src.flatten()))
    Im1_shift = map_coordinates(Im1, coords, order=1).reshape(Im1.shape)
    
    # Apply Gaussian smoothing to reduce artifacts
    Im1_shift = gaussian_filter(Im1_shift, sigma=0.5)
    
    # Ensure output is in the same range as input
    if np.issubdtype(Im1.dtype, np.integer):
        Im1_shift = np.clip(Im1_shift, 0, 255).astype(Im1.dtype)
    
    return Im1_shift, ux_interp, uy_interp

# For backward compatibility
pre_processing_a = pre_processing
