import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from typing import Tuple

def pre_processing_a(Im1: np.ndarray, Im2: np.ndarray, scale_im: float, size_filter: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-process images by resizing and applying a Gaussian filter.

    Args:
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.
        scale_im (float): Scale factor for resizing images.
        size_filter (int): Size of the Gaussian filter for removing random noise.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Pre-processed images Im1 and Im2.
    """
    Im1 = Im1.astype(np.float64)
    Im2 = Im2.astype(np.float64)

    # Resize images
    if scale_im < 1:
        # Use scipy.ndimage.zoom instead of cv2.resize
        # zoom uses a scale factor directly
        Im1 = zoom(Im1, scale_im, order=1)  # order=1 for bilinear interpolation
        Im2 = zoom(Im2, scale_im, order=1)

    # Apply Gaussian filter to images
    # Apply Gaussian filter to images
    sigma = size_filter * 0.62 if size_filter > 1 else size_filter
    Im1 = gaussian_filter(Im1, sigma=sigma)
    Im2 = gaussian_filter(Im2, sigma=sigma)

    return Im1, Im2











