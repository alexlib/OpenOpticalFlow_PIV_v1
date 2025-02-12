import numpy as np
from scipy.ndimage import gaussian_filter

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
        Im1 = cv2.resize(Im1, None, fx=scale_im, fy=scale_im, interpolation=cv2.INTER_AREA)
        Im2 = cv2.resize(Im2, None, fx=scale_im, fy=scale_im, interpolation=cv2.INTER_AREA)

    # Apply Gaussian filter to images
    Im1 = gaussian_filter(Im1, sigma=size_filter)
    Im2 = gaussian_filter(Im2, sigma=size_filter)

    return Im1, Im2











