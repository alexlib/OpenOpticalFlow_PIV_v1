import numpy as np
from scipy import ndimage

def correction_illumination(im1, im2, window_shifting, size_average):
    """
    Correct illumination differences between two images.

    This function performs two types of illumination correction:
    1. Global correction: Adjusts overall brightness of second image to match first
    2. Local correction: Eliminates local illumination changes using moving average

    Args:
        im1 (np.ndarray): First input image
        im2 (np.ndarray): Second input image
        window_shifting (list): [x3, x4, y3, y4] defining the window boundaries
        size_average (int): Size for averaging kernel. If 0, no local correction is applied

    Returns:
        tuple: (i1, i2) Corrected image pair
    """
    # Convert to float64
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    i1 = im1.copy()  # Prevent modifying input
    i2 = im2.copy()

    # Global illumination correction
    x3, x4, y3, y4 = window_shifting
    i1_mean = np.mean(i1[y3:y4, x3:x4])
    i2_mean = np.mean(i2[y3:y4, x3:x4])
    r12 = i1_mean / i2_mean
    i2 = r12 * i2            

    # Local illumination correction
    if size_average > 0:
        kernel = np.ones((size_average, size_average)) / (size_average * size_average)
        i1_filtered = ndimage.convolve(i1, kernel, mode='reflect')
        i2_filtered = ndimage.convolve(i2, kernel, mode='reflect')
        i12f = i1_filtered - i2_filtered
        i2 = i2 + i12f

    return i1, i2
