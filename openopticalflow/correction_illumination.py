import numpy as np
from scipy import ndimage

def correction_illumination(im1, im2, window_shifting, size_average):
    """
    Correct illumination differences between two images.

    This function performs two types of illumination correction:
    1. Global correction: Adjusts the overall brightness of the second image to match the first
    2. Local correction: Eliminates local illumination changes using a moving average filter

    Args:
        im1 (np.ndarray): First input image
        im2 (np.ndarray): Second input image
        window_shifting (list or array): Region [x_min, x_max, y_min, y_max] defining the window
                                        for mean intensity calculation
        size_average (int): Size of averaging window kernel. If 0, no local illumination
                           correction is applied

    Returns:
        tuple: (i1, i2) Corrected versions of the input images

    Examples:
        >>> import numpy as np
        >>> img1 = np.ones((100, 100)) * 100  # Bright image
        >>> img2 = np.ones((100, 100)) * 50   # Darker image
        >>> window = [0, 100, 0, 100]  # Full image
        >>> i1, i2 = correction_illumination(img1, img2, window, 0)
        >>> np.mean(i2)  # Should be close to 100 now
        100.0
    """
    # Convert to float if needed
    im1 = im1.astype(float)
    im2 = im2.astype(float)

    i1 = im1.copy()
    i2 = im2.copy()

    # Extract window coordinates
    x3, x4, y3, y4 = window_shifting

    # Adjust overall illumination change
    i1_mean = np.mean(i1[y3:y4, x3:x4])
    i2_mean = np.mean(i2[y3:y4, x3:x4])
    r12 = i1_mean / i2_mean
    i2 = r12 * i2

    # Normalize intensity for i2 to eliminate local illumination changes
    if size_average > 0:
        # Create averaging kernel
        kernel = np.ones((size_average, size_average)) / (size_average * size_average)

        # Apply filtering and calculate difference
        i1_filtered = ndimage.convolve(i1, kernel, mode='reflect')
        i2_filtered = ndimage.convolve(i2, kernel, mode='reflect')
        i12f = i1_filtered - i2_filtered

        # Adjust i2
        i2 = i2 + i12f

    return i1, i2
