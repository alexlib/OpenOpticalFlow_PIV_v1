import numpy as np
from scipy.ndimage import convolve

def correction_illumination(Im1, Im2, window_shifting, size_average):
    """
    Correct illumination differences between two images.

    Args:
        Im1 (np.ndarray): First input image
        Im2 (np.ndarray): Second input image
        window_shifting (list): [x3, x4, y3, y4] defining the window boundaries
        size_average (int): Size for averaging kernel. If 0, no local correction is applied

    Returns:
        tuple: (I1, I2) Corrected image pair
    """
    # Convert to float64
    Im1 = Im1.astype(np.float64)
    Im2 = Im2.astype(np.float64)

    I1 = Im1
    I2 = Im2

    # Adjusting the overall illumination change    
    x3, x4, y3, y4 = window_shifting

    I1_mean = np.mean(I1[y3:y4, x3:x4])
    I2_mean = np.mean(I2[y3:y4, x3:x4])
    R12 = I1_mean / I2_mean
    I2 = R12 * I2            

    # Normalize the intensity for I2 to eliminate the local change of illumination light
    if size_average > 0:
        N = size_average
        h = np.ones((N, N)) / (N * N)
        I12F = convolve(I1, h, mode='reflect') - convolve(I2, h, mode='reflect')
        I2 = I2 + I12F
    else:
        I2 = I2

    return I1, I2




