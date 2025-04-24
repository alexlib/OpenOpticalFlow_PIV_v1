import numpy as np
from scipy.ndimage import convolve

def laplacian(u: np.ndarray, h: int) -> np.ndarray:
    """
    Compute the Laplacian of an image using a convolution filter.

    This function calculates the Laplacian using a 3x3 kernel with all neighbors weighted equally.
    For edge handling, reflection boundary conditions are used.

    Args:
        u (np.ndarray): Input image/array
        h (int): Step size for the convolution filter

    Returns:
        np.ndarray: Computed Laplacian of the image

    Notes:
        An alternative kernel that weights only direct neighbors (not diagonals) can be:
        H = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    """
    # Define kernel
    H = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # Calculate Laplacian
    ones_filtered = convolve(np.ones_like(u), H/(h*h), mode='reflect')
    u_filtered = convolve(u, H/(h*h), mode='reflect')
    delu = -u * ones_filtered + u_filtered

    return delu
