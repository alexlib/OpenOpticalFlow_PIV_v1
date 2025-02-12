import numpy as np
from scipy.ndimage import convolve

def laplacian(u: np.ndarray, h: int) -> np.ndarray:
    """
    Compute the Laplacian of an image using a convolution filter.

    Args:
        u (np.ndarray): Input image.
        h (int): Step size for the convolution filter.

    Returns:
        np.ndarray: Computed Laplacian of the image.
    """
    H = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    delu = -u * convolve(np.ones(u.shape), H / (h * h), mode='same') + convolve(u, H / (h * h), mode='same')
    return delu
