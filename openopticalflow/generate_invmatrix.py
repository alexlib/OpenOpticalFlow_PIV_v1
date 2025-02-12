import numpy as np
from scipy.ndimage import convolve

def generate_invmatrix(I: np.ndarray, alpha: float, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an inverse matrix using convolution filters.

    Args:
        I (np.ndarray): Input matrix.
        alpha (float): Scaling factor.
        h (int): Size of the convolution filter.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Generated inverse matrices B11, B12, B22.
    """
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # Partial derivative
    M = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4  # Mixed partial derivatives
    F = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]) / 4  # Average
    D2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])  # Partial derivative
    H = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # Filter

    r, c = I.shape
    cmtx = np.ones((r, c)) / (h * h)

    A11 = I * (convolve(I, D2, mode='reflect') - 2 * I / (h * h)) - alpha * cmtx
    A22 = I * (convolve(I, D2.T, mode='reflect') - 2 * I / (h * h)) - alpha * cmtx
    A12 = I * convolve(I, M, mode='reflect')

    DetA = A11 * A22 - A12 * A12

    B11 = A22 / DetA
    B12 = -A12 / DetA
    B22 = A11 / DetA

    return B11, B12, B22
