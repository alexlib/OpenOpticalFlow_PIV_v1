import numpy as np
from scipy.ndimage import convolve
from typing import Tuple

def liu_shen_estimator(I0: np.ndarray, I1: np.ndarray, f: np.ndarray, dx: int, dt: int, lambda_: float, tol: float, maxnum: int, u0: np.ndarray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the optical flow using the Liu-Shen estimator.

    Args:
        I0 (np.ndarray): First input image.
        I1 (np.ndarray): Second input image.
        f (np.ndarray): Related all boundary assumption.
        dx (int): Step size in x-direction.
        dt (int): Step size in time.
        lambda_ (float): Regularization parameter.
        tol (float): Tolerance for the error.
        maxnum (int): Maximum number of iterations.
        u0 (np.ndarray): Initial velocity field in x-direction.
        v0 (np.ndarray): Initial velocity field in y-direction.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Computed velocity fields u, v, and error.
    """
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # Partial derivative
    M = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4  # Mixed partial derivatives
    F = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]) / 4  # Average
    D2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])  # Partial derivative
    H = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # Filter

    IIx = convolve(I0, D / dx, mode='reflect')
    IIy = convolve(I0, D / dx, mode='reflect')
    II = I0 * convolve(I1, H / (dx * dx), mode='reflect')

    Ixt = convolve((I1 - I0) / dt - f, D2 / dx, mode='reflect')
    Iyt = convolve((I1 - I0) / dt - f, D2 / dx, mode='reflect')

    # Initialize the parameters
    r, c = I0.shape
    total_error = np.inf
    u = u0.copy()
    v = v0.copy()
    error = []

    # Iterative computation
    k = 0
    while total_error > tol and k < maxnum:
        bu = 2 * convolve(u, D / dx, mode='reflect') + convolve(v, D / dx, mode='reflect')
        bv = convolve(u, D / dx, mode='reflect') + 2 * convolve(v, D / dx, mode='reflect')
        b = lambda_ * convolve(u, H / (dx * dx), mode='reflect') + II

        unew = -(b * bu + bv) / (2 * bu * bv + b ** 2)
        vnew = -(b * bv + bu) / (2 * bu * bv + b ** 2)

        total_error = np.linalg.norm(unew - u) + np.linalg.norm(vnew - v)
        u = unew
        v = vnew
        error.append(total_error)
        k += 1

    return u, v, np.array(error)
