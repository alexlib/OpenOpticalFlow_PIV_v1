import numpy as np
from scipy.ndimage import convolve
from typing import Tuple, List
from openopticalflow.generate_invmatrix import generate_invmatrix

def liu_shen_estimator(I0: np.ndarray, I1: np.ndarray, f: np.ndarray, dx: float, dt: float,
                       lambda_param: float, tol: float, maxnum: int, u0: np.ndarray, v0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Compute the optical flow using the Liu-Shen estimator.

    This implementation follows the Liu-Shen method for optical flow estimation,
    which is particularly suitable for fluid flows. It uses a physics-based
    approach that can incorporate transport equations.

    Args:
        I0 (np.ndarray): First input image
        I1 (np.ndarray): Second input image
        f (np.ndarray): Physical transport term (typically zero for standard optical flow)
        dx (float): Spatial step size
        dt (float): Temporal step size
        lambda_param (float): Regularization parameter
        tol (float): Convergence tolerance
        maxnum (int): Maximum number of iterations
        u0 (np.ndarray): Initial velocity field in x-direction
        v0 (np.ndarray): Initial velocity field in y-direction

    Returns:
        Tuple[np.ndarray, np.ndarray, List[float]]: Computed velocity fields u, v, and convergence error history
    """
    # Define derivative kernels
    d = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # Partial derivative
    m = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4  # Mixed partial derivatives
    f_kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])  # Average
    h = np.ones((3, 3))  # Filter

    # Calculate derivatives and products
    iix = I0 * convolve(I0, d/dx, mode='reflect')
    iiy = I0 * convolve(I0, d.T/dx, mode='reflect')
    ii = I0 * I0
    ixt = I0 * convolve((I1-I0)/dt-f, d/dx, mode='reflect')
    iyt = I0 * convolve((I1-I0)/dt-f, d.T/dx, mode='reflect')

    # Initialize parameters
    r, c = I0.shape
    k = 0
    total_error = np.inf
    u = u0.copy()
    v = v0.copy()
    error = []

    # Generate inverse matrix
    b11, b12, b22 = generate_invmatrix(I0, lambda_param, dx)

    # Iterative computation
    while total_error > tol and k < maxnum:
        # Calculate right-hand side terms
        bu = (2*iix * convolve(u, d/dx, mode='reflect') +
              iix * convolve(v, d.T/dx, mode='reflect') +
              iiy * convolve(v, d/dx, mode='reflect') +
              ii * convolve(u, f_kernel/(dx*dx), mode='reflect') +
              ii * convolve(v, m/(dx*dx), mode='reflect') +
              lambda_param * convolve(u, h/(dx*dx), mode='reflect') + ixt)

        bv = (iiy * convolve(u, d/dx, mode='reflect') +
              iix * convolve(u, d.T/dx, mode='reflect') +
              2*iiy * convolve(v, d.T/dx, mode='reflect') +
              ii * convolve(u, m/(dx*dx), mode='reflect') +
              ii * convolve(v, f_kernel.T/(dx*dx), mode='reflect') +
              lambda_param * convolve(v, h/(dx*dx), mode='reflect') + iyt)

        # Calculate new flow values
        unew = -(b11*bu + b12*bv)
        vnew = -(b12*bu + b22*bv)

        # Replace any NaN or Inf values with the previous iteration's values
        unew = np.where(np.isfinite(unew), unew, u)
        vnew = np.where(np.isfinite(vnew), vnew, v)

        # Calculate error and normalize by image size
        total_error = (np.linalg.norm(unew-u, 'fro') +
                      np.linalg.norm(vnew-v, 'fro'))/(r*c)

        # Update flow fields
        u = unew
        v = vnew
        error.append(float(total_error))
        k += 1

    return u, v, error
