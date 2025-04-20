import numpy as np
from scipy.ndimage import convolve
from typing import Tuple

def horn_schunk_estimator(Ix: np.ndarray, Iy: np.ndarray, It: np.ndarray, lambda_: float, tol: float, maxiter: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the optical flow using the Horn-Schunck estimator.

    Args:
        Ix (np.ndarray): Partial derivative for x-axis.
        Iy (np.ndarray): Partial derivative for y-axis.
        It (np.ndarray): Partial derivative for time t.
        lambda_ (float): Regularization parameter.
        tol (float): Tolerance for the error.
        maxiter (int): Maximum number of iterations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Velocity fields u and v.
    """
    # Initialize velocities
    u = np.zeros_like(Ix)
    v = np.zeros_like(Ix)

    # Laplacian kernel for averaging
    kernel = np.array([[0, 0.25, 0],
                      [0.25, 0, 0.25],
                      [0, 0.25, 0]])

    for _ in range(maxiter):
        # Calculate averages
        uAvg = convolve(u, kernel, mode='reflect')
        vAvg = convolve(v, kernel, mode='reflect')

        # Calculate update according to Horn-Schunck equations
        den = lambda_ + Ix*Ix + Iy*Iy

        un = uAvg - Ix * (Ix*uAvg + Iy*vAvg + It) / den
        vn = vAvg - Iy * (Ix*uAvg + Iy*vAvg + It) / den

        # Check convergence
        du = np.abs(un - u).max()
        dv = np.abs(vn - v).max()

        u = un
        v = vn

        if max(du, dv) < tol:
            break

    return u, v
