import numpy as np
from scipy.ndimage import convolve
from typing import Tuple

def horn_schunk_estimator(Ix: np.ndarray, Iy: np.ndarray, It: np.ndarray, lambda_: float, tol: float, maxiter: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horn-Schunck optical flow estimation algorithm.

    This implementation follows the classic Horn-Schunck method for estimating optical flow
    between two images. It solves the optical flow constraint equation with a smoothness
    regularization term.

    Args:
        Ix (np.ndarray): Spatial derivative of the image in the x direction.
        Iy (np.ndarray): Spatial derivative of the image in the y direction.
        It (np.ndarray): Temporal derivative of the image.
        lambda_ (float): Regularization parameter. Higher values produce smoother flow fields.
        tol (float): Convergence tolerance. The algorithm stops when the maximum change
                    in the flow field is less than this value.
        maxiter (int): Maximum number of iterations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (u, v) - Optical flow components in x and y directions.

    References:
        Horn, B. K., & Schunck, B. G. (1981). Determining optical flow.
        Artificial intelligence, 17(1-3), 185-203.
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
