import numpy as np
from scipy.ndimage import convolve
from typing import Tuple

def horn_schunk_estimator(Ix: np.ndarray, Iy: np.ndarray, It: np.ndarray, lambda_: float, tol: float, maxnum: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the optical flow using the Horn-Schunck estimator.

    Args:
        Ix (np.ndarray): Partial derivative for x-axis.
        Iy (np.ndarray): Partial derivative for y-axis.
        It (np.ndarray): Partial derivative for time t.
        lambda_ (float): Regularization parameter.
        tol (float): Tolerance for the error.
        maxnum (int): Maximum number of iterations.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Velocity fields u and v.
    """
    r, c = Ix.shape

    # Initialize the parameters
    horizontal = np.array([[3, 1, 3], [1, 0, 1], [3, 1, 3]]) / 8
    vertical = np.array([[3, 3, 3], [1, 0, 1], [1, 1, 1]]) / 8
    cmtx = 8 * np.ones((r, c))
    cmtx[1, :] = horizontal
    cmtx[r, :] = horizontal
    cmtx[:, 1] = vertical
    cmtx[:, c] = vertical

    uv = (Ix * Iy) / (cmtx * (Ix**2 + Iy**2) + lambda_ * cmtx**2)
    u1 = (Iy**2 + lambda_ * cmtx) / (cmtx * (Ix**2 + Iy**2) + lambda_ * cmtx**2)
    u2 = (Ix * It) / (Ix**2 + Iy**2 + lambda_ * cmtx)
    v1 = (Ix**2 + lambda_ * cmtx) / (cmtx * (Ix**2 + Iy**2) + lambda_ * cmtx**2)
    v2 = (Iy * It) / (Ix**2 + Iy**2 + lambda_ * cmtx)

    # Initialize the velocity fields
    u = np.zeros((r, c))
    v = np.zeros((r, c))

    # Iterative computation
    k = 0
    total_error = np.inf
    while total_error > tol and k < maxnum:
        total_error = 0
        for n in range(1, c):
            for m in range(1, r):
                if n == 1:
                    if m == 1:
                        tmpu = u[m + 1, n] + u[m, n + 1] + u[m + 1, n + 1]
                        tmpv = v[m + 1, n] + v[m, n + 1] + v[m + 1, n + 1]
                    elif m == r:
                        tmpu = u[m - 1, n] + u[m, n + 1] + u[m - 1, n + 1]
                        tmpv = v[m - 1, n] + v[m, n + 1] + v[m - 1, n + 1]
                    else:
                        tmpu = u[m - 1, n] + u[m + 1, n] + u[m - 1, n + 1] + u[m + 1, n + 1] + u[m, n + 1]
                        tmpv = v[m - 1, n] + v[m + 1, n] + v[m - 1, n + 1] + v[m + 1, n + 1] + v[m, n + 1]
                elif n == c:
                    if m == 1:
                        tmpu = u[m + 1, n] + u[m, n - 1] + u[m + 1, n - 1]
                        tmpv = v[m + 1, n] + v[m, n - 1] + v[m + 1, n - 1]
                    elif m == r:
                        tmpu = u[m - 1, n] + u[m, n - 1] + u[m - 1, n - 1]
                        tmpv = v[m - 1, n] + v[m, n - 1] + v[m - 1, n - 1]
                    else:
                        tmpu = u[m - 1, n] + u[m + 1, n] + u[m - 1, n - 1] + u[m + 1, n - 1] + u[m, n - 1]
                        tmpv = v[m - 1, n] + v[m + 1, n] + v[m - 1, n - 1] + v[m + 1, n - 1] + v[m, n - 1]
                else:
                    if m == 1:
                        tmpu = u[m, n - 1] + u[m, n + 1] + u[m + 1, n - 1] + u[m + 1, n] + u[m + 1, n + 1]
                        tmpv = v[m, n - 1] + v[m, n + 1] + v[m + 1, n - 1] + v[m + 1, n] + v[m + 1, n + 1]
                    elif m == r:
                        tmpu = u[m, n - 1] + u[m, n + 1] + u[m - 1, n - 1] + u[m - 1, n] + u[m - 1, n + 1]
                        tmpv = v[m, n - 1] + v[m, n + 1] + v[m - 1, n - 1] + v[m - 1, n] + v[m - 1, n + 1]
                    else:
                        tmpu = u[m - 1, n - 1] + u[m + 1, n - 1] + u[m - 1, n + 1] + u[m + 1, n + 1] + u[m - 1, n] + u[m + 1, n] + u[m, n - 1] + u[m, n + 1]
                        tmpv = v[m - 1, n - 1] + v[m + 1, n - 1] + v[m - 1, n + 1] + v[m + 1, n + 1] + v[m - 1, n] + v[m + 1, n] + v[m, n - 1] + v[m, n + 1]

                unew = u1 * tmpu - uv * tmpv - u2
                vnew = v1 * tmpv - uv * tmpu - v2

                total_error += np.linalg.norm(unew - u) + np.linalg.norm(vnew - v)
                u[m, n] = unew
                v[m, n] = vnew

        k += 1

    return u, v
