import numpy as np
from scipy.ndimage import convolve

def invariant2_factor(Vx: np.ndarray, Vy: np.ndarray, factor_x: float, factor_y: float) -> np.ndarray:
    """
    Compute the invariant QQ using convolution filters.

    Args:
        Vx (np.ndarray): Input matrix Vx.
        Vy (np.ndarray): Input matrix Vy.
        factor_x (float): Conversion factor from pixel to m (m/pixel) in x.
        factor_y (float): Conversion factor from pixel to m (m/pixel) in y.

    Returns:
        np.ndarray: Computed invariant QQ.
    """
    dx = 1
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # Partial derivative

    Vx_x = convolve(Vx, D / dx, mode='symmetric') / factor_x
    Vx_y = convolve(Vx, D / dx, mode='symmetric') / factor_y
    Vy_x = convolve(Vy, D / dx, mode='symmetric') / factor_x
    Vy_y = convolve(Vy, D / dx, mode='symmetric') / factor_y

    M, N = Vx.shape

    QQ = np.zeros((M, N))

    for m in range(M):
        for n in range(N):
            u = np.array([[Vx_x[m, n], Vx_y[m, n]], [Vy_x[m, n], Vy_y[m, n]]])
            S = (u + u.T) / 2
            Q = (np.trace(S) ** 2 - np.trace(S @ S)) / 2
            QQ[m, n] = Q

    return QQ
