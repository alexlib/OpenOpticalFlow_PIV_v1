import numpy as np
from scipy.ndimage import convolve

def invariant2_factor(Vx, Vy, factor_x, factor_y):
    # factor_x: converting factor from pixel to m (m/pixel) in x
    # factor_y: converting factor from pixel to m (m/pixel) in y

    dx = 1
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # partial derivative

    Vx_x = convolve(Vx, D.T / dx, mode='reflect') / factor_x
    Vx_y = convolve(Vx, D / dx, mode='reflect') / factor_y

    Vy_x = convolve(Vy, D.T / dx, mode='reflect') / factor_x
    Vy_y = convolve(Vy, D / dx, mode='reflect') / factor_y

    M, N = Vx.shape
    QQ = np.zeros((M, N))

    for m in range(M):
        for n in range(N):
            u = np.array([[Vx_x[m, n], Vx_y[m, n]], [Vy_x[m, n], Vy_y[m, n]]])

            S = (u + u.T) / 2
            Q = (u - u.T) / 2

            QQ[m, n] = (np.trace(Q @ Q.T) - np.trace(S @ S.T)) / 2

    return QQ
