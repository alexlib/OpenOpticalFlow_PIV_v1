import numpy as np
from scipy.ndimage import convolve

def liu_shen_estimator(I0, I1, f, dx, dt, lambda_, tol, maxnum, u0, v0):
    # Define filters
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    M = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    F = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    D2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    H = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # Compute partial derivatives
    IIx = I0 * convolve(I0, D / dx, mode='nearest')
    IIy = I0 * convolve(I0, D.T / dx, mode='nearest')
    II = I0 * I0
    Ixt = I0 * convolve((I1 - I0) / dt - f, D / dx, mode='nearest')
    Iyt = I0 * convolve((I1 - I0) / dt - f, D.T / dx, mode='nearest')

    # Initialize parameters
    k = 0
    total_error = 100000000
    u = u0
    v = v0

    r, c = I1.shape

    # Generate inverse matrix (assuming generate_invmatrix is defined elsewhere)
    B11, B12, B22 = generate_invmatrix(I0, lambda_, dx)
    error = []

    while total_error > tol and k < maxnum:
        bu = (2 * IIx * convolve(u, D / dx, mode='nearest') +
              IIx * convolve(v, D.T / dx, mode='nearest') +
              IIy * convolve(v, D / dx, mode='nearest') +
              II * convolve(u, F / (dx * dx), mode='nearest') +
              II * convolve(v, M / (dx * dx), mode='nearest') +
              lambda_ * convolve(u, H / (dx * dx), mode='nearest') + Ixt)

        bv = (IIy * convolve(u, D / dx, mode='nearest') +
              IIx * convolve(u, D.T / dx, mode='nearest') +
              2 * IIy * convolve(v, D.T / dx, mode='nearest') +
              II * convolve(u, M / (dx * dx), mode='nearest') +
              II * convolve(v, F.T / (dx * dx), mode='nearest') +
              lambda_ * convolve(v, H / (dx * dx), mode='nearest') + Iyt)

        unew = -(B11 * bu + B12 * bv)
        vnew = -(B12 * bu + B22 * bv)
        total_error = (np.linalg.norm(unew - u, 'fro') + np.linalg.norm(vnew - v, 'fro')) / (r * c)
        u = unew
        v = vnew
        error.append(total_error)
        k += 1

    return u, v, error
