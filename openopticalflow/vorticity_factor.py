import numpy as np
from scipy.ndimage import convolve

def vorticity_factor(Vx, Vy, factor_x, factor_y):
    # factor_x: converting factor from pixel to m (m/pixel) in x
    # factor_y: converting factor from pixel to m (m/pixel) in y

    # Vx = convolve(Vx, np.ones((5, 5))/25, mode='reflect')
    # Vy = convolve(Vy, np.ones((5, 5))/25, mode='reflect')

    dx = 1
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # partial derivative
    Vy_x = convolve(Vy, D.T / dx, mode='reflect')
    Vx_y = convolve(Vx, D / dx, mode='reflect')
    omega = (Vy_x / factor_x - Vx_y / factor_y)
    
    return omega




















