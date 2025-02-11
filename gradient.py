import numpy as np
from scipy.ndimage import convolve

def gradient(Vx, Vy):
    dx = 1
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # partial derivative

    Vy_x = convolve(Vy, D.T / dx, mode='reflect')
    Vx_y = convolve(Vx, D / dx, mode='reflect')
    
    grad_mag = np.sqrt(Vy_x**2 + Vx_y**2)
    
    return grad_mag




















