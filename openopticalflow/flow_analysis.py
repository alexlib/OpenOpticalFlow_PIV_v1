import numpy as np
from scipy import ndimage

def vorticity(vx, vy):
    """
    Calculate vorticity from velocity components
    """
    dx = 1
    d_kernel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    
    vy_x = ndimage.convolve(vy, d_kernel.T/dx, mode='reflect')
    vx_y = ndimage.convolve(vx, d_kernel/dx, mode='reflect')
    
    return vy_x - vx_y

def invariant2_factor(vx, vy, factor_x=1, factor_y=1):
    """
    Calculate Q-criterion (second invariant)
    """
    dx = 1
    d_kernel = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    
    # Velocity gradients
    ux = ndimage.convolve(vx, d_kernel/dx, mode='reflect') / factor_x
    uy = ndimage.convolve(vx, d_kernel.T/dx, mode='reflect') / factor_y
    vx = ndimage.convolve(vy, d_kernel/dx, mode='reflect') / factor_x
    vy = ndimage.convolve(vy, d_kernel.T/dx, mode='reflect') / factor_y
    
    # Q-criterion
    Q = -0.5 * (ux**2 + vy**2 + 2*uy*vx)
    
    return Q