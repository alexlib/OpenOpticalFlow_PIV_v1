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

    Note: This is a simplified version. For a more accurate implementation,
    use the dedicated invariant2_factor module.

    Args:
        vx (np.ndarray): x-component of velocity field
        vy (np.ndarray): y-component of velocity field
        factor_x (float): Conversion factor from pixel to m (m/pixel) in x
        factor_y (float): Conversion factor from pixel to m (m/pixel) in y

    Returns:
        np.ndarray: Q-criterion field
    """
    # Import the dedicated implementation
    from openopticalflow.invariant2_factor import invariant2_factor as invariant2_factor_dedicated

    # Use the dedicated implementation
    return invariant2_factor_dedicated(vx, vy, factor_x, factor_y)