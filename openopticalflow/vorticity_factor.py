import numpy as np
from scipy.ndimage import gaussian_filter, convolve

def vorticity_factor(vx, vy, factor_x=None, factor_y=None, smooth=False, sigma=0.5):
    """
    Calculate vorticity from velocity components with optional scaling.
    
    Vorticity is a measure of local rotation in a fluid flow, calculated as the curl of the velocity field.
    For a 2D flow, vorticity is a scalar field representing the rotation around the z-axis.
    
    Parameters:
        vx (np.ndarray): 2D array of x-component velocities
        vy (np.ndarray): 2D array of y-component velocities
        factor_x (float, optional): Converting factor from pixel to m (m/pixel) in x. 
                                  If None, no scaling is applied.
        factor_y (float, optional): Converting factor from pixel to m (m/pixel) in y. 
                                  If None, no scaling is applied.
        smooth (bool): Apply Gaussian smoothing to velocity field before calculation
        sigma (float): Standard deviation for Gaussian smoothing kernel
    
    Returns:
        np.ndarray: 2D array of vorticity values (omega). Positive values indicate counterclockwise rotation.
    """
    # Apply Gaussian smoothing if requested
    if smooth:
        vx = gaussian_filter(vx, sigma=sigma)
        vy = gaussian_filter(vy, sigma=sigma)
    
    # Simple 2nd-order central differences using convolution
    d_kernel = np.array([[0, -1, 0],
                       [0, 0, 0],
                       [0, 1, 0]]) / 2
    
    dvdx = convolve(vy, d_kernel.T, mode='reflect')
    dudy = convolve(vx, d_kernel, mode='reflect')
    
    # Calculate vorticity with optional scaling factors
    if factor_x is not None and factor_y is not None:
        omega = dvdx/factor_x - dudy/factor_y
    else:
        omega = dvdx - dudy
    
    return omega
