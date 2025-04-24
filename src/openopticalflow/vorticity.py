import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from typing import Optional

def vorticity(vx: np.ndarray, vy: np.ndarray, 
             factor_x: Optional[float] = None, 
             factor_y: Optional[float] = None,
             smooth: bool = False, 
             sigma: float = 0.5,
             high_order: bool = True) -> np.ndarray:
    """
    Calculate vorticity from velocity components with optional scaling and high-order derivatives.

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
        high_order (bool): Use high-order finite differences when possible

    Returns:
        np.ndarray: 2D array of vorticity values (omega). Positive values indicate counterclockwise rotation.

    Notes:
        - When high_order=True:
            * Uses 4th-order central differences for interior points when grid size allows (>4x4)
            * Uses 2nd-order central differences for near-boundary points
            * Uses 2nd-order one-sided differences for edge and corner points
        - When high_order=False:
            * Uses 2nd-order central differences with convolution
        - For a rigid body rotation with angular velocity ω:
            * Without scaling factors: theoretical vorticity equals 2ω
            * With scaling factors: theoretical vorticity equals 2ω/(factor_x*factor_y)
    """
    # Apply Gaussian smoothing if requested
    if smooth:
        vx = gaussian_filter(vx, sigma=sigma)
        vy = gaussian_filter(vy, sigma=sigma)

    if high_order:
        # Get the shape of the velocity field
        ny, nx = vx.shape

        # Initialize derivative arrays
        dvdx = np.zeros_like(vx)
        dudy = np.zeros_like(vy)

        if nx > 4 and ny > 4:  # Need at least 5 points for 4th order
            # Interior points (using 4th-order central differences)
            dvdx[2:-2, 2:-2] = (-vy[2:-2, 4:] + 8*vy[2:-2, 3:-1] - 8*vy[2:-2, 1:-3] + vy[2:-2, 0:-4]) / 12.0
            dudy[2:-2, 2:-2] = (-vx[4:, 2:-2] + 8*vx[3:-1, 2:-2] - 8*vx[1:-3, 2:-2] + vx[0:-4, 2:-2]) / 12.0

            # Near-boundary points (using 2nd-order central differences)
            dvdx[1:-1, [1,-2]] = (vy[1:-1, 2:-1:nx-3] - vy[1:-1, 0:-3:nx-3]) / 2.0
            dvdx[[1,-2], 1:-1] = (vy[2:-1:ny-3, 1:-1] - vy[0:-3:ny-3, 1:-1]) / 2.0
            dudy[1:-1, [1,-2]] = (vx[1:-1, 2:-1:nx-3] - vx[1:-1, 0:-3:nx-3]) / 2.0
            dudy[[1,-2], 1:-1] = (vx[2:-1:ny-3, 1:-1] - vx[0:-3:ny-3, 1:-1]) / 2.0
        else:
            # Use 2nd-order central differences for all interior points if grid is small
            dvdx[1:-1, 1:-1] = (vy[1:-1, 2:] - vy[1:-1, :-2]) / 2.0
            dudy[1:-1, 1:-1] = (vx[2:, 1:-1] - vx[:-2, 1:-1]) / 2.0

        # Edge points (using one-sided differences)
        dvdx[1:-1, 0] = (-3*vy[1:-1, 0] + 4*vy[1:-1, 1] - vy[1:-1, 2]) / 2.0
        dvdx[1:-1, -1] = (3*vy[1:-1, -1] - 4*vy[1:-1, -2] + vy[1:-1, -3]) / 2.0
        dudy[0, 1:-1] = (-3*vx[0, 1:-1] + 4*vx[1, 1:-1] - vx[2, 1:-1]) / 2.0
        dudy[-1, 1:-1] = (3*vx[-1, 1:-1] - 4*vx[-2, 1:-1] + vx[-3, 1:-1]) / 2.0

        # Corners (using 2nd-order one-sided differences)
        dvdx[0, 0] = (-3*vy[0, 0] + 4*vy[0, 1] - vy[0, 2]) / 2.0
        dudy[0, 0] = (-3*vx[0, 0] + 4*vx[1, 0] - vx[2, 0]) / 2.0
        dvdx[0, -1] = (3*vy[0, -1] - 4*vy[0, -2] + vy[0, -3]) / 2.0
        dudy[0, -1] = (-3*vx[0, -1] + 4*vx[1, -1] - vx[2, -1]) / 2.0
        dvdx[-1, 0] = (-3*vy[-1, 0] + 4*vy[-1, 1] - vy[-1, 2]) / 2.0
        dudy[-1, 0] = (3*vx[-1, 0] - 4*vx[-2, 0] + vx[-3, 0]) / 2.0
        dvdx[-1, -1] = (3*vy[-1, -1] - 4*vy[-1, -2] + vy[-1, -3]) / 2.0
        dudy[-1, -1] = (3*vx[-1, -1] - 4*vx[-2, -1] + vx[-3, -1]) / 2.0

    else:
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
