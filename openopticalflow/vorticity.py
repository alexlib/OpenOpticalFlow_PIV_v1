import numpy as np

def vorticity(vx, vy, smooth=False, sigma=0.5):
    """
    Calculate vorticity from velocity components using high-order finite differences.

    Vorticity is a measure of local rotation in a fluid flow, calculated as the curl of the velocity field.
    For a 2D flow, vorticity is a scalar field representing the rotation around the z-axis.

    Parameters:
        vx (np.ndarray): x-component of velocity field
        vy (np.ndarray): y-component of velocity field
        smooth (bool, optional): Whether to apply Gaussian smoothing to the velocity field. Default is False.
        sigma (float, optional): Standard deviation for Gaussian smoothing. Default is 0.5.

    Returns:
        np.ndarray: Vorticity field (omega)
    """
    # Get the shape of the velocity field
    ny, nx = vx.shape

    # Apply Gaussian smoothing if requested
    if smooth:
        from scipy.ndimage import gaussian_filter
        vx = gaussian_filter(vx, sigma=sigma)
        vy = gaussian_filter(vy, sigma=sigma)

    # Initialize derivative arrays
    dvdx = np.zeros_like(vx)
    dudy = np.zeros_like(vy)

    # Use 4th-order central differences for interior points
    # Formula: df/dx = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    if nx > 4 and ny > 4:  # Need at least 5 points for 4th order
        # Interior points (using 4th-order central differences)
        dvdx[2:-2, 2:-2] = (-vy[2:-2, 4:] + 8*vy[2:-2, 3:-1] - 8*vy[2:-2, 1:-3] + vy[2:-2, 0:-4]) / 12.0
        dudy[2:-2, 2:-2] = (-vx[4:, 2:-2] + 8*vx[3:-1, 2:-2] - 8*vx[1:-3, 2:-2] + vx[0:-4, 2:-2]) / 12.0

        # Near-boundary points (using 2nd-order central differences)
        # One point away from boundary
        dvdx[1:-1, [1,-2]] = (vy[1:-1, 2:-1:nx-3] - vy[1:-1, 0:-3:nx-3]) / 2.0
        dvdx[[1,-2], 1:-1] = (vy[2:-1:ny-3, 1:-1] - vy[0:-3:ny-3, 1:-1]) / 2.0
        dudy[1:-1, [1,-2]] = (vx[1:-1, 2:-1:nx-3] - vx[1:-1, 0:-3:nx-3]) / 2.0
        dudy[[1,-2], 1:-1] = (vx[2:-1:ny-3, 1:-1] - vx[0:-3:ny-3, 1:-1]) / 2.0
    else:
        # Use 2nd-order central differences for all interior points if grid is small
        dvdx[1:-1, 1:-1] = (vy[1:-1, 2:] - vy[1:-1, :-2]) / 2.0
        dudy[1:-1, 1:-1] = (vx[2:, 1:-1] - vx[:-2, 1:-1]) / 2.0

    # Edge points (using one-sided differences)
    # Left and right edges (excluding corners)
    dvdx[1:-1, 0] = (-3*vy[1:-1, 0] + 4*vy[1:-1, 1] - vy[1:-1, 2]) / 2.0  # 2nd-order forward
    dvdx[1:-1, -1] = (3*vy[1:-1, -1] - 4*vy[1:-1, -2] + vy[1:-1, -3]) / 2.0  # 2nd-order backward

    # Top and bottom edges (excluding corners)
    dudy[0, 1:-1] = (-3*vx[0, 1:-1] + 4*vx[1, 1:-1] - vx[2, 1:-1]) / 2.0  # 2nd-order forward
    dudy[-1, 1:-1] = (3*vx[-1, 1:-1] - 4*vx[-2, 1:-1] + vx[-3, 1:-1]) / 2.0  # 2nd-order backward

    # Corners (using 2nd-order one-sided differences)
    # Top-left corner
    dvdx[0, 0] = (-3*vy[0, 0] + 4*vy[0, 1] - vy[0, 2]) / 2.0
    dudy[0, 0] = (-3*vx[0, 0] + 4*vx[1, 0] - vx[2, 0]) / 2.0

    # Top-right corner
    dvdx[0, -1] = (3*vy[0, -1] - 4*vy[0, -2] + vy[0, -3]) / 2.0
    dudy[0, -1] = (-3*vx[0, -1] + 4*vx[1, -1] - vx[2, -1]) / 2.0

    # Bottom-left corner
    dvdx[-1, 0] = (-3*vy[-1, 0] + 4*vy[-1, 1] - vy[-1, 2]) / 2.0
    dudy[-1, 0] = (3*vx[-1, 0] - 4*vx[-2, 0] + vx[-3, 0]) / 2.0

    # Bottom-right corner
    dvdx[-1, -1] = (3*vy[-1, -1] - 4*vy[-1, -2] + vy[-1, -3]) / 2.0
    dudy[-1, -1] = (3*vx[-1, -1] - 4*vx[-2, -1] + vx[-3, -1]) / 2.0

    # Calculate vorticity
    # For 2D flow, vorticity = dv/dx - du/dy
    # Note: The sign depends on the coordinate system convention
    # For a counterclockwise rotation, vorticity should be positive
    omega = dvdx - dudy

    # Note: For a rigid body rotation with angular velocity ω,
    # the theoretical vorticity should be 2ω.

    return omega

# Example usage
if __name__ == "__main__":
    # Create a sample flow field (rotating vortex)
    size = 100
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    vx = -y
    vy = x

    # Calculate vorticity
    omega = vorticity(vx, vy)

    # Display results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.quiver(x[::5, ::5], y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
    plt.title('Velocity Field')
    plt.axis('equal')

    plt.subplot(132)
    plt.imshow(omega, cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title('Vorticity Field')

    plt.subplot(133)
    # Calculate theoretical vorticity for rigid body rotation
    # For this example, angular velocity = 1
    angular_velocity = 1.0
    theoretical = 2 * angular_velocity * np.ones_like(vx)
    plt.imshow(theoretical, cmap='RdBu_r')
    plt.colorbar(label='Theoretical Vorticity')
    plt.title('Theoretical Vorticity')

    plt.tight_layout()
    plt.show()
