import numpy as np
from scipy.ndimage import convolve

def vorticity_factor(vx: np.ndarray, vy: np.ndarray, factor_x: float, factor_y: float) -> np.ndarray:
    """
    Calculate vorticity from velocity components with scaling factors.

    Vorticity is a measure of local rotation in a fluid flow, calculated as the curl of the velocity field.
    For a 2D flow, vorticity is a scalar field representing the rotation around the z-axis.

    Parameters:
        vx (np.ndarray): x-component of velocity field
        vy (np.ndarray): y-component of velocity field
        factor_x (float): Converting factor from pixel to m (m/pixel) in x
        factor_y (float): Converting factor from pixel to m (m/pixel) in y

    Returns:
        np.ndarray: Vorticity field (omega)
    """
    # Optional smoothing (commented out by default)
    # kernel = np.ones((5, 5)) / 25
    # vx = convolve(vx, kernel, mode='reflect')
    # vy = convolve(vy, kernel, mode='reflect')

    # Define derivative kernel
    dx = 1
    d_kernel = np.array([[0, -1, 0],
                        [0, 0, 0],
                        [0, 1, 0]]) / 2

    # Calculate velocity gradients
    # Note: d_kernel.T for x-derivatives, d_kernel for y-derivatives
    vy_x = convolve(vy, d_kernel.T/dx, mode='reflect')
    vx_y = convolve(vx, d_kernel/dx, mode='reflect')

    # Calculate vorticity with scaling factors
    # For 2D flow, vorticity = dv/dx - du/dy
    # Note: The sign depends on the coordinate system convention
    # For a counterclockwise rotation, vorticity should be positive
    omega = (vy_x/factor_x - vx_y/factor_y)

    # Note: For a rigid body rotation with angular velocity ω,
    # the theoretical vorticity should be 2ω.
    # With scaling factors, this becomes 2ω/(factor_x*factor_y)

    return omega

# Example usage
if __name__ == "__main__":
    # Create a sample flow field (rotating vortex)
    size = 100
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    vx = -y
    vy = x

    # Set conversion factors
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel

    # Calculate vorticity
    omega = vorticity_factor(vx, vy, factor_x, factor_y)

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
    theoretical = 2 * angular_velocity * np.ones_like(vx) / (factor_x * factor_y)
    plt.imshow(theoretical, cmap='RdBu_r')
    plt.colorbar(label='Theoretical Vorticity')
    plt.title('Theoretical Vorticity')

    plt.tight_layout()
    plt.show()
