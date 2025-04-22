import numpy as np
import matplotlib.pyplot as plt

def vis_flow(vx, vy, gx=25, offset=0, mag=1, color='b', show_plot=True):
    """
    Visualize flow field using quiver plot with subsampling and normalized arrows.

    This function creates a quiver plot (vector field visualization) of the velocity field
    defined by vx and vy components. It subsamples the field to avoid overcrowding and
    normalizes the arrows for better visualization.

    Parameters:
        vx (numpy.ndarray): x-component of velocity field
        vy (numpy.ndarray): y-component of velocity field
        gx (int): Grid spacing parameter. Controls the density of arrows.
                 Higher values result in fewer arrows.
        offset (int): Offset for sampling grid points
        mag (float): Magnitude scaling factor. Higher values result in longer arrows.
        color (str): Color of the arrows
        show_plot (bool): Whether to call plt.show() (default: True)

    Returns:
        matplotlib.axes.Axes: Axes object containing the quiver plot
    """
    sy, sx = vx.shape

    if gx == 0:
        jmp = 1
    else:
        jmp = max(1, sx // gx)

    indx = np.arange(offset, sx, jmp)
    indy = np.arange(offset, sy, jmp)

    X, Y = np.meshgrid(indx, indy)

    # Extract velocity components at the sampled points
    U = vx[indy][:, indx]
    V = vy[indy][:, indx]

    # Handle NaN values
    mask = ~(np.isnan(U) | np.isnan(V))
    if not mask.any():
        U[0,0] = 1
        V[0,0] = 0
        X[0,0] = 1
        Y[0,0] = 1
        mask[0,0] = True

    # Calculate the scale for arrow lengths
    vel_magnitude = np.sqrt(U[mask]**2 + V[mask]**2)

    # Calculate scale for arrow lengths
    if vel_magnitude.size > 0 and np.mean(vel_magnitude) > 0:
        # Calculate scale - smaller values make longer arrows
        scale_factor = 1.0 / (np.max(vel_magnitude) * mag)
    else:
        scale_factor = 1.0

    # Plot quiver with normalized arrows
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.quiver(X[mask], Y[mask], U[mask], V[mask],
              scale=scale_factor,    # Apply calculated scale
              scale_units='xy',      # Use xy coordinate system for scaling
              angles='xy',           # Use xy coordinate system for angles
              width=0.003,           # Normalized arrow width (smaller = thinner)
              headwidth=4,           # Head width relative to shaft (larger = wider)
              headlength=6,          # Head length relative to shaft (larger = longer)
              headaxislength=5,      # Head length at shaft intersection
              color=color,
              pivot='mid')

    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)

    if show_plot:
        plt.show()

    return ax

def plot_streamlines(vx, vy, gx=25, offset=0, density=2, color='blue', show_plot=True):
    """
    Visualize flow field using streamlines.

    Parameters:
        vx (numpy.ndarray): x-component of velocity field
        vy (numpy.ndarray): y-component of velocity field
        gx (int): Grid spacing parameter
        offset (int): Offset for sampling grid points
        density (float or tuple): Controls the density of streamlines
        color (str): Color of the streamlines
        show_plot (bool): Whether to call plt.show() (default: True)

    Returns:
        matplotlib.axes.Axes: Axes object containing the streamlines plot
    """
    sy, sx = vx.shape

    # Create grid for streamlines
    y, x = np.meshgrid(np.arange(sy), np.arange(sx), indexing='ij')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.streamplot(x, y, vx, vy, density=density, color=color)

    ax.set_xlim(0, sx)
    ax.set_ylim(0, sy)
    ax.invert_yaxis()  # Invert y-axis to match image coordinates

    if show_plot:
        plt.show()

    return ax

# Example usage
if __name__ == "__main__":
    # Create a sample flow field (rotating vortex)
    size = 100
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    vx = -y
    vy = x

    # Visualize with quiver plot
    ax1 = vis_flow(vx, vy, gx=10, mag=2, color='red')
    ax1.set_title('Velocity Field (Quiver Plot)')

    # Visualize with streamlines
    ax2 = plot_streamlines(vx, vy, density=1.5)
    ax2.set_title('Velocity Field (Streamlines)')

    plt.show()
