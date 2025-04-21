import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from openopticalflow.vis_flow import vis_flow, plot_streamlines

def plots_set_1(im1, im2, ux, uy, Im1=None, Im2=None, ux_full=None, uy_full=None, show_plot=True):
    """
    Create a comprehensive set of plots for optical flow analysis.

    This function creates a set of plots to visualize optical flow results, including:
    1. Original images
    2. Velocity field visualization
    3. Animation showing motion between images

    Parameters:
        im1 (numpy.ndarray): First image (processed/downsampled version)
        im2 (numpy.ndarray): Second image (processed/downsampled version)
        ux (numpy.ndarray): x-component of velocity field
        uy (numpy.ndarray): y-component of velocity field
        Im1 (numpy.ndarray, optional): First image (original/full resolution)
        Im2 (numpy.ndarray, optional): Second image (original/full resolution)
        ux_full (numpy.ndarray, optional): x-component of velocity field (full resolution)
        uy_full (numpy.ndarray, optional): y-component of velocity field (full resolution)
        show_plot (bool): Whether to call plt.show() (default: True)

    Returns:
        list: List of created figure objects
    """
    figures = []

    # If full resolution images not provided, use the processed ones
    if Im1 is None:
        Im1 = im1
    if Im2 is None:
        Im2 = im2
    if ux_full is None:
        ux_full = ux
    if uy_full is None:
        uy_full = uy

    # Figure 1: Original images
    fig1 = plt.figure(figsize=(12, 5))
    figures.append(fig1)

    plt.subplot(121)
    plt.imshow(im1, cmap='gray')
    plt.title('First Image')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(im2, cmap='gray')
    plt.title('Second Image')
    plt.colorbar()

    # Figure 2: Velocity field visualization
    fig2 = plt.figure(figsize=(12, 10))
    figures.append(fig2)

    plt.subplot(221)
    # Use our improved vis_flow function
    ax = vis_flow(ux, uy, gx=25, mag=2, color='red', show_plot=False)
    plt.title('Velocity Field')

    plt.subplot(222)
    # Use our improved plot_streamlines function
    ax = plot_streamlines(ux, uy, density=1.5, show_plot=False)
    plt.title('Streamlines')

    plt.subplot(223)
    # Calculate velocity magnitude
    vel_mag = np.sqrt(ux**2 + uy**2)
    plt.imshow(vel_mag, cmap='jet')
    plt.colorbar(label='Velocity Magnitude')
    plt.title('Velocity Magnitude')

    plt.subplot(224)
    # Calculate vorticity
    from scipy.ndimage import convolve
    kernel_x = np.array([[-1, 0, 1]]) / 2
    kernel_y = np.array([[-1], [0], [1]]) / 2
    duy_dx = convolve(uy, kernel_x, mode='reflect')
    dux_dy = convolve(ux, kernel_y, mode='reflect')
    vorticity = duy_dx - dux_dy
    plt.imshow(vorticity, cmap='RdBu_r')
    plt.colorbar(label='Vorticity')
    plt.title('Vorticity')

    # Figure 3: Animation showing motion between images
    fig3 = plt.figure(figsize=(8, 8))
    figures.append(fig3)
    ax = plt.subplot(111)
    plt.title('Image Sequence Animation')

    # Create animation function
    frames = [im1, im2]
    im = plt.imshow(frames[0], cmap='gray', animated=True)

    def update(frame):
        im.set_array(frames[frame])
        return [im]

    ani = FuncAnimation(fig3, update, frames=2, interval=500, blit=True)

    if show_plot:
        plt.tight_layout()
        plt.show()

    return figures

# Example usage
if __name__ == "__main__":
    # Create sample data
    size = 100
    # Create sample images
    im1 = np.random.rand(size, size)
    im2 = np.random.rand(size, size)

    # Create sample flow field (rotating vortex)
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    ux = -y
    uy = x

    # Create plots
    figures = plots_set_1(im1, im2, ux, uy)

    plt.show()
