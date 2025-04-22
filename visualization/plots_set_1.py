import numpy as np
import matplotlib.pyplot as plt
from visualization.vis_flow import vis_flow, plot_streamlines

def plots_set_1(I_region1, I_region2, ux0, uy0, Im1=None, Im2=None, ux=None, uy=None):
    """
    Create plots similar to MATLAB's plots_set_1.m script.

    Parameters:
        I_region1: Preprocessed first image
        I_region2: Preprocessed second image
        ux0: x-component of coarse-grained velocity field
        uy0: y-component of coarse-grained velocity field
        Im1: Original first image (optional)
        Im2: Original second image (optional)
        ux: x-component of refined velocity field (optional)
        uy: y-component of refined velocity field (optional)
    """
    # If original images not provided, use the processed ones
    if Im1 is None:
        Im1 = I_region1
    if Im2 is None:
        Im2 = I_region2
    if ux is None:
        ux = ux0
    if uy is None:
        uy = uy0

    # Show the pre-processed images in initial estimation
    plt.figure(1)
    plt.imshow(I_region1.astype(np.uint8), cmap='gray')
    plt.colorbar()
    plt.axis('image')
    plt.title('Downsampled Image 1')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')

    plt.figure(2)
    plt.imshow(I_region2.astype(np.uint8), cmap='gray')
    plt.colorbar()
    plt.axis('image')
    plt.title('Downsampled Image 2')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')

    # Plot initial velocity vector field and streamlines
    plt.figure(3)
    gx = 30
    offset = 1
    h = vis_flow(ux0, uy0, gx, offset, 3, 'm')
    plt.setp(h, color='red')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Coarse-Grained Velocity Field')

    # Plot streamlines
    plt.figure(4)
    m, n = ux0.shape
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    dn = 10
    dm = 10
    h = plt.streamplot(x, y, ux0, uy0, density=1.5, color='blue')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Coarse-Grained Streamlines')

    # Plot the original images
    plt.figure(10)
    plt.imshow(Im1, cmap='gray')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.title('Image 1')

    plt.figure(11)
    plt.imshow(Im2, cmap='gray')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.title('Image 2')

    # Plot refined velocity vector field
    plt.figure(12)
    gx = 50
    offset = 1
    h = vis_flow(ux, uy, gx, offset, 5, 'm')
    plt.setp(h, color='red')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Refined Velocity Field')

    # Plot streamlines
    plt.figure(13)
    m, n = ux.shape
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    dn = 10
    dm = 10
    h = plt.streamplot(x, y, ux, uy, density=1.5, color='blue')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Refined Streamlines')

# This is just a plotting script, similar to MATLAB scripts
# It doesn't run anything when imported
