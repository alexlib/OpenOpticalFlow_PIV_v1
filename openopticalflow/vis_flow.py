import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from shift_image_fun_refine_1 import shift_image_fun_refine_1

def vis_flow(Vx, Vy, gx=30, offset=1, mag=1, col='b'):
    """
    Visualize the velocity field and streamlines.

    Args:
        Vx (np.ndarray): Velocity field in x-direction.
        Vy (np.ndarray): Velocity field in y-direction.
        gx (int): Grid spacing for streamlines.
        offset (int): Offset for streamlines.
        mag (int): Magnification factor for velocity vectors.
        col (str): Color for streamlines.

    Returns:
        None
    """
    # Calculate the velocity magnitude
    Vmag = np.sqrt(Vx**2 + Vy**2)

    # Normalize the velocity vectors
    Vx = Vx / Vmag
    Vy = Vy / Vmag

    # Create a grid for streamlines
    x = np.arange(0, Vx.shape[1], gx)
    y = np.arange(0, Vx.shape[0], gx)
    x, y = np.meshgrid(x, y)

    # Calculate the velocity vectors for streamlines
    u = Vx[::gx, ::gx]
    v = Vy[::gx, ::gx]

    # Plot the velocity field
    plt.figure(figsize=(10, 8))
    plt.quiver(x, y, u, v, scale=mag, angles='xy', scale_units='xy', color=col)
    plt.title('Velocity Field')
    plt.axis('equal')

    # Plot streamlines
    plt.streamplot(x, y, u, v, density=[0.5, 1], color=col)
    plt.title('Streamlines')
    plt.axis('equal')

    plt.show()

def main():
    """
    Main program for extraction of velocity field from a pair of flow visualization images.
    """
    # Read a pair of images
    Im1 = cv2.imread('White_oval_1.tif', cv2.IMREAD_GRAYSCALE)
    Im2 = cv2.imread('White_Oval_2.tif', cv2.IMREAD_GRAYSCALE)

    # Set the parameters for optical flow computation
    lambda_1 = 20  # Horn-Schunck estimator for initial field
    lambda_2 = 2000  # Liu-Shen estimator for refined estimation
    no_iteration = 1  # Number of iterations in the coarse-to-fine iterative process
    scale_im = 0.5  # Scale factor for downsizing images
    size_average = 0  # Size for averaging to bypass local illumination intensity adjustment
    size_filter = 4  # Gaussian filter size for removing random noise
    index_region = 1  # Select a region for processing (1) or process the whole image (0)

    # Select a region of interest for diagnostics
    if index_region == 1:
        plt.imshow(Im1, cmap='gray')
        plt.colorbar()
        plt.axis('image')
        plt.show()

        xy = plt.ginput(2)
        x1, x2 = int(np.floor(min(xy[:, 0]))), int(np.floor(max(xy[:, 0])))
        y1, y2 = int(np.floor(min(xy[:, 1]))), int(np.floor(max(xy[:, 1])))
        Im1 = Im1[y1:y2, x1:x2]
        Im2 = Im2[y1:y2, x1:x2]
    elif index_region == 0:
        pass

    Im1_original = Im1.copy()
    Im2_original = Im2.copy()

    # Correct the global and local intensity change in images
    window_shifting = np.array([1, Im1.shape[0], 1, Im1.shape[1]])
    Im1, Im2 = correction_illumination(Im1, Im2, window_shifting, size_average)

    # Pre-process for reducing random noise and downsampling images if displacements are large
    Im1, Im2 = pre_processing_a(Im1, Im2, scale_im, size_filter)

    I_region1 = Im1.copy()
    I_region2 = Im2.copy()

    # Initial optical flow calculation for a coarse-grained velocity field
    ux0, uy0, vor, ux_horn, uy_horn, error1 = optical_flow_physics_fun(I_region1, I_region2, lambda_1, lambda_2)

    # Generate the shifted image from Im1 based on the initial coarse-grained velocity field (ux0, uy0)
    Im1 = cv2.convertScaleAbs(Im1_original)
    Im2 = cv2.convertScaleAbs(Im2_original)

    ux_corr = ux0.copy()
    uy_corr = uy0.copy()

    # Estimate the displacement vector and make correction in iterations
    k = 1
    while k <= no_iteration:
        Im1_shift, uxI, uyI = shift_image_fun_refine_1(ux_corr, uy_corr, Im1, Im2)

        # Calculation of correction of the optical flow
        ux_corr, uy_corr, vor, ux_horn, uy_horn, error2 = optical_flow_physics_fun(Im1_shift, Im2, lambda_1, lambda_2)

        k += 1

    # Refined velocity field
    ux = ux_corr
    uy = uy_corr

    # Show the images and processed results
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(Im1_original, cmap='gray')
    plt.title('Original Image 1')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(Im2_original, cmap='gray')
    plt.title('Original Image 2')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.quiver(ux, uy, scale=1, angles='xy', scale_units='xy')
    plt.title('Velocity Field')
    plt.axis('equal')

    plt.subplot(2, 2, 4)
    plt.imshow(vor, cmap='jet')
    plt.title('Velocity Magnitude')
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    main()
