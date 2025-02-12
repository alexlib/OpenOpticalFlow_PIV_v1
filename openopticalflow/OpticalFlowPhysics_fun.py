import numpy as np
from scipy.ndimage import convolve

def OpticalFlowPhysics_fun(I1: np.ndarray, I2: np.ndarray, lambda_1: float, lambda_2: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute the optical flow using the Horn-Schunck estimator.

    Args:
        I1 (np.ndarray): First input image.
        I2 (np.ndarray): Second input image.
        lambda_1 (float): Smoothness parameter for the initial field.
        lambda_2 (float): Smoothness parameter for the refined estimation.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: Velocity fields and error.
    """
    # Define convolution filters
    D1 = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # Partial derivative for x-axis
    F1 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]) / 4  # Average
    D2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])  # Partial derivative for y-axis

    # Compute partial derivatives
    Ix = convolve(I1, D1, mode='reflect')
    Iy = convolve(I1, D2, mode='reflect')
    It = convolve(I2, F1, mode='reflect')

    # Horn-Schunck estimator
    ux = (Ix * Iy) / (lambda_1 + (Ix**2 + Iy**2 + lambda_2 * It**2))
    uy = (Ix * It) / (lambda_1 + (Ix**2 + Iy**2 + lambda_2 * It**2))

    # Compute velocity magnitude
    vor = np.sqrt(ux**2 + uy**2)

    # Compute Horn-Schunck estimator for refined estimation
    ux_horn = convolve(ux, F1, mode='reflect')
    uy_horn = convolve(uy, F1, mode='reflect')

    # Compute error
    error1 = np.mean(np.abs(ux_horn) + np.abs(uy_horn))

    return ux, uy, vor, ux_horn, uy_horn, error1

def shift_image_fun_refine_1(ux_corr: np.ndarray, uy_corr: np.ndarray, Im1: np.ndarray, Im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shift the image Im1 based on the velocity field (ux_corr, uy_corr) and compute the velocity difference for iterative correction.

    Args:
        ux_corr (np.ndarray): Velocity field in the x-direction.
        uy_corr (np.ndarray): Velocity field in the y-direction.
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Shifted image Im1, and velocity differences uxI, uyI.
    """
    rows, cols = Im1.shape
    flow = np.stack((ux_corr, uy_corr), axis=-1)
    Im1_shift = convolve(Im1, flow, mode='reflect')

    flow_inv = np.stack((-ux_corr, -uy_corr), axis=-1)
    Im2_shift = convolve(Im2, flow_inv, mode='reflect')

    uxI, uyI = OpticalFlowPhysics_fun(Im1_shift, Im2_shift, lambda_1, lambda_2)[0:2]

    return Im1_shift, uxI, uyI

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
    ux0, uy0, vor, ux_horn, uy_horn, error1 = OpticalFlowPhysics_fun(I_region1, I_region2, lambda_1, lambda_2)

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
        ux_corr, uy_corr, vor, ux_horn, uy_horn, error2 = OpticalFlowPhysics_fun(Im1_shift, Im2, lambda_1, lambda_2)

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
