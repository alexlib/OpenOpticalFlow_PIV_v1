import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import Tuple

# Import local modules
from .correction_illumination import correction_illumination
from .pre_processing_a import pre_processing_a
from .OpticalFlowPhysics_fun import OpticalFlowPhysics_fun

def shift_image_fun_refine_1(ux, uy, Im1, Im2):
    """
    Shift the image Im1 based on the velocity field (ux, uy) and compute the velocity difference for iterative correction.

    Args:
        ux (np.ndarray): Velocity field in x-direction.
        uy (np.ndarray): Velocity field in y-direction.
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Shifted image Im1, and velocity differences uxI, uyI.
    """
    m1, n1 = Im1.shape
    m2, n2 = Im2.shape
    # Create proper map for cv2.remap
    h, w = Im1.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Resize velocity field to match image dimensions if needed
    if ux.shape != Im1.shape:
        from scipy.ndimage import zoom
        zoom_factor_y = h / ux.shape[0]
        zoom_factor_x = w / ux.shape[1]
        ux_resized = zoom(ux, (zoom_factor_y, zoom_factor_x))
        uy_resized = zoom(uy, (zoom_factor_y, zoom_factor_x))
    else:
        ux_resized = ux
        uy_resized = uy

    # Add displacement to coordinates
    map_x = (x_coords + ux_resized).astype(np.float32)
    map_y = (y_coords + uy_resized).astype(np.float32)

    # Ensure images are uint8 for OpenCV
    # Handle NaN values and normalize to 0-255 range
    Im1_clean = Im1.copy()
    Im2_clean = Im2.copy()

    # Replace NaN values with 0
    Im1_clean[np.isnan(Im1_clean)] = 0
    Im2_clean[np.isnan(Im2_clean)] = 0

    # Normalize to 0-255 range if not already uint8
    if Im1_clean.dtype != np.uint8:
        Im1_min, Im1_max = np.min(Im1_clean), np.max(Im1_clean)
        if Im1_max > Im1_min:  # Avoid division by zero
            Im1_clean = ((Im1_clean - Im1_min) * 255 / (Im1_max - Im1_min)).astype(np.uint8)
        else:
            Im1_clean = np.zeros_like(Im1_clean, dtype=np.uint8)

    if Im2_clean.dtype != np.uint8:
        Im2_min, Im2_max = np.min(Im2_clean), np.max(Im2_clean)
        if Im2_max > Im2_min:  # Avoid division by zero
            Im2_clean = ((Im2_clean - Im2_min) * 255 / (Im2_max - Im2_min)).astype(np.uint8)
        else:
            Im2_clean = np.zeros_like(Im2_clean, dtype=np.uint8)

    Im1_uint8 = Im1_clean
    Im2_uint8 = Im2_clean

    # Remap the image
    Im1_shift = cv2.remap(Im1_uint8, map_x, map_y, cv2.INTER_LINEAR)

    # Inverse mapping
    map_x_inv = (x_coords - ux_resized).astype(np.float32)
    map_y_inv = (y_coords - uy_resized).astype(np.float32)
    Im2_shift = cv2.remap(Im2_uint8, map_x_inv, map_y_inv, cv2.INTER_LINEAR)

    uxI, uyI = cv2.calcOpticalFlowFarneback(Im1_shift, Im2_shift, None, 0.5, 3, 15, 3, 5, 1.2, 0)[..., 0], cv2.calcOpticalFlowFarneback(Im1_shift, Im2_shift, None, 0.5, 3, 15, 3, 5, 1.2, 0)[..., 1]

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
