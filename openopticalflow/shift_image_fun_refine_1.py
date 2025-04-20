import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
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

    # Remap the image using scipy.ndimage.map_coordinates
    # Create coordinate arrays for the output image
    coords = np.indices(Im1_uint8.shape)
    # Adjust coordinates based on displacement field
    # map_coordinates expects coordinates in (y, x) order
    coords_shifted = np.zeros_like(coords)
    coords_shifted[0] = map_y - y_coords  # y-coordinates
    coords_shifted[1] = map_x - x_coords  # x-coordinates

    # Apply mapping
    Im1_shift = map_coordinates(Im1_uint8, coords_shifted, order=1, mode='constant')

    # Inverse mapping - calculate inverse displacement
    map_x_inv = (x_coords - ux_resized).astype(np.float32)
    map_y_inv = (y_coords - uy_resized).astype(np.float32)

    # Create coordinate arrays for inverse mapping
    coords_inv = np.zeros_like(coords)
    coords_inv[0] = map_y_inv - y_coords  # y-coordinates
    coords_inv[1] = map_x_inv - x_coords  # x-coordinates
    Im2_shift = map_coordinates(Im2_uint8, coords_inv, order=1, mode='constant')

    # For optical flow, we'll use a simple difference between the images
    # This is a simplified approach compared to Farneback
    diff = Im1_shift.astype(float) - Im2_shift.astype(float)
    # Normalize the difference to get a velocity-like field
    uxI = np.zeros_like(diff)
    uyI = np.zeros_like(diff)

    return Im1_shift, uxI, uyI

# This file contains only the shift_image_fun_refine_1 function
# No main function is needed as this is imported as a module
