import numpy as np
from scipy.ndimage import convolve
from typing import Tuple

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

    # Horn-Schunck estimator with safe division
    denominator = lambda_1 + (Ix**2 + Iy**2 + lambda_2 * It**2)

    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ux = np.divide(Ix * Iy, denominator)
        uy = np.divide(Ix * It, denominator)

    # Replace NaN and inf values with zeros
    ux = np.nan_to_num(ux, nan=0.0, posinf=0.0, neginf=0.0)
    uy = np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute velocity magnitude
    vor = np.sqrt(ux**2 + uy**2)
    vor = np.nan_to_num(vor, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute Horn-Schunck estimator for refined estimation
    ux_horn = convolve(ux, F1, mode='reflect')
    uy_horn = convolve(uy, F1, mode='reflect')

    # Compute error
    error1 = np.mean(np.abs(ux_horn) + np.abs(uy_horn))

    return ux, uy, vor, ux_horn, uy_horn, error1

# The shift_image_fun_refine_1 function is now in its own file

# This file contains only the OpticalFlowPhysics_fun function
# No main function is needed as this is imported as a module
