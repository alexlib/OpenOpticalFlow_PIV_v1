import numpy as np
from typing import Tuple

def rescaling_intensity(Im1: np.ndarray, Im2: np.ndarray, max_intensity_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rescale the intensity of images Im1 and Im2 to a maximum intensity value.

    This function normalizes each image to the range [0, 1] and then scales
    to the specified maximum intensity value. It handles constant-valued images
    by returning the constant value directly.

    Args:
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.
        max_intensity_value (float): Maximum intensity value for rescaling.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rescaled images Im1 and Im2 as float arrays.
    """
    # Make copies to avoid modifying the input arrays
    I1 = Im1.copy().astype(float)
    I2 = Im2.copy().astype(float)

    # First image
    Imax1 = np.max(I1)
    Imin1 = np.min(I1)
    if Imax1 > Imin1:  # Check for non-constant image
        I1 = (I1 - Imin1) / (Imax1 - Imin1) * max_intensity_value
    else:  # Constant image - avoid division by zero
        I1 = np.ones_like(I1) * max_intensity_value / 2  # Set to half the max value

    # Second image
    Imax2 = np.max(I2)
    Imin2 = np.min(I2)
    if Imax2 > Imin2:  # Check for non-constant image
        I2 = (I2 - Imin2) / (Imax2 - Imin2) * max_intensity_value
    else:  # Constant image - avoid division by zero
        I2 = np.ones_like(I2) * max_intensity_value / 2  # Set to half the max value

    return I1, I2
