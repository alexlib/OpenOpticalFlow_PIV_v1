import numpy as np

def rescaling_intensity(Im1: np.ndarray, Im2: np.ndarray, max_intensity_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rescale the intensity of images Im1 and Im2 to a maximum intensity value.

    Args:
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.
        max_intensity_value (float): Maximum intensity value for rescaling.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Rescaled images Im1 and Im2.
    """
    Imax1 = np.max(Im1)
    Imin1 = np.min(Im1)
    Im1a = (Im1 - Imin1) / (Imax1 - Imin1)

    Imax2 = np.max(Im2)
    Imin2 = np.min(Im2)
    Im2a = (Im2 - Imin2) / (Imax2 - Imin2)

    Im1 = Im1a * max_intensity_value
    Im2 = Im2a * max_intensity_value

    return Im1, Im2
