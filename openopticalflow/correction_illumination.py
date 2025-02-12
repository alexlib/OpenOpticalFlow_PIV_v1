import numpy as np

def correction_illumination(Im1: np.ndarray, Im2: np.ndarray, window_shifting: np.ndarray, size_average: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adjust the illumination of images Im1 and Im2 based on a window shifting and averaging process.

    Args:
        Im1 (np.ndarray): First input image.
        Im2 (np.ndarray): Second input image.
        window_shifting (np.ndarray): Window shifting parameters.
        size_average (int): Size for averaging.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Adjusted images Im1 and Im2.
    """
    # Convert images to double precision
    Im1 = Im1.astype(np.float64)
    Im2 = Im2.astype(np.float64)

    # Adjust the overall illumination change
    x3, y3, x4, y4 = window_shifting[:4]
    I1_mean = np.mean(Im1[y3:y4, x3:x4])
    I2_mean = np.mean(Im2[y3:y4, x3:x4])
    R12 = I1_mean / I2_mean
    Im2 = R12 * Im2

    # Normalize the intensity for Im2 to eliminate the local change of illumination light
    if size_average > 0:
        N = size_average
        h = np.ones((N, N)) / (N * N)
        I12F = np.apply_over_axes(lambda x: np.convolve(x, h, mode='same'), Im1) - np.apply_over_axes(lambda x: np.convolve(x, h, mode='same'), Im2)
        Im2 = Im2 + I12F
    else:
        Im2 = Im2

    return Im1, Im2









