import numpy as np
from scipy import ndimage

def generate_invmatrix(i: np.ndarray, alpha: float, h: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate inverse matrix for Liu-Shen optical flow estimation.
    Matches MATLAB implementation.

    Args:
        i: Input image
        alpha: Regularization parameter
        h: Spatial step

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Inverse matrix components (b11, b12, b22)
    """
    # Define kernels exactly as in comparison version
    d = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2
    m = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    d2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
    h_kernel = np.ones((3, 3))

    # Calculate matrix components using ndimage.convolve with reflect mode
    cmtx = ndimage.convolve(np.ones_like(i), h_kernel/(h*h), mode='reflect')

    # Calculate A11, A22, A12 exactly as in comparison version
    a11 = i * (ndimage.convolve(i, d2/(h*h), mode='reflect') - 2*i/(h*h)) - alpha*cmtx
    a22 = i * (ndimage.convolve(i, d2.T/(h*h), mode='reflect') - 2*i/(h*h)) - alpha*cmtx
    a12 = i * ndimage.convolve(i, m/(h*h), mode='reflect')

    # Calculate determinant
    det_a = a11*a22 - a12*a12
    
    # Add epsilon for numerical stability
    epsilon = 1e-10
    
    # Initialize output arrays
    b11 = np.zeros_like(det_a)
    b12 = np.zeros_like(det_a)
    b22 = np.zeros_like(det_a)
    
    # Calculate inverse matrix components only where determinant is valid
    valid_mask = np.abs(det_a) > epsilon
    b11[valid_mask] = a22[valid_mask] / det_a[valid_mask]
    b12[valid_mask] = -a12[valid_mask] / det_a[valid_mask]
    b22[valid_mask] = a11[valid_mask] / det_a[valid_mask]
    
    # Handle invalid determinants
    invalid_mask = ~valid_mask
    if np.any(invalid_mask):
        if np.any(valid_mask):
            scale = 0.5 * (np.mean(np.abs(b11[valid_mask])) + np.mean(np.abs(b22[valid_mask])))
        else:
            scale = 1.0
        b11[invalid_mask] = scale
        b22[invalid_mask] = scale
        b12[invalid_mask] = 0.0

    return b11, b12, b22
