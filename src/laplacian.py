import numpy as np
from scipy.ndimage import convolve

def laplacian(u, h):
    """
    u: Given image (numpy array)
    h: step size
    """
    H = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # You could also choose (which seems more natural)
    # H = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    delu = -u * convolve(np.ones_like(u), H / (h * h), mode='constant', cval=0.0) + convolve(u, H / (h * h), mode='constant', cval=0.0)
    return delu