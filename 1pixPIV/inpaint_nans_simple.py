import numpy as np
from scipy.interpolate import griddata

def inpaint_nans(array, method_num=None):
    """
    Simple function to replace NaNs with interpolated values using griddata
    
    Parameters:
        array: 2D numpy array with NaN values to be replaced
        method_num: Ignored, kept for compatibility
        
    Returns:
        2D numpy array with NaN values replaced by interpolated values
    """
    # Get array shape
    ny, nx = array.shape
    
    # Create meshgrid for all points
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    
    # Find NaN values
    mask = np.isnan(array)
    
    # If no NaNs, return the original array
    if not np.any(mask):
        return array
    
    # Get only valid values
    x1 = xx[~mask]
    y1 = yy[~mask]
    values = array[~mask]
    
    # Points where we need to interpolate
    x2 = xx[mask]
    y2 = yy[mask]
    
    # Interpolate using nearest neighbor
    result = array.copy()
    
    # If we have valid points, interpolate
    if len(values) > 0:
        result[mask] = griddata((x1, y1), values, (x2, y2), method='nearest')
    else:
        # If all values are NaN, fill with zeros
        result[mask] = 0
    
    return result
