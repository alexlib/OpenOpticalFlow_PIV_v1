import numpy as np

def safe_mean(arr, **kwargs):
    """
    Calculate the mean of an array, safely handling empty arrays.
    
    Args:
        arr: Input array
        **kwargs: Additional arguments to pass to np.mean
        
    Returns:
        Mean of the array, or 0 if the array is empty
    """
    if arr.size == 0:
        return 0
    return np.mean(arr, **kwargs)

def safe_divide(a, b):
    """
    Safely divide two arrays, handling division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        
    Returns:
        Result of a/b, with 0 where b is 0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, np.ndarray):
            result[np.isinf(result)] = 0
            result[np.isnan(result)] = 0
        else:
            if np.isinf(result) or np.isnan(result):
                result = 0
        return result
