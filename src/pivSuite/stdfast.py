"""
function [out] = stdfast(in)
    % computes root-mean-square (reprogramed, because std in Matlab is somewhat slow due to some additional tests)
    in = reshape(in,1,numel(in));
    notnan = ~isnan(in);
    n = sum(notnan);
    in(~notnan) = 0;
    avg = sum(in)/n;
    out = sqrt(sum(((in - avg).*notnan).^2)/(n-0)); % there should be -1 in the denominator for true std
end
"""
import numpy as np

def stdfast(in_array):
    """
    Computes root-mean-square (reprogrammed, because std in Matlab is somewhat slow due to some additional tests).
    """
    in_array = in_array.flatten()
    notnan = ~np.isnan(in_array)
    n = np.sum(notnan)

    # Handle the case when all values are NaN
    if n == 0:
        return 0.0

    in_array_copy = in_array.copy()  # Create a copy to avoid modifying the original
    in_array_copy[~notnan] = 0
    avg = np.sum(in_array_copy) / n
    out = np.sqrt(np.sum(((in_array_copy - avg) * notnan) ** 2) / n)  # there should be -1 in the denominator for true std
    return out