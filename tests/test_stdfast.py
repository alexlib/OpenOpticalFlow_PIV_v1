import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pivSuite.stdfast import stdfast

def matlab_stdfast(in_array):
    """
    Direct implementation of the MATLAB code from the docstring in stdfast.py
    
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
    # Flatten the array (equivalent to MATLAB's reshape(in,1,numel(in)))
    in_array = in_array.flatten()
    
    # Find non-NaN values
    notnan = ~np.isnan(in_array)
    
    # Count non-NaN values
    n = np.sum(notnan)
    
    # Handle the case when all values are NaN
    if n == 0:
        return 0.0
    
    # Create a copy to avoid modifying the original
    in_array_copy = in_array.copy()
    
    # Set NaN values to 0
    in_array_copy[~notnan] = 0
    
    # Calculate average of non-NaN values
    avg = np.sum(in_array_copy) / n
    
    # Calculate root-mean-square
    # Note: In the MATLAB code, there's a comment about using (n-0) instead of (n-1) in the denominator
    out = np.sqrt(np.sum(((in_array_copy - avg) * notnan) ** 2) / n)
    
    return out

def test_stdfast():
    """Test that the Python implementation matches the MATLAB implementation."""
    
    # Test cases
    test_cases = [
        {
            "name": "Random array",
            "input": np.random.rand(100)
        },
        {
            "name": "Array with NaN values",
            "input": np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0])
        },
        {
            "name": "2D array",
            "input": np.random.rand(10, 10)
        },
        {
            "name": "Array with all identical values",
            "input": np.ones(50)
        },
        {
            "name": "Array with all NaN values",
            "input": np.full(10, np.nan)
        }
    ]
    
    # Run tests for each case
    for case in test_cases:
        print(f"Testing {case['name']}...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Get results using both implementations
        matlab_result = matlab_stdfast(case["input"])
        python_result = stdfast(case["input"])
        
        # Check if results are identical
        is_equal = np.isclose(matlab_result, python_result, rtol=1e-10, atol=1e-10)
        
        if is_equal:
            print(f"✓ {case['name']}: MATLAB and Python implementations produce identical results")
            print(f"  Result: {python_result}")
        else:
            print(f"✗ {case['name']}: Results differ!")
            print(f"  MATLAB result: {matlab_result}")
            print(f"  Python result: {python_result}")
            print(f"  Absolute difference: {abs(matlab_result - python_result)}")
            print(f"  Relative difference: {abs((matlab_result - python_result) / matlab_result) if matlab_result != 0 else 'N/A'}")
        
        print()

if __name__ == "__main__":
    test_stdfast()
