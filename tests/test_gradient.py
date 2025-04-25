import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pivSuite.gradient import gradient

def matlab_gradient(Vx, Vy):
    """
    Direct implementation of the MATLAB code from the docstring
    
    function [grad_mag]=gradient(Vx, Vy)
    % Vx = imfilter(Vx, [1 1 1 1 1]'*[1 1 1 1 1]/25,'symmetric');
    % Vy = imfilter(Vy, [1 1 1 1 1]'*[1,1 1 1,1]/25,'symmetric');
    dx=1;
    D = [0, -1, 0; 0,0,0; 0,1,0]/2; %%% partial derivative 
    Vy_x = imfilter(Vy, D'/dx, 'symmetric',  'same'); 
    Vx_y = imfilter(Vx, D/dx, 'symmetric',  'same');
    grad_mag=(Vy_x.^2+Vx_y.^2).^0.5;
    """
    dx = 1
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # partial derivative
    
    # MATLAB's imfilter with 'symmetric' is equivalent to scipy's convolve with 'reflect'
    Vy_x = convolve(Vy, D.T / dx, mode='reflect')
    Vx_y = convolve(Vx, D / dx, mode='reflect')
    
    grad_mag = np.sqrt(Vy_x**2 + Vx_y**2)
    
    return grad_mag

def test_gradient():
    """Test that the Python implementation matches the MATLAB implementation"""
    
    # Create test data - a simple vector field with a known gradient
    size = 50
    x, y = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size))
    
    # Test case 1: Simple linear vector field
    Vx1 = x
    Vy1 = y
    
    # Test case 2: Vortex-like vector field
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    Vx2 = -r * np.sin(theta)
    Vy2 = r * np.cos(theta)
    
    # Test case 3: Random vector field
    np.random.seed(42)  # For reproducibility
    Vx3 = np.random.rand(size, size)
    Vy3 = np.random.rand(size, size)
    
    # List of test cases
    test_cases = [
        ("Linear field", Vx1, Vy1),
        ("Vortex field", Vx2, Vy2),
        ("Random field", Vx3, Vy3)
    ]
    
    # Run tests for each case
    for name, Vx, Vy in test_cases:
        print(f"Testing {name}...")
        
        # Calculate gradient using both implementations
        matlab_result = matlab_gradient(Vx, Vy)
        python_result = gradient(Vx, Vy)
        
        # Check if results are identical
        is_equal = np.allclose(matlab_result, python_result, rtol=1e-10, atol=1e-10)
        
        if is_equal:
            print(f"✓ {name}: MATLAB and Python implementations produce identical results")
        else:
            print(f"✗ {name}: Results differ!")
            max_diff = np.max(np.abs(matlab_result - python_result))
            print(f"  Maximum absolute difference: {max_diff}")
            
            # Plot the differences for visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(matlab_result)
            plt.colorbar()
            plt.title('MATLAB Implementation')
            
            plt.subplot(1, 3, 2)
            plt.imshow(python_result)
            plt.colorbar()
            plt.title('Python Implementation')
            
            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(matlab_result - python_result))
            plt.colorbar()
            plt.title('Absolute Difference')
            
            plt.tight_layout()
            plt.savefig(f'gradient_diff_{name.replace(" ", "_")}.png')
            plt.close()
        
        print()

if __name__ == "__main__":
    test_gradient()
