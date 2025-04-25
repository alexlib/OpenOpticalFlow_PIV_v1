import sys
import os
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pivSuite.dcn import dcn

def matlab_dcn(X1, X2, MaxD):
    """
    Direct implementation of the MATLAB code from the docstring in dcn.py

    function [cc] = dcn(X1,X2,MaxD)
    % computes cross-correlation using discrete convolution
    Nx = size(X1,2);
    Ny = size(X1,1);
    cc = zeros(Ny,Nx);
    % create variables defining where is cc(0,0)
    dx0 = Nx/2;
    dy0 = Ny/2;
    if rem(Nx,2) == 0
        dx0 = dx0+1;
    else
        dx0 = dx0+0.5;
    end
    if rem(Ny,2) == 0
        dy0 = dy0+1;
    else
        dy0 = dy0+0.5;
    end
    % pad IAs
    X1p = zeros(Ny+2*MaxD,Nx+2*MaxD);
    X2p = zeros(Ny+2*MaxD,Nx+2*MaxD);
    X1p(MaxD+1:MaxD+Ny,MaxD+1:MaxD+Nx) = X1;
    X2p(MaxD+1:MaxD+Ny,MaxD+1:MaxD+Nx) = X2;
    % convolve
    for kx = -MaxD:MaxD
        for ky = -MaxD:MaxD
            if abs(kx)+abs(ky)>MaxD, continue; end
            cc(dy0+ky,dx0+kx) = sum(sum(...
                X2p(ky+MaxD+1 : ky+MaxD+Ny,  kx+MaxD+1 : kx+MaxD+Nx) .* ...
                X1p(   MaxD+1 : MaxD+Ny,        MaxD+1 : MaxD+Nx)));
        end
    end
    """
    Nx = X1.shape[1]
    Ny = X1.shape[0]
    cc = np.zeros((Ny, Nx))

    # create variables defining where is cc(0,0)
    dx0 = Nx / 2
    dy0 = Ny / 2
    if Nx % 2 == 0:
        dx0 = dx0 + 1
    else:
        dx0 = dx0 + 0.5
    if Ny % 2 == 0:
        dy0 = dy0 + 1
    else:
        dy0 = dy0 + 0.5

    # pad IAs
    X1p = np.zeros((Ny + 2 * MaxD, Nx + 2 * MaxD))
    X2p = np.zeros((Ny + 2 * MaxD, Nx + 2 * MaxD))
    X1p[MaxD:MaxD + Ny, MaxD:MaxD + Nx] = X1
    X2p[MaxD:MaxD + Ny, MaxD:MaxD + Nx] = X2

    # convolve
    for kx in range(-MaxD, MaxD + 1):
        for ky in range(-MaxD, MaxD + 1):
            if abs(kx) + abs(ky) > MaxD:
                continue
            # Note: MATLAB is 1-indexed, Python is 0-indexed
            # In MATLAB: X1p(MaxD+1:MaxD+Ny,MaxD+1:MaxD+Nx)
            # In Python: X1p[MaxD:MaxD+Ny,MaxD:MaxD+Nx]
            # The +1 offset in MATLAB indices is already accounted for in the Python implementation
            cc[int(dy0 + ky), int(dx0 + kx)] = np.sum(
                X2p[ky + MaxD:ky + MaxD + Ny, kx + MaxD:kx + MaxD + Nx] *
                X1p[MaxD:MaxD + Ny, MaxD:MaxD + Nx]
            )

    return cc

def test_dcn():
    """Test that the Python implementation matches the MATLAB implementation."""

    # Test cases
    test_cases = [
        {
            "name": "Small random arrays",
            "X1": np.random.rand(8, 8),
            "X2": np.random.rand(8, 8),
            "MaxD": 2
        },
        {
            "name": "Medium random arrays",
            "X1": np.random.rand(16, 16),
            "X2": np.random.rand(16, 16),
            "MaxD": 3
        },
        {
            "name": "Odd-sized arrays",
            "X1": np.random.rand(9, 9),
            "X2": np.random.rand(9, 9),
            "MaxD": 2
        },
        {
            "name": "Rectangular arrays",
            "X1": np.random.rand(12, 8),
            "X2": np.random.rand(12, 8),
            "MaxD": 2
        }
    ]

    # Run tests for each case
    for case in test_cases:
        print(f"Testing {case['name']}...")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Get results using both implementations
        matlab_result = matlab_dcn(case["X1"], case["X2"], case["MaxD"])
        python_result = dcn(case["X1"], case["X2"], case["MaxD"])

        # Check if results are identical
        is_equal = np.allclose(matlab_result, python_result, rtol=1e-10, atol=1e-10)

        if is_equal:
            print(f"✓ {case['name']}: MATLAB and Python implementations produce identical results")
        else:
            print(f"✗ {case['name']}: Results differ!")
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
            plt.savefig(f'dcn_diff_{case["name"].replace(" ", "_")}.png')
            plt.close()

        print()

if __name__ == "__main__":
    test_dcn()
