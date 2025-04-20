import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.openopticalflow.generate_invmatrix import generate_invmatrix as generate_invmatrix_1
from openopticalflow.generate_invmatrix import generate_invmatrix as generate_invmatrix_2

def test_generate_invmatrix_implementations():
    # Test case 1: Random image
    size = 50
    I = np.random.rand(size, size)
    alpha = 0.1
    h = 2

    # Test case 2: Zero image (edge case)
    I_zero = np.zeros((size, size))

    # Test case 3: Constant image (edge case)
    I_const = np.ones((size, size))

    # Test case 4: Checkerboard pattern
    I_check = np.indices((size, size)).sum(axis=0) % 2

    test_cases = [
        ("Random", I),
        ("Zero", I_zero),
        ("Constant", I_const),
        ("Checkerboard", I_check)
    ]

    for name, test_image in test_cases:
        print(f"\nTesting {name} image:")
        b11_1, b12_1, b22_1 = generate_invmatrix_1(test_image, alpha, h)
        b11_2, b12_2, b22_2 = generate_invmatrix_2(test_image, alpha, h)

        diff_b11 = np.abs(b11_1 - b11_2).max()
        diff_b12 = np.abs(b12_1 - b12_2).max()
        diff_b22 = np.abs(b22_1 - b22_2).max()

        print(f"Maximum differences:")
        print(f"B11: {diff_b11}")
        print(f"B12: {diff_b12}")
        print(f"B22: {diff_b22}")

if __name__ == "__main__":
    test_generate_invmatrix_implementations()
