import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.vorticity import vorticity as vorticity_open

def test_vorticity():
    """Test and compare implementations of vorticity calculation"""

    # Create test cases
    test_cases = []

    # Test case 1: Rigid body rotation (vortex)
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # Scale by 10 to make vorticity more visible
    vx = -10 * y
    vy = 10 * x
    test_cases.append(("Rigid body rotation", vx, vy))

    # Test case 2: Shear flow
    vx = 10 * y
    vy = np.zeros_like(x)
    test_cases.append(("Shear flow", vx, vy))

    # Test case 3: Stagnation point flow
    vx = 10 * x
    vy = -10 * y
    test_cases.append(("Stagnation point flow", vx, vy))

    # Test case 4: Random flow
    np.random.seed(42)
    vx = 10 * np.random.randn(size, size)
    vy = 10 * np.random.randn(size, size)
    test_cases.append(("Random flow", vx, vy))

    # Run tests
    for name, vx, vy in test_cases:
        print(f"\nTesting {name}:")

        # Time the implementation
        start = time()
        vort_open = vorticity_open(vx, vy)
        time_open = time() - start

        # Print results
        print(f"Vorticity calculation completed successfully")
        print(f"Execution times:")
        print(f"  Open implementation: {time_open:.6f} seconds")

        # Calculate statistics
        vort_open_mean = np.mean(vort_open)

        print(f"Mean values:")
        print(f"  Open: {vort_open_mean:.6f}")

        # Calculate theoretical vorticity for rigid body rotation
        if name == "Rigid body rotation":
            # For rigid body rotation with angular velocity ω, vorticity = 2ω
            # In our case, we scaled the velocity by 10, so angular velocity = 10
            angular_velocity = 10.0
            theoretical = 2 * angular_velocity * np.ones_like(vx)
            error_open = np.abs(vort_open - theoretical).mean()
            print(f"Mean error from theoretical value:")
            print(f"  Open: {error_open:.6f}")

        # Plot results
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.quiver(x[::3, ::3], y[::3, ::3], vx[::3, ::3], vy[::3, ::3])
        plt.title(f'Velocity Field: {name}')
        plt.axis('equal')

        plt.subplot(122)
        plt.imshow(vort_open, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Vorticity')

        plt.tight_layout()
        plt.savefig(f'results/vorticity_simple_{name.replace(" ", "_").lower()}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run tests
    test_vorticity()
