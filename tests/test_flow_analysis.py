import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.flow_analysis import vorticity as vorticity_open
from openopticalflow.flow_analysis import invariant2_factor as invariant2_factor_open

# Import the corrected invariant2_factor implementation for reference
from openopticalflow.invariant2_factor import invariant2_factor as invariant2_factor_corrected

def test_flow_analysis():
    """Test and compare implementations of flow analysis functions"""

    # Create test cases
    test_cases = []

    # Test case 1: Rigid body rotation (vortex)
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    vx = -y
    vy = x
    test_cases.append(("Rigid body rotation", vx, vy))

    # Test case 2: Shear flow
    vx = y
    vy = np.zeros_like(x)
    test_cases.append(("Shear flow", vx, vy))

    # Test case 3: Stagnation point flow
    vx = x
    vy = -y
    test_cases.append(("Stagnation point flow", vx, vy))

    # Set conversion factors
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel

    # Run tests
    for name, vx, vy in test_cases:
        print(f"\nTesting {name}:")

        # Test vorticity
        vort_open = vorticity_open(vx, vy)

        # Test invariant2_factor
        q_open = invariant2_factor_open(vx, vy, factor_x, factor_y)

        # Test corrected invariant2_factor
        q_corrected = invariant2_factor_corrected(vx, vy, factor_x, factor_y)

        # Calculate differences
        diff_q_corrected = np.abs(q_open - q_corrected).max()

        # Print results
        print(f"Maximum differences:")
        print(f"  Q-criterion vs corrected: {diff_q_corrected:.8f}")

        # Calculate statistics
        vort_open_mean = np.mean(vort_open)
        q_open_mean = np.mean(q_open)
        q_corrected_mean = np.mean(q_corrected)

        print(f"Mean values:")
        print(f"  Vorticity: open={vort_open_mean:.4f}")
        print(f"  Q-criterion: open={q_open_mean:.4f}, corrected={q_corrected_mean:.4f}")

        # Plot results
        plt.figure(figsize=(15, 10))

        # Velocity field
        plt.subplot(331)
        plt.quiver(x[::3, ::3], y[::3, ::3], vx[::3, ::3], vy[::3, ::3])
        plt.title(f'Velocity Field: {name}')
        plt.axis('equal')

        # Vorticity
        plt.subplot(332)
        plt.imshow(vort_open, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Vorticity (Open)')

        # Q-criterion
        plt.subplot(334)
        plt.imshow(q_open, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Q-criterion (Open)')

        plt.subplot(335)
        plt.imshow(q_corrected, cmap='RdBu_r')
        plt.colorbar()
        plt.title('Q-criterion (Corrected)')

        # Differences
        plt.subplot(337)
        diff = np.abs(q_open - q_corrected)
        plt.imshow(diff, cmap='viridis')
        plt.colorbar()
        plt.title('Q-criterion vs Corrected Difference')

        plt.tight_layout()
        plt.savefig(f'results/flow_analysis_{name.replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run tests
    test_flow_analysis()
