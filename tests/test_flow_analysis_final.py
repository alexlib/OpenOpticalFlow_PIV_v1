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
from openopticalflow.invariant2_factor import invariant2_factor as invariant2_factor_dedicated

def test_flow_analysis_final():
    """Test and compare implementations of flow analysis functions"""

    # Create test case: Rigid body rotation (vortex)
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    vx = -y
    vy = x

    # Set conversion factors
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel

    # Calculate vorticity
    vort_open = vorticity_open(vx, vy)

    # Calculate Q-criterion
    q_open = invariant2_factor_open(vx, vy, factor_x, factor_y)

    # Calculate with dedicated implementation
    q_standard = invariant2_factor_dedicated(vx, vy, factor_x, factor_y)
    q_compatible = invariant2_factor_dedicated(vx, vy, factor_x, factor_y, compatibility_mode=True)

    # Calculate differences
    diff_q_standard = np.abs(q_open - q_standard).max()

    # Print results
    print(f"Maximum differences:")
    print(f"  Q-criterion (open vs standard): {diff_q_standard:.8f}")

    # Calculate statistics
    vort_open_mean = np.mean(vort_open)
    q_open_mean = np.mean(q_open)
    q_standard_mean = np.mean(q_standard)
    q_compatible_mean = np.mean(q_compatible)

    print(f"Mean values:")
    print(f"  Vorticity: open={vort_open_mean:.4f}")
    print(f"  Q-criterion: open={q_open_mean:.4f}")
    print(f"  Q-criterion (dedicated): standard={q_standard_mean:.4f}, compatible={q_compatible_mean:.4f}")

    # Plot results
    plt.figure(figsize=(12, 10))

    # Velocity field
    plt.subplot(231)
    plt.quiver(x[::3, ::3], y[::3, ::3], vx[::3, ::3], vy[::3, ::3])
    plt.title('Velocity Field: Rigid Body Rotation')
    plt.axis('equal')

    # Vorticity
    plt.subplot(232)
    plt.imshow(vort_open, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Vorticity (Open)')

    # Q-criterion
    plt.subplot(233)
    plt.imshow(q_open, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Q-criterion (Open)')

    # Dedicated implementations
    plt.subplot(234)
    plt.imshow(q_standard, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Q-criterion (Standard)')

    plt.subplot(235)
    plt.imshow(q_compatible, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Q-criterion (Compatible)')

    # Differences
    plt.subplot(236)
    diff = np.abs(q_open - q_standard)
    plt.imshow(diff, cmap='viridis')
    plt.colorbar()
    plt.title('Open vs Standard Difference')

    plt.tight_layout()
    plt.savefig('results/flow_analysis_final.png')
    plt.close()

    print("Test completed. See results/flow_analysis_final.png for visualization.")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run test
    test_flow_analysis_final()
