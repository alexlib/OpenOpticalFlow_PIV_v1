import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.vorticity_factor import vorticity_factor

def test_vorticity_scaling():
    """Test the scaling of vorticity calculation"""

    # Create a rigid body rotation flow field
    size = 100
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    vx = -y
    vy = x

    # For rigid body rotation with angular velocity = 1, the theoretical vorticity is 2
    # But with scaling factors, it becomes 2/(factor_x*factor_y)

    # Test different scaling factors
    scaling_factors = [
        (1.0, 1.0),           # No scaling
        (0.001, 0.001),       # mm/pixel
        (0.01, 0.01),         # cm/pixel
        (0.001, 0.002),       # Different x and y scaling
    ]

    plt.figure(figsize=(15, 10))

    for i, (factor_x, factor_y) in enumerate(scaling_factors):
        # Calculate vorticity
        omega = vorticity_factor(vx, vy, factor_x, factor_y)

        # Calculate theoretical vorticity
        # For rigid body rotation, vorticity = 2 * angular velocity
        # Angular velocity = 1 in our case
        # With scaling factors, theoretical vorticity = 2 * angular_velocity / (factor_x * factor_y)
        angular_velocity = 1.0
        theoretical = 2 * angular_velocity / (factor_x * factor_y)

        # Calculate error
        error = np.abs(omega - theoretical).mean()

        # Print results
        print(f"Scaling factors: factor_x={factor_x}, factor_y={factor_y}")
        print(f"  Mean vorticity: {np.mean(omega):.6f}")
        print(f"  Theoretical vorticity: {theoretical:.6f}")
        print(f"  Mean error: {error:.6f}")

        # Plot results
        plt.subplot(2, 2, i+1)
        plt.imshow(omega, cmap='RdBu_r')
        plt.colorbar(label='Vorticity')
        plt.title(f'Vorticity with scaling factors: ({factor_x}, {factor_y})\nMean: {np.mean(omega):.2f}, Theoretical: {theoretical:.2f}')

    plt.tight_layout()
    plt.savefig('results/vorticity_scaling.png')
    plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run test
    test_vorticity_scaling()
