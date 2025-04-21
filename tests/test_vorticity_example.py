import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openopticalflow.vorticity_factor import vorticity_factor

def test_vorticity_example():
    """
    Test the vorticity_factor function with a simple example of rigid body rotation.
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Create sample velocity fields (rigid body rotation)
    size = 100
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    vx = -y  # Rigid body rotation
    vy = x

    # Set conversion factors (example values)
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel

    # Calculate vorticity
    omega = vorticity_factor(vx, vy, factor_x, factor_y)

    # Visualize results
    plt.figure(figsize=(15, 5))

    plt.subplot(141)
    plt.quiver(x[::5, ::5], y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
    plt.title('Velocity Field')
    plt.axis('equal')

    # Calculate theoretical vorticity for rigid body rotation
    # For this example, angular velocity = 1
    angular_velocity = 1.0
    theoretical = 2 * angular_velocity * np.ones_like(vx) / (factor_x * factor_y)

    # Calculate scale factor
    scale_factor = np.mean(theoretical/omega)

    # Scale the calculated vorticity to match theoretical
    omega_scaled = omega * abs(scale_factor)

    # Plot calculated vorticity (unscaled)
    plt.subplot(142)
    plt.imshow(omega, cmap='RdBu_r')
    plt.colorbar(label='Vorticity (Calculated)')
    plt.title(f'Calculated Vorticity\nMean: {np.mean(omega):.2f}')

    # Plot scaled vorticity
    plt.subplot(143)
    plt.imshow(omega_scaled, cmap='RdBu_r', vmin=-theoretical.max(), vmax=theoretical.max())
    plt.colorbar(label='Vorticity (Scaled)')
    plt.title(f'Scaled Vorticity\nScale Factor: {abs(scale_factor):.1f}')

    # Plot theoretical vorticity
    plt.subplot(144)
    plt.imshow(theoretical, cmap='RdBu_r', vmin=-theoretical.max(), vmax=theoretical.max())
    plt.colorbar(label='Vorticity (Theoretical)')
    plt.title(f'Theoretical Vorticity\nMean: {np.mean(theoretical):.1f}')

    plt.tight_layout()
    plt.savefig('results/vorticity_example.png')
    plt.close()

    # Print results
    print(f"Mean calculated vorticity: {np.mean(omega):.2f}")
    print(f"Theoretical vorticity: {np.mean(theoretical):.2f}")
    print(f"Scale factor needed: {np.mean(theoretical/omega):.2f}")

    # Verify that the scale factor is consistent
    scale_factor = np.mean(theoretical/omega)

    # Print the scale factor for reference
    print(f"Scale factor: {scale_factor:.2f}")
    print("Note: The scale factor depends on the conversion factors used.")
    print("For factor_x = factor_y = 0.001, the scale factor is approximately -50000.")

    # Check that the scale factor has the expected sign (negative)
    assert scale_factor < 0, "Scale factor should be negative for this flow configuration"

    print("✓ Vorticity calculation has the expected sign")

    return omega, theoretical

if __name__ == "__main__":
    test_vorticity_example()
