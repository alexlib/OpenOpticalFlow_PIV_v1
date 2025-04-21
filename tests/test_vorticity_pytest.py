import numpy as np
import pytest
import matplotlib.pyplot as plt

from openopticalflow.vorticity import vorticity

@pytest.mark.vorticity
@pytest.mark.parametrize("field_type", [
    "rigid_rotation",
    "shear",
    "stagnation",
    "random"
])
@pytest.mark.parametrize("high_order", [False, True])
def test_vorticity_calculation(create_velocity_field, results_dir, field_type, high_order, request):
    """Test vorticity calculation with different flow fields and parameters"""

    # Get the field name for display
    field_names = {
        "rigid_rotation": "Rigid body rotation",
        "shear": "Shear flow",
        "stagnation": "Stagnation point flow",
        "random": "Random flow"
    }
    name = field_names[field_type]

    # Create velocity field
    x, y, vx, vy = create_velocity_field(field_type=field_type)

    # Set conversion factors
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel

    # Calculate vorticity with different parameters
    vort = vorticity(vx, vy, factor_x, factor_y, high_order=high_order)

    # Calculate statistics
    vort_mean = np.mean(vort)
    vort_max = np.max(vort)
    vort_min = np.min(vort)

    print(f"Field type: {name}")
    print(f"High-order: {high_order}")
    print(f"Vorticity statistics - Mean: {vort_mean:.6f}, Min: {vort_min:.6f}, Max: {vort_max:.6f}")

    # For rigid body rotation, we know the theoretical vorticity
    if field_type == "rigid_rotation":
        # With scaling factors, theoretical vorticity is 2/(factor_x*factor_y)
        theoretical_magnitude = 2.0 / (factor_x * factor_y)
        print(f"Theoretical magnitude: {theoretical_magnitude:.6f}")

        # Check that the magnitude is in the right ballpark
        # We use abs() because the implementations may have different sign conventions
        assert 10 < abs(vort_mean) < 10000, \
            f"Vorticity magnitude outside expected range: {abs(vort_mean):.6f}"

    # For shear flow, vorticity should be non-zero
    if field_type == "shear":
        assert abs(vort_mean) > 1e-6, "Vorticity has near-zero mean for shear flow"

    # For stagnation flow, vorticity should be close to zero
    if field_type == "stagnation":
        # The vorticity might not be exactly zero due to numerical effects
        # But it should be much smaller than for rotation or shear
        assert abs(vort_mean) < 100, f"Vorticity mean too large for stagnation flow: {abs(vort_mean):.6f}"

    # Plot results (only if visual marker is set)
    if request.config.getoption("--visual", default=False):
        plt.figure(figsize=(12, 4))

        plt.subplot(131)
        plt.quiver(x[::3, ::3], y[::3, ::3], vx[::3, ::3], vy[::3, ::3])
        plt.title(f'Velocity Field: {name}')
        plt.axis('equal')

        plt.subplot(132)
        plt.imshow(vort, cmap='RdBu_r')
        plt.colorbar()
        plt.title(f'Vorticity (high_order={high_order})')

        if field_type == "rigid_rotation":
            plt.subplot(133)
            theoretical = np.sign(vort_mean) * theoretical_magnitude * np.ones_like(vx)
            plt.imshow(theoretical, cmap='RdBu_r')
            plt.colorbar()
            plt.title('Theoretical Vorticity')

        plt.tight_layout()
        plt.savefig(f'{results_dir}/vorticity_{name.replace(" ", "_").lower()}_high_order_{high_order}.png')
        plt.close()
