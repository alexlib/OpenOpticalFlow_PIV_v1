import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.vorticity import vorticity

def taylor_green_vortex(x, y, t=0, nu=0.01, A=1.0):
    """
    Taylor-Green vortex - an exact solution to the 2D Navier-Stokes equations.

    Parameters:
        x, y: Coordinates (meshgrid)
        t: Time
        nu: Kinematic viscosity
        A: Amplitude

    Returns:
        vx, vy: Velocity components
        omega: Analytical vorticity
    """
    # Decay factor
    decay = np.exp(-2 * nu * t)

    # Velocity components
    vx = -A * np.cos(x) * np.sin(y) * decay
    vy = A * np.sin(x) * np.cos(y) * decay

    # Analytical vorticity (curl of velocity)
    omega = 2 * A * np.sin(x) * np.sin(y) * decay

    return vx, vy, omega

def rankine_vortex(x, y, r0=0.2, Gamma=1.0):
    """
    Rankine vortex - a simple model with constant vorticity inside a core.

    Parameters:
        x, y: Coordinates (meshgrid)
        r0: Core radius
        Gamma: Circulation

    Returns:
        vx, vy: Velocity components
        omega: Analytical vorticity
    """
    # Calculate radius from center
    r = np.sqrt(x**2 + y**2)

    # Initialize velocity components
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)

    # Inside the core (solid body rotation)
    mask_inside = r <= r0
    vx[mask_inside] = -Gamma * y[mask_inside] / (2 * np.pi * r0**2)
    vy[mask_inside] = Gamma * x[mask_inside] / (2 * np.pi * r0**2)

    # Outside the core (potential vortex)
    mask_outside = r > r0
    vx[mask_outside] = -Gamma * y[mask_outside] / (2 * np.pi * r[mask_outside]**2)
    vy[mask_outside] = Gamma * x[mask_outside] / (2 * np.pi * r[mask_outside]**2)

    # Analytical vorticity
    omega = np.zeros_like(x)
    omega[mask_inside] = Gamma / (np.pi * r0**2)  # Constant inside core
    # Vorticity is zero outside core

    return vx, vy, omega

def lamb_oseen_vortex(x, y, t=1.0, nu=0.01, Gamma=1.0):
    """
    Lamb-Oseen vortex - a solution to the diffusion of a point vortex.

    Parameters:
        x, y: Coordinates (meshgrid)
        t: Time
        nu: Kinematic viscosity
        Gamma: Circulation

    Returns:
        vx, vy: Velocity components
        omega: Analytical vorticity
    """
    # Calculate radius from center
    r = np.sqrt(x**2 + y**2)

    # Core radius (grows with time)
    r_core = 2 * np.sqrt(nu * t)

    # Velocity components (avoid division by zero at r=0)
    r_safe = np.maximum(r, 1e-10)
    v_theta = Gamma / (2 * np.pi * r_safe) * (1 - np.exp(-r**2 / (4 * nu * t)))

    vx = -v_theta * y / r_safe
    vy = v_theta * x / r_safe

    # Analytical vorticity
    omega = Gamma / (4 * np.pi * nu * t) * np.exp(-r**2 / (4 * nu * t))

    return vx, vy, omega

def test_vorticity_analytical():
    """Test vorticity calculation against analytical solutions"""

    # Create a grid
    n = 100
    x = np.linspace(-np.pi, np.pi, n)
    y = np.linspace(-np.pi, np.pi, n)
    X, Y = np.meshgrid(x, y)

    # Test cases
    test_cases = [
        ("Taylor-Green Vortex", lambda: taylor_green_vortex(X, Y)),
        ("Rankine Vortex", lambda: rankine_vortex(X, Y)),
        ("Lamb-Oseen Vortex", lambda: lamb_oseen_vortex(X, Y))
    ]

    # Run tests
    for name, vortex_func in test_cases:
        print(f"\nTesting {name}:")

        # Generate velocity field and analytical vorticity
        vx, vy, omega_analytical = vortex_func()

        # Calculate numerical vorticity
        start = time()
        # Apply smoothing for Rankine vortex to handle discontinuity
        if name == "Rankine Vortex":
            omega_numerical = vorticity(vx, vy, smooth=True, sigma=0.7)
        else:
            omega_numerical = vorticity(vx, vy)
        end = time()

        # Calculate error metrics
        abs_error = np.abs(omega_numerical - omega_analytical)
        mean_abs_error = np.mean(abs_error)
        max_abs_error = np.max(abs_error)
        rel_error = np.zeros_like(abs_error)
        nonzero = np.abs(omega_analytical) > 1e-10
        rel_error[nonzero] = abs_error[nonzero] / np.abs(omega_analytical[nonzero])
        mean_rel_error = np.mean(rel_error[nonzero]) if np.any(nonzero) else 0

        # Print results
        print(f"Execution time: {(end - start)*1000:.3f} ms")
        print(f"Mean absolute error: {mean_abs_error:.6f}")
        print(f"Maximum absolute error: {max_abs_error:.6f}")
        print(f"Mean relative error: {mean_rel_error:.6f}")

        # Plot results
        plt.figure(figsize=(15, 10))

        # Velocity field
        plt.subplot(231)
        plt.quiver(X[::5, ::5], Y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
        plt.title(f'{name} - Velocity Field')
        plt.axis('equal')

        # Analytical vorticity
        plt.subplot(232)
        im = plt.imshow(omega_analytical, extent=[-np.pi, np.pi, -np.pi, np.pi],
                   origin='lower', cmap='RdBu_r')
        plt.colorbar(im)
        plt.title('Analytical Vorticity')

        # Numerical vorticity
        plt.subplot(233)
        im = plt.imshow(omega_numerical, extent=[-np.pi, np.pi, -np.pi, np.pi],
                   origin='lower', cmap='RdBu_r')
        plt.colorbar(im)
        plt.title('Numerical Vorticity')

        # Absolute error
        plt.subplot(234)
        im = plt.imshow(abs_error, extent=[-np.pi, np.pi, -np.pi, np.pi],
                   origin='lower', cmap='viridis')
        plt.colorbar(im)
        plt.title(f'Absolute Error\nMean: {mean_abs_error:.6f}, Max: {max_abs_error:.6f}')

        # Relative error (where analytical vorticity is non-zero)
        plt.subplot(235)
        rel_error_plot = np.copy(rel_error)
        rel_error_plot[~nonzero] = 0
        im = plt.imshow(rel_error_plot, extent=[-np.pi, np.pi, -np.pi, np.pi],
                   origin='lower', cmap='viridis', vmax=min(1.0, np.max(rel_error_plot)*1.2))
        plt.colorbar(im)
        plt.title(f'Relative Error\nMean: {mean_rel_error:.6f}')

        # Cross-section plot
        plt.subplot(236)
        mid_idx = n // 2
        plt.plot(x, omega_analytical[mid_idx, :], 'b-', label='Analytical')
        plt.plot(x, omega_numerical[mid_idx, :], 'r--', label='Numerical')
        plt.legend()
        plt.title('Cross-section at y=0')
        plt.xlabel('x')
        plt.ylabel('Vorticity')

        plt.tight_layout()
        plt.savefig(f'results/vorticity_{name.lower().replace("-", "_").replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run tests
    test_vorticity_analytical()
