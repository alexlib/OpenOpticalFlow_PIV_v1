import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.invariant2_factor import invariant2_factor as invariant2_open
from openopticalflow.invariant2_factor import invariant2_factor_loop as invariant2_open_loop
from openopticalflow.invariant2_factor import invariant2_factor_vectorized as invariant2_open_vectorized
from comparison.openopticalflow.invariant2_factor import invariant2_factor as invariant2_comparison
from comparison.openopticalflow.invariant2_factor import invariant2_factor_vectorized as invariant2_comparison_vectorized

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
        q: Analytical Q-criterion
    """
    # Decay factor
    decay = np.exp(-2 * nu * t)
    
    # Velocity components
    vx = -A * np.cos(x) * np.sin(y) * decay
    vy = A * np.sin(x) * np.cos(y) * decay
    
    # Analytical Q-criterion
    # For Taylor-Green vortex, Q = 0.5 * (cos(2x) + cos(2y)) * decay^2
    q = 0.5 * A**2 * (np.cos(2*x) + np.cos(2*y)) * decay**2
    
    return vx, vy, q

def rankine_vortex(x, y, r0=0.2, Gamma=1.0):
    """
    Rankine vortex - a simple model with constant vorticity inside a core.
    
    Parameters:
        x, y: Coordinates (meshgrid)
        r0: Core radius
        Gamma: Circulation
        
    Returns:
        vx, vy: Velocity components
        q: Analytical Q-criterion
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
    
    # Analytical Q-criterion
    q = np.zeros_like(x)
    # Inside core: Q = 0.5 * (Gamma/(2*pi*r0^2))^2
    q[mask_inside] = 0.5 * (Gamma / (2 * np.pi * r0**2))**2
    # Outside core: Q = 0
    
    return vx, vy, q

def stagnation_point_flow(x, y, a=1.0):
    """
    Stagnation point flow - a simple flow with zero vorticity.
    
    Parameters:
        x, y: Coordinates (meshgrid)
        a: Strain rate
        
    Returns:
        vx, vy: Velocity components
        q: Analytical Q-criterion (always negative for pure strain)
    """
    # Velocity components
    vx = a * x
    vy = -a * y
    
    # Analytical Q-criterion
    # For stagnation point flow, Q = -0.5 * a^2
    q = -0.5 * a**2 * np.ones_like(x)
    
    return vx, vy, q

def test_invariant2_factor_analytical():
    """Test invariant2_factor implementations against analytical solutions"""
    
    # Create a grid
    n = 100
    x = np.linspace(-np.pi, np.pi, n)
    y = np.linspace(-np.pi, np.pi, n)
    X, Y = np.meshgrid(x, y)
    
    # Test cases
    test_cases = [
        ("Taylor-Green Vortex", lambda: taylor_green_vortex(X, Y)),
        ("Rankine Vortex", lambda: rankine_vortex(X, Y)),
        ("Stagnation Point Flow", lambda: stagnation_point_flow(X, Y))
    ]
    
    # Set conversion factors
    factor_x = 1.0  # Use 1.0 to simplify comparison with analytical solutions
    factor_y = 1.0
    
    # Run tests
    for name, vortex_func in test_cases:
        print(f"\nTesting {name}:")
        
        # Generate velocity field and analytical Q-criterion
        vx, vy, q_analytical = vortex_func()
        
        # Calculate Q-criterion using different implementations
        implementations = [
            ("Open (Default)", lambda: invariant2_open(vx, vy, factor_x, factor_y)),
            ("Open (Loop)", lambda: invariant2_open_loop(vx, vy, factor_x, factor_y)),
            ("Open (Vectorized)", lambda: invariant2_open_vectorized(vx, vy, factor_x, factor_y)),
            ("Open (Compatibility)", lambda: invariant2_open(vx, vy, factor_x, factor_y, compatibility_mode=True)),
            ("Comparison (Loop)", lambda: invariant2_comparison(vx, vy, factor_x, factor_y)),
            ("Comparison (Vectorized)", lambda: invariant2_comparison_vectorized(vx, vy, factor_x, factor_y))
        ]
        
        results = {}
        times = {}
        
        for impl_name, impl_func in implementations:
            # Time the implementation
            start = time()
            try:
                q = impl_func()
                times[impl_name] = time() - start
                results[impl_name] = q
                print(f"  {impl_name}: {times[impl_name]*1000:.3f} ms")
            except Exception as e:
                print(f"  {impl_name} failed: {e}")
        
        # Compare with analytical solution
        print("\nComparison with analytical solution:")
        for impl_name, q in results.items():
            # Calculate scaling factor to match analytical solution
            # Use median to be robust against outliers
            nonzero = np.abs(q_analytical) > 1e-10
            if np.any(nonzero):
                scale_factor = np.median(q_analytical[nonzero] / q[nonzero])
                print(f"  {impl_name}: Scale factor = {scale_factor:.6f}")
                
                # Apply scaling factor
                q_scaled = q * scale_factor
                
                # Calculate error metrics
                abs_error = np.abs(q_scaled - q_analytical)
                mean_abs_error = np.mean(abs_error)
                max_abs_error = np.max(abs_error)
                
                print(f"    Mean error = {mean_abs_error:.8f}, Max error = {max_abs_error:.8f}")
            else:
                print(f"  {impl_name}: Cannot calculate scale factor (analytical solution is zero)")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Velocity field
        plt.subplot(331)
        plt.quiver(X[::5, ::5], Y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
        plt.title(f'{name} - Velocity Field')
        plt.axis('equal')
        
        # Analytical Q-criterion
        plt.subplot(332)
        im = plt.imshow(q_analytical, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                   origin='lower', cmap='RdBu_r')
        plt.colorbar(im)
        plt.title('Analytical Q-criterion')
        
        # Plot each implementation
        positions = [334, 335, 336, 337, 338, 339]
        for i, (impl_name, q) in enumerate(results.items()):
            if i < len(positions):
                # Calculate scaling factor
                nonzero = np.abs(q_analytical) > 1e-10
                if np.any(nonzero):
                    scale_factor = np.median(q_analytical[nonzero] / q[nonzero])
                    q_scaled = q * scale_factor
                else:
                    q_scaled = q
                    scale_factor = 1.0
                
                plt.subplot(positions[i])
                im = plt.imshow(q_scaled, extent=[-np.pi, np.pi, -np.pi, np.pi], 
                           origin='lower', cmap='RdBu_r')
                plt.colorbar(im)
                plt.title(f'{impl_name}\nScale: {scale_factor:.2f}, Error: {np.mean(np.abs(q_scaled - q_analytical)):.6f}')
        
        plt.tight_layout()
        plt.savefig(f'results/invariant2_analytical_{name.lower().replace("-", "_").replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run tests
    test_invariant2_factor_analytical()
