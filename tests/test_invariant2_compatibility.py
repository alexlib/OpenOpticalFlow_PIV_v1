import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
from openopticalflow.invariant2_factor import invariant2_factor
from comparison.openopticalflow.invariant2_factor import invariant2_factor as invariant2_factor_comparison

def test_compatibility_mode():
    """Test the compatibility mode of invariant2_factor"""
    
    # Create a simple stagnation point flow
    size = 50
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    vx = x
    vy = -y
    
    # Set conversion factors
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel
    
    # Calculate Q-criterion using different methods
    qq_standard = invariant2_factor(vx, vy, factor_x, factor_y)
    qq_compatible = invariant2_factor(vx, vy, factor_x, factor_y, compatibility_mode=True)
    qq_comparison = invariant2_factor_comparison(vx, vy, factor_x, factor_y)
    
    # Calculate differences
    diff_standard_comparison = np.abs(qq_standard - qq_comparison).max()
    diff_compatible_comparison = np.abs(qq_compatible - qq_comparison).max()
    
    print(f"Maximum differences:")
    print(f"  Standard vs Comparison: {diff_standard_comparison:.8f}")
    print(f"  Compatible vs Comparison: {diff_compatible_comparison:.8f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.quiver(x[::3, ::3], y[::3, ::3], vx[::3, ::3], vy[::3, ::3])
    plt.title('Stagnation Point Flow')
    plt.axis('equal')
    
    plt.subplot(142)
    plt.imshow(qq_standard, cmap='RdBu_r')
    plt.colorbar(label='Q-criterion')
    plt.title('Standard Implementation')
    
    plt.subplot(143)
    plt.imshow(qq_compatible, cmap='RdBu_r')
    plt.colorbar(label='Q-criterion')
    plt.title('Compatible Implementation')
    
    plt.subplot(144)
    plt.imshow(qq_comparison, cmap='RdBu_r')
    plt.colorbar(label='Q-criterion')
    plt.title('Comparison Implementation')
    
    plt.tight_layout()
    plt.savefig('results/invariant2_factor_compatibility.png')
    plt.close()
    
    print("Compatibility test completed. See results/invariant2_factor_compatibility.png for visualization.")

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Run test
    test_compatibility_mode()
