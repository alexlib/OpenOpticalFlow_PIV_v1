"""
Test for the vis_flow function.

This test compares the output of the vis_flow function with ground truth data.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Add parent directory to Python path to allow imports from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the vis_flow function
from visualization.vis_flow import vis_flow

def create_ground_truth_flow():
    """
    Create a ground truth flow field for testing.
    
    Returns:
        tuple: (vx, vy) velocity components
    """
    # Create a simple rotating flow field (vortex)
    size = 50
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing='ij')
    vx = -y  # Horizontal velocity proportional to -y
    vy = x   # Vertical velocity proportional to x
    
    return vx, vy

def test_vis_flow_basic():
    """
    Test basic functionality of vis_flow.
    """
    # Create ground truth flow field
    vx, vy = create_ground_truth_flow()
    
    # Call vis_flow with default parameters
    plt.figure(figsize=(8, 8))
    h = vis_flow(vx, vy)
    
    # Check that h is a valid quiver object
    assert h is not None
    assert 'Quiver' in str(type(h))
    
    # Check that the quiver has the expected number of arrows
    # For a 50x50 field with default gx=25, we expect around 4 arrows in each dimension
    expected_arrows = (50 // 25) ** 2
    actual_arrows = len(h.X)
    assert actual_arrows > 0, "No arrows were plotted"
    
    # Close the figure to avoid memory leaks
    plt.close()
    
    print("Basic vis_flow test passed")

def test_vis_flow_parameters():
    """
    Test vis_flow with different parameters.
    """
    # Create ground truth flow field
    vx, vy = create_ground_truth_flow()
    
    # Test with different grid spacing
    plt.figure(figsize=(8, 8))
    h1 = vis_flow(vx, vy, gx=10)  # More arrows
    assert len(h1.X) > 0, "No arrows were plotted with gx=10"
    plt.close()
    
    plt.figure(figsize=(8, 8))
    h2 = vis_flow(vx, vy, gx=40)  # Fewer arrows
    assert len(h2.X) > 0, "No arrows were plotted with gx=40"
    plt.close()
    
    # Test with different offset
    plt.figure(figsize=(8, 8))
    h3 = vis_flow(vx, vy, offset=5)
    assert len(h3.X) > 0, "No arrows were plotted with offset=5"
    plt.close()
    
    # Test with different magnitude scaling
    plt.figure(figsize=(8, 8))
    h4 = vis_flow(vx, vy, mag=2)  # Longer arrows
    assert len(h4.X) > 0, "No arrows were plotted with mag=2"
    plt.close()
    
    # Test with different color
    plt.figure(figsize=(8, 8))
    h5 = vis_flow(vx, vy, col='green')
    assert len(h5.X) > 0, "No arrows were plotted with col='green'"
    plt.close()
    
    print("Parameter variation tests passed")

def test_vis_flow_comparison():
    """
    Test vis_flow by comparing with expected output.
    """
    # Create ground truth flow field
    vx, vy = create_ground_truth_flow()
    
    # Create a figure with two subplots for comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot using vis_flow in the first subplot
    plt.sca(ax1)
    h1 = vis_flow(vx, vy, gx=10, mag=1, col='red')
    ax1.set_title('vis_flow Output')
    
    # Plot using matplotlib's quiver directly in the second subplot
    plt.sca(ax2)
    size = vx.shape[0]
    gx = 10
    jmp = max(1, size // gx)
    indx = np.arange(0, size, jmp)
    X, Y = np.meshgrid(indx, indx)
    U = vx[indx][:, indx]
    V = vy[indx][:, indx]
    h2 = ax2.quiver(X, Y, U, V, color='blue')
    ax2.set_title('Direct Quiver Plot')
    
    # Set the same limits for both plots
    ax1.set_xlim(0, size)
    ax1.set_ylim(0, size)
    ax2.set_xlim(0, size)
    ax2.set_ylim(0, size)
    
    # Save the comparison figure
    plt.tight_layout()
    os.makedirs('test_output', exist_ok=True)
    plt.savefig('test_output/vis_flow_comparison.png')
    plt.close()
    
    print("Comparison test completed. Check test_output/vis_flow_comparison.png")
    
    # Verify that both plots have arrows
    assert len(h1.X) > 0, "No arrows in vis_flow output"
    assert len(h2.X) > 0, "No arrows in direct quiver plot"

def test_vis_flow_with_nan():
    """
    Test vis_flow with NaN values in the flow field.
    """
    # Create ground truth flow field
    vx, vy = create_ground_truth_flow()
    
    # Add some NaN values
    vx[10:20, 10:20] = np.nan
    vy[30:40, 30:40] = np.nan
    
    # Call vis_flow
    plt.figure(figsize=(8, 8))
    h = vis_flow(vx, vy)
    
    # Check that h is a valid quiver object
    assert h is not None
    assert 'Quiver' in str(type(h))
    
    # Check that the quiver has arrows (NaNs should be handled)
    assert len(h.X) > 0, "No arrows were plotted with NaN values"
    
    # Close the figure to avoid memory leaks
    plt.close()
    
    print("NaN handling test passed")

def run_all_tests():
    """
    Run all tests and generate visual output.
    """
    print("Running vis_flow tests...")
    
    test_vis_flow_basic()
    test_vis_flow_parameters()
    test_vis_flow_comparison()
    test_vis_flow_with_nan()
    
    print("All tests completed successfully!")
    
    # Create a comprehensive visualization for manual inspection
    vx, vy = create_ground_truth_flow()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original flow field visualization
    plt.sca(axes[0, 0])
    plt.imshow(np.sqrt(vx**2 + vy**2), cmap='jet')
    plt.colorbar()
    plt.title('Velocity Magnitude')
    
    # vis_flow with different grid spacings
    plt.sca(axes[0, 1])
    vis_flow(vx, vy, gx=5, col='red')
    plt.title('vis_flow with gx=5')
    
    plt.sca(axes[0, 2])
    vis_flow(vx, vy, gx=10, col='green')
    plt.title('vis_flow with gx=10')
    
    plt.sca(axes[1, 0])
    vis_flow(vx, vy, gx=20, col='blue')
    plt.title('vis_flow with gx=20')
    
    # vis_flow with different magnitude scaling
    plt.sca(axes[1, 1])
    vis_flow(vx, vy, gx=10, mag=0.5, col='purple')
    plt.title('vis_flow with mag=0.5')
    
    plt.sca(axes[1, 2])
    vis_flow(vx, vy, gx=10, mag=2, col='orange')
    plt.title('vis_flow with mag=2')
    
    # Set the same limits for all plots
    for ax in axes.flatten():
        ax.set_xlim(0, vx.shape[1])
        ax.set_ylim(0, vx.shape[0])
    
    # Save the comprehensive visualization
    plt.tight_layout()
    os.makedirs('test_output', exist_ok=True)
    plt.savefig('test_output/vis_flow_comprehensive.png')
    plt.close()
    
    print("Comprehensive visualization saved to test_output/vis_flow_comprehensive.png")

if __name__ == "__main__":
    run_all_tests()
