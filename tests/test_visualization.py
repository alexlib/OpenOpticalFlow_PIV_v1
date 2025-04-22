"""
Test script to verify that the visualization files are working correctly.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the visualization modules
from visualization.vis_flow import vis_flow, plot_streamlines
from visualization.plots_set_1 import plots_set_1
from visualization.plots_set_2 import plots_set_2

def create_test_flow_field(size=(50, 50)):
    """Create a synthetic flow field for testing"""
    # Create a rotating flow field
    y, x = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]), indexing='ij')
    ux = -y  # horizontal flow
    uy = x   # vertical flow

    # Create sample images
    im1 = np.random.rand(size[0], size[1])
    im2 = np.random.rand(size[0], size[1])

    return ux, uy, im1, im2

def test_vis_flow():
    """Test the vis_flow function"""
    # Create test data
    ux, uy, im1, im2 = create_test_flow_field()

    # Parameters
    gx = 5  # Grid spacing for visualization
    offset = 1

    # Test vis_flow
    plt.figure(figsize=(10, 8))
    plt.suptitle('vis_flow test')
    try:
        vis_flow(ux, uy, gx, offset)
        plt.savefig('vis_flow_test.png')
        print("vis_flow test: SUCCESS")
    except Exception as e:
        print("vis_flow test: FAILED -", e)
    plt.close()

    # Test plot_streamlines
    plt.figure(figsize=(10, 8))
    plt.suptitle('plot_streamlines test')
    try:
        plot_streamlines(ux, uy, density=1.5)
        plt.savefig('plot_streamlines_test.png')
        print("plot_streamlines test: SUCCESS")
    except Exception as e:
        print("plot_streamlines test: FAILED -", e)
    plt.close()

def test_plots_set_1():
    """Test the plots_set_1 function"""
    # Create test data
    ux, uy, im1, im2 = create_test_flow_field()

    # Temporarily disable showing plots
    old_show = plt.show
    plt.show = lambda: None

    # Test plots_set_1
    try:
        figures = plots_set_1(im1, im2, ux, uy, show_plot=False)
        print("plots_set_1 test: SUCCESS")
        # Save one of the figures
        figures[0].savefig('plots_set_1_test.png')
    except Exception as e:
        print("plots_set_1 test: FAILED -", e)

    # Restore showing plots
    plt.show = old_show

    # Close all figures to avoid memory leaks
    plt.close('all')

if __name__ == "__main__":
    # Run tests
    test_vis_flow()
    test_plots_set_1()
