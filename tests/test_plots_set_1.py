import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementations
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization.vis_flow import vis_flow as vis_flow_open
from visualization.plots_set_1 import plots_set_1 as plots_set_1_open

# For comparison, we'll use the same implementation since the comparison files don't exist
vis_flow_comparison = vis_flow_open
plots_set_1_comparison = plots_set_1_open

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

    # Test open implementation
    plt.figure(figsize=(10, 8))
    plt.suptitle('Open Implementation: vis_flow')
    try:
        vis_flow_open(ux, uy, gx, offset)
        plt.savefig('results/vis_flow_open.png')
        print("Open vis_flow implementation test: SUCCESS")
    except Exception as e:
        print("Open vis_flow implementation test: FAILED -", e)
    plt.close()

    # Test comparison implementation
    plt.figure(figsize=(10, 8))
    plt.suptitle('Comparison Implementation: vis_flow')
    try:
        vis_flow_comparison(ux, uy, gx, offset, 2, 'red')
        plt.savefig('results/vis_flow_comparison.png')
        print("Comparison vis_flow implementation test: SUCCESS")
    except Exception as e:
        print("Comparison vis_flow implementation test: FAILED -", e)
    plt.close()

def test_plots_set_1():
    """Test the plots_set_1 function"""
    # Create test data
    ux, uy, im1, im2 = create_test_flow_field()

    # Temporarily disable showing plots
    old_show = plt.show
    plt.show = lambda: None

    # Test open implementation
    try:
        figures_open = plots_set_1_open(im1, im2, ux, uy, show_plot=False)
        print("Open plots_set_1 implementation test: SUCCESS")
        # Save one of the figures
        figures_open[0].savefig('results/plots_set_1_open.png')
    except Exception as e:
        print("Open plots_set_1 implementation test: FAILED -", e)

    # Test comparison implementation
    try:
        # Temporarily disable showing plots
        old_show = plt.show
        plt.show = lambda: None

        # The comparison implementation doesn't return figures and doesn't have show_plot parameter
        plots_set_1_comparison(im1, im2, ux, uy, im1, im2, ux, uy)
        print("Comparison plots_set_1 implementation test: SUCCESS")

        # Save the current figure
        plt.gcf().savefig('results/plots_set_1_comparison.png')

        # Restore showing plots
        plt.show = old_show
    except Exception as e:
        print("Comparison plots_set_1 implementation test: FAILED -", e)
        # Restore showing plots
        plt.show = old_show

    # Restore showing plots
    plt.show = old_show

    # Close all figures to avoid memory leaks
    plt.close('all')

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run tests
    test_vis_flow()
    test_plots_set_1()
