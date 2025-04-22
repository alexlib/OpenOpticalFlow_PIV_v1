"""
OpenOpticalFlow_PIV_Run_Figures.py

This script runs the OpenOpticalFlow PIV analysis and generates figures similar to those
in the matlab_figures folder for comparison.
"""
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
import sys

# Add parent directory to Python path to allow imports from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from openopticalflow package
from openopticalflow.correction_illumination import correction_illumination
from openopticalflow.pre_processing_a import pre_processing_a
from openopticalflow.OpticalFlowPhysics_fun import OpticalFlowPhysics_fun
from openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1
from openopticalflow.vorticity import vorticity
from openopticalflow.invariant2_factor import invariant2_factor
# Import visualization functions if needed
# from openopticalflow.vis_flow import vis_flow, plot_streamlines

# Import PIV analysis functions
from pivSuite.pivAnalyzeImagePair import piv_analyze_image_pair

def run_piv_analysis(image_pair='white_oval'):
    """
    Run the PIV analysis and generate figures.

    Args:
        image_pair (str): Name of the image pair to use. Options:
            - 'vortex_pair': Vortex pair images
            - 'white_oval': White oval images

    Returns:
        dict: Dictionary containing the analysis results
    """
    print(f"Running PIV analysis on {image_pair} images...")

    # Create output directory for figures
    output_dir = 'python_figures'
    os.makedirs(output_dir, exist_ok=True)

    # Load Images
    if image_pair == 'vortex_pair':
        try:
            Im1 = imageio.imread('./images/vortex_pair_particles_1.tif')
            Im2 = imageio.imread('./images/vortex_pair_particles_2.tif')
            print("Successfully loaded vortex_pair_particles images")
        except Exception as e:
            print(f"Error loading vortex_pair_particles images: {e}")
            return None
    else:  # Default to white_oval
        try:
            Im1 = imageio.imread('./images/White_Oval_1.tif')
            Im2 = imageio.imread('./images/White_Oval_2.tif')
            print("Successfully loaded White_Oval images")
        except Exception as e:
            print(f"Error loading White_Oval images: {e}")
            return None

    # Select region of interest
    index_region = 0
    Im1 = Im1.astype(np.float64)
    Im2 = Im2.astype(np.float64)

    if index_region == 1:
        plt.imshow(Im1, cmap='gray')
        plt.axis('image')
        xy = plt.ginput(2)
        plt.close()
        x1, x2 = int(min(xy[0][0], xy[1][0])), int(max(xy[0][0], xy[1][0]))
        y1, y2 = int(min(xy[0][1], xy[1][1])), int(max(xy[0][1], xy[1][1]))
        I1 = Im1[y1:y2, x1:x2]
        I2 = Im2[y1:y2, x1:x2]
    else:
        I1 = Im1
        I2 = Im2

    I1_original = I1.copy()
    I2_original = I2.copy()

    # Set the Parameters for Optical Flow Computation
    lambda_1 = 20
    lambda_2 = 2000
    no_iteration = 1
    scale_im = 1
    size_average = 0
    size_filter = 6

    # Correcting the global and local intensity change in images
    m1, n1 = I1.shape
    window_shifting = [1, n1, 1, m1]
    I1, I2 = correction_illumination(I1, I2, window_shifting, size_average)

    # Cleaning the left and upper edges
    edge_width = 1

    # Pre-processing for reducing random noise
    I1, I2 = pre_processing_a(I1, I2, scale_im, size_filter)
    I_region1 = I1.copy()
    I_region2 = I2.copy()

    # Figure 1: Original Images
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(I1_original, cmap='gray')
    plt.title('First Image')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(I2_original, cmap='gray')
    plt.title('Second Image')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1.jpg', dpi=150)
    plt.close()

    # Figure 2: Processed Images
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(I_region1, cmap='gray')
    plt.title('Processed First Image')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(I_region2, cmap='gray')
    plt.title('Processed Second Image')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2.jpg', dpi=150)
    plt.close()

    # Run PIV analysis with coarse grid
    print("Running PIV analysis with coarse grid...")
    pivPar_coarse = {
        'iaSizeX': [64],
        'iaStepX': [32],
        'ccMethod': 'fft'
    }

    pivData_coarse = piv_analyze_image_pair(I1, I2, pivPar_coarse)
    ux_coarse = pivData_coarse['U']
    uy_coarse = pivData_coarse['V']

    # Handle 3D arrays (if the result is a 3D array with a single time slice)
    if len(ux_coarse.shape) > 2:
        ux_coarse = ux_coarse[:, :, 0]
        uy_coarse = uy_coarse[:, :, 0]

    # Figure 3: Coarse Grid Velocity Field
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    step_coarse = max(1, min(ux_coarse.shape) // 15)
    y_coarse, x_coarse = np.mgrid[0:ux_coarse.shape[0]:step_coarse, 0:ux_coarse.shape[1]:step_coarse]
    plt.quiver(x_coarse, y_coarse, ux_coarse[::step_coarse, ::step_coarse], uy_coarse[::step_coarse, ::step_coarse],
              scale=10, angles='xy', scale_units='xy')
    plt.title('Coarse Grid Velocity Field')
    plt.axis('equal')

    # Calculate velocity magnitude for coarse grid
    vel_mag_coarse = np.sqrt(ux_coarse**2 + uy_coarse**2)
    plt.subplot(222)
    plt.imshow(vel_mag_coarse, cmap='jet')
    plt.title('Coarse Grid Velocity Magnitude')
    plt.colorbar()

    # Calculate vorticity for coarse grid
    vor_coarse = vorticity(ux_coarse, uy_coarse)
    plt.subplot(223)
    plt.imshow(vor_coarse, cmap='RdBu_r')
    plt.title('Coarse Grid Vorticity')
    plt.colorbar()

    # Plot streamlines for coarse grid
    plt.subplot(224)
    y, x = np.meshgrid(np.arange(ux_coarse.shape[0]), np.arange(ux_coarse.shape[1]), indexing='ij')
    plt.streamplot(x, y, ux_coarse, uy_coarse, density=1.5, color='blue')
    plt.title('Coarse Grid Streamlines')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3.jpg', dpi=150)
    plt.close()

    # Run PIV analysis with multi-pass grid refinement
    print("Running PIV analysis with multi-pass grid refinement...")
    pivPar = {
        'iaSizeX': [64, 16, 8],
        'iaStepX': [32, 8, 4],
        'ccMethod': 'fft'
    }

    pivData1 = piv_analyze_image_pair(I1, I2, pivPar)
    ux0 = pivData1['U']
    uy0 = pivData1['V']

    # Handle 3D arrays (if the result is a 3D array with a single time slice)
    if len(ux0.shape) > 2:
        ux0 = ux0[:, :, 0]
        uy0 = uy0[:, :, 0]

    # Resize the initial velocity field
    n0, m0 = ux0.shape
    n1, m1 = I1.shape
    scale = round((n1 * m1 / (n0 * m0)) ** 0.5)
    ux0 = zoom(ux0, scale)
    uy0 = zoom(uy0, scale)

    # Figure 4: Multi-pass Grid Velocity Field
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    step = max(1, min(ux0.shape) // 25)
    y, x = np.mgrid[0:ux0.shape[0]:step, 0:ux0.shape[1]:step]
    plt.quiver(x, y, ux0[::step, ::step], uy0[::step, ::step], scale=10, angles='xy', scale_units='xy')
    plt.title('Multi-pass Grid Velocity Field')
    plt.axis('equal')

    # Calculate velocity magnitude
    vel_mag = np.sqrt(ux0**2 + uy0**2)
    plt.subplot(222)
    plt.imshow(vel_mag, cmap='jet')
    plt.title('Multi-pass Grid Velocity Magnitude')
    plt.colorbar()

    # Calculate vorticity
    vor = vorticity(ux0, uy0)
    plt.subplot(223)
    plt.imshow(vor, cmap='RdBu_r')
    plt.title('Multi-pass Grid Vorticity')
    plt.colorbar()

    # Plot streamlines
    plt.subplot(224)
    y, x = np.meshgrid(np.arange(ux0.shape[0]), np.arange(ux0.shape[1]), indexing='ij')
    plt.streamplot(x, y, ux0, uy0, density=1.5, color='blue')
    plt.title('Multi-pass Grid Streamlines')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4.jpg', dpi=150)
    plt.close()

    # Generate the shifted image and calculate velocity difference iteratively
    ux = ux0.copy()
    uy = uy0.copy()

    print("Iterative refinement with optical flow...")
    k = 1
    while k <= no_iteration:
        print(f"Iteration {k}/{no_iteration}")
        Im1_shift, uxI, uyI = shift_image_fun_refine_1(ux, uy, I1, I2)

        I1_shifted = Im1_shift.astype(np.float64)
        I2_current = I2.astype(np.float64)

        dux, duy, vor, _, _, _ = OpticalFlowPhysics_fun(I1_shifted, I2_current, lambda_1, lambda_2)

        ux_corr = uxI + dux
        uy_corr = uyI + duy

        k += 1

    # Refined velocity field
    ux = ux_corr
    uy = uy_corr

    # Clean up the edges
    ux[:, :edge_width] = ux[:, edge_width:2*edge_width]
    uy[:, :edge_width] = uy[:, edge_width:2*edge_width]
    ux[:edge_width, :] = ux[edge_width:2*edge_width, :]
    uy[:edge_width, :] = uy[edge_width:2*edge_width, :]

    # Figure 12: Refined Velocity Field
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    step = max(1, min(ux.shape) // 25)
    y, x = np.mgrid[0:ux.shape[0]:step, 0:ux.shape[1]:step]
    plt.quiver(x, y, ux[::step, ::step], uy[::step, ::step], scale=10, angles='xy', scale_units='xy')
    plt.title('Refined Velocity Field')
    plt.axis('equal')

    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(ux**2 + uy**2)
    plt.subplot(222)
    plt.imshow(velocity_magnitude, cmap='jet')
    plt.title('Refined Velocity Magnitude')
    plt.colorbar()

    # Calculate vorticity
    vor = vorticity(ux, uy)
    plt.subplot(223)
    plt.imshow(vor, cmap='RdBu_r')
    plt.title('Refined Vorticity')
    plt.colorbar()

    # Plot streamlines
    plt.subplot(224)
    y, x = np.meshgrid(np.arange(ux.shape[0]), np.arange(ux.shape[1]), indexing='ij')
    plt.streamplot(x, y, ux, uy, density=1.5, color='blue')
    plt.title('Refined Streamlines')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig12.jpg', dpi=150)
    plt.close()

    # Figure 13: Comparison of Initial and Refined Velocity Fields
    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    step = max(1, min(ux0.shape) // 25)
    y, x = np.mgrid[0:ux0.shape[0]:step, 0:ux0.shape[1]:step]
    plt.quiver(x, y, ux0[::step, ::step], uy0[::step, ::step], scale=10, angles='xy', scale_units='xy')
    plt.title('Initial Velocity Field')
    plt.axis('equal')

    plt.subplot(222)
    step = max(1, min(ux.shape) // 25)
    y, x = np.mgrid[0:ux.shape[0]:step, 0:ux.shape[1]:step]
    plt.quiver(x, y, ux[::step, ::step], uy[::step, ::step], scale=10, angles='xy', scale_units='xy')
    plt.title('Refined Velocity Field')
    plt.axis('equal')

    # Calculate initial vorticity
    vor0 = vorticity(ux0, uy0)
    plt.subplot(223)
    plt.imshow(vor0, cmap='RdBu_r')
    plt.title('Initial Vorticity')
    plt.colorbar()

    # Calculate refined vorticity
    plt.subplot(224)
    plt.imshow(vor, cmap='RdBu_r')
    plt.title('Refined Vorticity')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig13.jpg', dpi=150)
    plt.close()

    # Figure 20: Vortex Identification
    plt.figure(figsize=(12, 10))

    # Calculate Q-criterion
    q_criterion = invariant2_factor(ux, uy, 1.0, 1.0)

    # Identify vortex regions using vorticity
    pos_vorticity = vor > np.max(vor) * 0.3  # Positive vorticity regions
    neg_vorticity = vor < np.min(vor) * 0.3  # Negative vorticity regions

    # Count number of distinct vortex regions
    from scipy import ndimage
    labeled_pos, num_pos = ndimage.label(pos_vorticity)
    labeled_neg, num_neg = ndimage.label(neg_vorticity)

    # Combine positive and negative regions for visualization
    vortex_regions = np.zeros_like(vor, dtype=int)
    if num_pos > 0:
        vortex_regions[labeled_pos > 0] = 1
    if num_neg > 0:
        vortex_regions[labeled_neg > 0] = 2

    plt.subplot(221)
    plt.imshow(velocity_magnitude, cmap='jet')
    plt.title('Velocity Magnitude')
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(vor, cmap='RdBu_r')
    plt.title('Vorticity')
    plt.colorbar()

    plt.subplot(223)
    plt.imshow(q_criterion, cmap='jet')
    plt.title('Q-criterion')
    plt.colorbar()

    plt.subplot(224)
    plt.imshow(vortex_regions, cmap='tab20')
    plt.title(f'Identified Vortex Regions: {num_pos + num_neg}')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig20.jpg', dpi=150)
    plt.close()

    # Figure 21: Detailed Vorticity Analysis
    plt.figure(figsize=(12, 10))

    # Plot vorticity with different color maps and thresholds
    plt.subplot(221)
    plt.imshow(vor, cmap='RdBu_r')
    plt.title('Vorticity (RdBu_r)')
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(vor, cmap='jet')
    plt.title('Vorticity (jet)')
    plt.colorbar()

    # Plot vorticity with contour lines
    plt.subplot(223)
    plt.imshow(vor, cmap='RdBu_r')
    plt.contour(vor, levels=10, colors='k', linewidths=0.5)
    plt.title('Vorticity with Contours')
    plt.colorbar()

    # Plot vorticity histogram
    plt.subplot(224)
    plt.hist(vor.flatten(), bins=50)
    plt.title('Vorticity Histogram')
    plt.xlabel('Vorticity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig21.jpg', dpi=150)
    plt.close()

    # Figure 22: Velocity Field Visualization
    plt.figure(figsize=(12, 10))

    # Plot velocity field with different visualization techniques
    plt.subplot(221)
    step = max(1, min(ux.shape) // 25)
    y, x = np.mgrid[0:ux.shape[0]:step, 0:ux.shape[1]:step]
    plt.quiver(x, y, ux[::step, ::step], uy[::step, ::step], scale=10, angles='xy', scale_units='xy')
    plt.title('Velocity Field (Quiver)')
    plt.axis('equal')

    plt.subplot(222)
    y, x = np.meshgrid(np.arange(ux.shape[0]), np.arange(ux.shape[1]), indexing='ij')
    plt.streamplot(x, y, ux, uy, density=1.5, color='blue')
    plt.title('Velocity Field (Streamlines)')
    plt.axis('equal')

    # Plot velocity components
    plt.subplot(223)
    plt.imshow(ux, cmap='RdBu_r')
    plt.title('X-Component of Velocity')
    plt.colorbar()

    plt.subplot(224)
    plt.imshow(uy, cmap='RdBu_r')
    plt.title('Y-Component of Velocity')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig22.jpg', dpi=150)
    plt.close()

    # Figure 23: Velocity Magnitude Analysis
    plt.figure(figsize=(12, 10))

    # Plot velocity magnitude with different color maps
    plt.subplot(221)
    plt.imshow(velocity_magnitude, cmap='jet')
    plt.title('Velocity Magnitude (jet)')
    plt.colorbar()

    plt.subplot(222)
    plt.imshow(velocity_magnitude, cmap='viridis')
    plt.title('Velocity Magnitude (viridis)')
    plt.colorbar()

    # Plot velocity magnitude with contour lines
    plt.subplot(223)
    plt.imshow(velocity_magnitude, cmap='jet')
    plt.contour(velocity_magnitude, levels=10, colors='k', linewidths=0.5)
    plt.title('Velocity Magnitude with Contours')
    plt.colorbar()

    # Plot velocity magnitude histogram
    plt.subplot(224)
    plt.hist(velocity_magnitude.flatten(), bins=50)
    plt.title('Velocity Magnitude Histogram')
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig23.jpg', dpi=150)
    plt.close()

    # Figure 24: Comparison of Initial and Final Results
    plt.figure(figsize=(12, 5))

    # Plot initial and final velocity magnitude
    vel_mag0 = np.sqrt(ux0**2 + uy0**2)
    plt.subplot(121)
    plt.imshow(vel_mag0, cmap='jet')
    plt.title('Initial Velocity Magnitude')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(velocity_magnitude, cmap='jet')
    plt.title('Final Velocity Magnitude')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig24.jpg', dpi=150)
    plt.close()

    print(f"All figures saved to {output_dir}/")

    # Return results for further analysis if needed
    return {
        'ux': ux,
        'uy': uy,
        'ux0': ux0,
        'uy0': uy0,
        'velocity_magnitude': velocity_magnitude,
        'vorticity': vor,
        'q_criterion': q_criterion
    }

if __name__ == "__main__":
    # Run the analysis on White_Oval images
    results_oval = run_piv_analysis(image_pair='white_oval')

    # Uncomment to run the analysis on vortex pair images
    # results_vortex = run_piv_analysis(image_pair='vortex_pair')
