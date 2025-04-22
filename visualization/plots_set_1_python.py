"""
plots_set_1_python.py

This script creates the same figures as the MATLAB plots_set_1.m script,
with similar default settings to match MATLAB's visualization style.
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
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
from openopticalflow.vis_flow import vis_flow

# Import PIV analysis functions
from pivSuite.pivAnalyzeImagePair import piv_analyze_image_pair

def create_matlab_style_plots(image_pair='white_oval', save_figures=True):
    """
    Create figures similar to those in MATLAB plots_set_1.m
    
    Args:
        image_pair (str): Name of the image pair to use. Options:
            - 'vortex_pair': Vortex pair images
            - 'white_oval': White oval images
        save_figures (bool): Whether to save the figures to files
    """
    print(f"Creating MATLAB-style plots for {image_pair} images...")
    
    # Create output directory for figures
    output_dir = 'matlab_style_figures'
    if save_figures:
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
    
    # Create figures similar to MATLAB plots_set_1.m
    
    # Figure 1: Downsampled Image 1
    plt.figure(figsize=(8, 6))
    plt.imshow(I_region1.astype(np.uint8), cmap='gray')
    plt.colorbar()
    plt.axis('image')
    plt.title('Downsampled Image 1')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'downsampled_image1.png'), dpi=150)
    plt.show()
    
    # Figure 2: Downsampled Image 2
    plt.figure(figsize=(8, 6))
    plt.imshow(I_region2.astype(np.uint8), cmap='gray')
    plt.colorbar()
    plt.axis('image')
    plt.title('Downsampled Image 2')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'downsampled_image2.png'), dpi=150)
    plt.show()
    
    # Figure 3: Coarse-Grained Velocity Field
    plt.figure(figsize=(8, 6))
    # Create a custom vis_flow function similar to MATLAB's
    def python_vis_flow(u, v, gx, offset, scale_factor, units='m'):
        """Python implementation of MATLAB's vis_flow function"""
        m, n = u.shape
        
        # Create grid points
        x = np.arange(offset, n, gx)
        y = np.arange(offset, m, gx)
        X, Y = np.meshgrid(x, y)
        
        # Sample velocity at grid points
        U = u[Y.flatten(), X.flatten()].reshape(Y.shape)
        V = v[Y.flatten(), X.flatten()].reshape(Y.shape)
        
        # Scale velocities for visualization
        scale = np.max(np.sqrt(U**2 + V**2)) / scale_factor
        if scale == 0:
            scale = 1
        
        # Plot quiver
        q = plt.quiver(X, Y, U, V, color='red', scale=scale, scale_units='inches')
        
        return q
    
    # Use a grid spacing of 30 pixels as in MATLAB
    gx = 30
    offset = 1
    q = python_vis_flow(ux0, uy0, gx, offset, 3)
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Coarse-Grained Velocity Field')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'coarse_velocity_field.png'), dpi=150)
    plt.show()
    
    # Figure 4: Coarse-Grained Streamlines
    plt.figure(figsize=(8, 6))
    m, n = ux0.shape
    y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    
    # Use streamplot with density similar to MATLAB's streamslice
    plt.streamplot(x, y, ux0, uy0, density=1.5, color='blue')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Coarse-Grained Streamlines')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'coarse_streamlines.png'), dpi=150)
    plt.show()
    
    # Figure 10: Original Image 1
    plt.figure(figsize=(8, 6))
    plt.imshow(Im1.astype(np.uint8), cmap='gray')
    plt.colorbar()
    plt.axis('image')
    plt.title('Image 1')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'original_image1.png'), dpi=150)
    plt.show()
    
    # Figure 11: Original Image 2
    plt.figure(figsize=(8, 6))
    plt.imshow(Im2.astype(np.uint8), cmap='gray')
    plt.colorbar()
    plt.axis('image')
    plt.title('Image 2')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'original_image2.png'), dpi=150)
    plt.show()
    
    # Figure 12: Refined Velocity Field
    plt.figure(figsize=(8, 6))
    # Use a grid spacing of 50 pixels as in MATLAB
    gx = 50
    offset = 1
    q = python_vis_flow(ux, uy, gx, offset, 5)
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Refined Velocity Field')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'refined_velocity_field.png'), dpi=150)
    plt.show()
    
    # Figure 13: Refined Streamlines
    plt.figure(figsize=(8, 6))
    m, n = ux.shape
    y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    
    # Use streamplot with density similar to MATLAB's streamslice
    plt.streamplot(x, y, ux, uy, density=1.5, color='blue')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Refined Streamlines')
    if save_figures:
        plt.savefig(os.path.join(output_dir, 'refined_streamlines.png'), dpi=150)
    plt.show()
    
    print(f"All figures created{' and saved to ' + output_dir if save_figures else ''}.")
    
    # Return results for further analysis if needed
    return {
        'ux': ux,
        'uy': uy,
        'ux0': ux0,
        'uy0': uy0
    }

if __name__ == "__main__":
    # Create MATLAB-style plots for White_Oval images
    results = create_matlab_style_plots(image_pair='white_oval', save_figures=True)
