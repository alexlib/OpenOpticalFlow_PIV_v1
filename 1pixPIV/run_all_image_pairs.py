#!/usr/bin/env python3
"""
Script to run optical flow analysis on all image pairs in the images folder.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from glob import glob
import sys
import time

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary modules
from openopticalflow.OpticalFlowPhysics_fun import OpticalFlowPhysics_fun
from openopticalflow.shift_image_fun_refine_1 import shift_image_fun_refine_1
from pivAnalyzeImagePair import piv_analyze_image_pair

def find_image_pairs(image_dir):
    """
    Find all image pairs in the given directory.

    Args:
        image_dir (str): Path to the directory containing the images.

    Returns:
        list: List of tuples containing the paths to the image pairs.
    """
    # Get all image files
    image_files = glob(os.path.join(image_dir, "*.tif"))

    # Group images by base name (removing _1, _2, etc.)
    image_groups = {}
    for image_file in image_files:
        # Extract the base name and the number
        base_name = os.path.basename(image_file)
        match = re.match(r'(.+?)_(\d+)\.tif', base_name)
        if match:
            group_name = match.group(1)
            number = int(match.group(2))

            if group_name not in image_groups:
                image_groups[group_name] = {}

            image_groups[group_name][number] = image_file

    # Create pairs
    image_pairs = []
    for group_name, files in image_groups.items():
        if 1 in files and 2 in files:
            image_pairs.append((files[1], files[2], group_name))

    return image_pairs

def process_image_pair(image1_path, image2_path, output_dir, pair_name):
    """
    Process a pair of images using optical flow analysis.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.
        output_dir (str): Path to the directory where the results will be saved.
        pair_name (str): Name of the image pair.

    Returns:
        None
    """
    print(f"Processing image pair: {pair_name}")

    # Load images
    Im1_original = imageio.imread(image1_path)
    Im2_original = imageio.imread(image2_path)

    # Convert to grayscale if needed
    if len(Im1_original.shape) > 2:
        Im1_original = np.mean(Im1_original, axis=2).astype(np.uint8)
    if len(Im2_original.shape) > 2:
        Im2_original = np.mean(Im2_original, axis=2).astype(np.uint8)

    # Make a copy for processing
    I1 = Im1_original.copy().astype(np.float64)
    I2 = Im2_original.copy().astype(np.float64)

    # Parameters for optical flow
    lambda_1 = 20.0  # Horn-Schunck estimator for initial field
    lambda_2 = 2000.0  # Liu-Shen estimator for refined estimation
    no_iteration = 1  # Number of iterations
    edge_width = 10  # Width of the edge to clean up

    # Initial correlation calculation for a coarse-grained velocity field
    pivPar = {
        'iaSizeX': [64, 16, 8],
        'iaStepX': [32, 8, 4],
        'ccMethod': 'fft'
    }

    print("Running PIV analysis...")
    pivData1 = piv_analyze_image_pair(I1, I2, pivPar)

    ux0 = pivData1['U']
    uy0 = pivData1['V']

    # Handle 3D arrays (if the result is a 3D array with a single time slice)
    if len(ux0.shape) > 2:
        ux0 = ux0[:, :, 0]
        uy0 = uy0[:, :, 0]

    # Resize the initial velocity field
    n0, m0 = ux0.shape
    n, m = I1.shape

    # Generate the shifted image and calculate velocity difference iteratively
    ux = ux0
    uy = uy0
    print(f"Initial velocity field min/max: {np.min(ux):.4f}/{np.max(ux):.4f}, {np.min(uy):.4f}/{np.max(uy):.4f}")

    k = 1
    while k <= no_iteration:
        print(f"\nIteration {k}/{no_iteration}")
        print("Shifting image based on velocity field...")
        Im1_shift, uxI, uyI = shift_image_fun_refine_1(ux, uy, I1, I2)
        print(f"Shifted image shape: {Im1_shift.shape}")
        print(f"Intermediate velocity field min/max: {np.min(uxI):.4f}/{np.max(uxI):.4f}, {np.min(uyI):.4f}/{np.max(uyI):.4f}")

        I1 = Im1_shift.astype(np.float64)
        I2 = I2.astype(np.float64)

        print("Calculating optical flow...")
        dux, duy, vor, dux_horn, duy_horn, error2 = OpticalFlowPhysics_fun(I1, I2, lambda_1, lambda_2)
        print(f"Optical flow correction min/max: {np.min(dux):.4f}/{np.max(dux):.4f}, {np.min(duy):.4f}/{np.max(duy):.4f}")

        ux_corr = uxI + dux
        uy_corr = uyI + duy
        print(f"Corrected velocity field min/max: {np.min(ux_corr):.4f}/{np.max(ux_corr):.4f}, {np.min(uy_corr):.4f}/{np.max(uy_corr):.4f}")

        k += 1

    # Refined velocity field
    ux = ux_corr
    uy = uy_corr
    print(f"\nFinal velocity field shape: {ux.shape}")
    print(f"Final velocity field min/max: {np.min(ux):.4f}/{np.max(ux):.4f}, {np.min(uy):.4f}/{np.max(uy):.4f}")

    # Clean up the edges
    print("Cleaning up edges...")
    ux[:, :edge_width] = ux[:, edge_width:2*edge_width]
    uy[:, :edge_width] = uy[:, edge_width:2*edge_width]
    ux[:edge_width, :] = ux[edge_width:2*edge_width, :]
    uy[:edge_width, :] = uy[edge_width:2*edge_width, :]
    print(f"After edge cleanup min/max: {np.min(ux):.4f}/{np.max(ux):.4f}, {np.min(uy):.4f}/{np.max(uy):.4f}")

    # Plot the results
    print("\nCreating plots...")
    # Create plots similar to plots_set_1 and plots_set_2
    plt.figure(figsize=(12, 10))

    # Plot original images
    print(f"Original image shapes: {Im1_original.shape}, {Im2_original.shape}")
    plt.subplot(2, 2, 1)
    plt.imshow(Im1_original, cmap='gray')
    plt.title('Original Image 1')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(Im2_original, cmap='gray')
    plt.title('Original Image 2')
    plt.colorbar()

    # Plot velocity field
    print(f"Velocity field shape for quiver plot: {ux.shape}, {uy.shape}")
    plt.subplot(2, 2, 3)
    # Use a downsampled version for quiver to avoid overcrowding
    step = max(1, min(ux.shape) // 25)  # Adjust step size based on array size
    print(f"Using step size {step} for quiver plot")

    # Create coordinate grid for quiver plot
    y, x = np.mgrid[0:ux.shape[0]:step, 0:ux.shape[1]:step]
    print(f"Quiver grid shape: {x.shape}, {y.shape}")

    # Plot quiver with proper coordinates
    plt.quiver(x, y, ux[::step, ::step], uy[::step, ::step], scale=10, angles='xy', scale_units='xy')
    plt.title('Velocity Field')
    plt.axis('equal')

    # Plot velocity magnitude
    velocity_magnitude = np.sqrt(ux**2 + uy**2)
    print(f"Velocity magnitude min/max: {np.min(velocity_magnitude):.4f}/{np.max(velocity_magnitude):.4f}")
    plt.subplot(2, 2, 4)
    plt.imshow(velocity_magnitude, cmap='jet')
    plt.title('Velocity Magnitude')
    plt.colorbar()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, f"{pair_name}_results.png")
    plt.savefig(output_path)
    print(f"Results saved to: {output_path}")

    # Close the figure to free memory
    plt.close()

def main():
    """
    Main function to run the script.
    """
    # Get the path to the images directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    image_dir = os.path.join(repo_dir, "images")

    # Create output directory for results
    output_dir = os.path.join(repo_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    # Find all image pairs
    image_pairs = find_image_pairs(image_dir)
    print(f"Found {len(image_pairs)} image pairs:")
    for i, (img1, img2, name) in enumerate(image_pairs):
        print(f"{i+1}. {name}: {os.path.basename(img1)} and {os.path.basename(img2)}")

    # Process each image pair
    for img1, img2, name in image_pairs:
        start_time = time.time()
        process_image_pair(img1, img2, output_dir, name)
        elapsed_time = time.time() - start_time
        print(f"Processing time: {elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
