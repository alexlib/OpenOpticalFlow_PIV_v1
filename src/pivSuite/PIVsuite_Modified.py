
"""
PIVsuite CC Algorithm Modified - Python implementation

This script runs the PIV analysis on a pair of images using the PIVsuite algorithm.
It matches the MATLAB implementation from the OpenOpticalFlow_PIV_v1 repository.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from pivSuite.pivAnalyzeImagePair import piv_analyze_image_pair

# Clear previous figures
plt.close('all')

# Load Images
im1_path = './examples/data/vortex_pair_particles_1.tif'
im2_path = './examples/data/vortex_pair_particles_2.tif'

# Load images
try:
    im1 = io.imread(im1_path)
    im2 = io.imread(im2_path)
    print(f"Successfully loaded images: {im1.shape}, {im2.shape}")
except Exception as e:
    print(f"Error loading images: {e}")
    raise

# Run PIV analysis with default parameters
# This matches the MATLAB implementation where no parameters are provided
piv_data = piv_analyze_image_pair(im1, im2)

# Extract velocity components
ux_cc = piv_data['U']
uy_cc = piv_data['V']

# Calculate velocity magnitude
u_mag_cc = np.sqrt(ux_cc**2 + uy_cc**2)

# Normalize velocity magnitude
u_max_cc = np.max(u_mag_cc)
u_mag_cc = u_mag_cc / u_max_cc

# Squeeze any extra dimensions
u_mag_cc = np.squeeze(u_mag_cc)

# Create figure
plt.figure(figsize=(8, 6))

# Plot velocity magnitude with exact settings to match MATLAB output
vlims = [0, 1]  # Velocity limits for colormap
plt.imshow(u_mag_cc, cmap='jet', origin='upper', vmin=vlims[0], vmax=vlims[1])
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')  # Maintain aspect ratio
plt.colorbar()

# Save the figure
plt.savefig('piv_velocity_field.png', dpi=150)
plt.show()

# Optionally save the velocity fields
# np.savetxt('ux_cc.dat', ux_cc)
# np.savetxt('uy_cc.dat', uy_cc)

print("PIV analysis complete. Results saved to piv_velocity_field.png")
