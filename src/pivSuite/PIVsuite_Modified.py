
from pivSuite.piv_parameters import PIVParameters
from pivSuite.pivAnalyzeImagePair import piv_analyze_image_pair
import numpy as np
import matplotlib.pyplot as plt

# Create PIV parameters
piv_par = PIVParameters(
    iaSizeX=[64, 32, 16, 16],  # Interrogation area sizes
    iaSizeY=[64, 32, 16, 16],  # Interrogation area sizes Y
    ccMethod=['fft'] + ['dcn'] * 3,  # Cross-correlation method for each pass
    anNpasses=4               # Number of passes
)

# Load and analyze image pair
im1 = './images/vortex_pair_particles_1.tif'
im2 = './images/vortex_pair_particles_2.tif'

# Run PIV analysis
piv_data = piv_analyze_image_pair(im1, im2, piv_par)

# Plot results
plt.figure(figsize=(10, 8))
plt.imshow(np.sqrt(piv_data['U']**2 + piv_data['V']**2), cmap='jet')
plt.colorbar(label='Velocity magnitude')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('PIV Velocity Field')
plt.axis('image')
plt.gca().invert_yaxis()
plt.show()
