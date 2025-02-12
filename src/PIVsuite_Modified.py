import numpy as np
import matplotlib.pyplot as plt
from pivsuite import piv_analyze_image_pair  # Assuming you have a similar function in Python

# Load Images
im1 = './images/vortex_pair_particles_1.tif'
im2 = './images/vortex_pair_particles_2.tif'

piv_data1 = piv_analyze_image_pair(im1, im2)

# Save Velocity Field
ux_cc = piv_data1['U']
uy_cc = piv_data1['V']

u_mag_cc = np.sqrt(ux_cc**2 + uy_cc**2)
u_max_cc = np.max(u_mag_cc)
u_mag_cc = u_mag_cc / u_max_cc

plt.figure(1)
vlims = [0, 1]
plt.imshow(u_mag_cc, vmin=vlims[0], vmax=vlims[1], cmap='jet')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')
plt.gca().invert_yaxis()
plt.colorbar()
plt.show()

# Uncomment and modify the following lines to save the data if needed
# np.savetxt('ux_cc.dat', ux_cc)
# np.savetxt('uy_cc.dat', uy_cc)


