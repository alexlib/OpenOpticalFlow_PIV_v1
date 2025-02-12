import matplotlib.pyplot as plt
import numpy as np

# Assuming I_region1, I_region2, ux0, uy0, Im1, Im2, ux, uy are already defined

# Show the pre-processed images in initial estimation
plt.figure(1)
plt.imshow(I_region1, cmap='gray')
plt.axis('image')
plt.title('Downsampled Image 1')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')

plt.figure(2)
plt.imshow(I_region2, cmap='gray')
plt.axis('image')
plt.title('Downsampled Image 2')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')

# Plot initial velocity vector field and streamlines
plt.figure(3)
gx = 30
offset = 1
# Assuming vis_flow is a custom function to visualize flow
h = vis_flow(ux0, uy0, gx, offset, 3, 'm')
for line in h:
    line.set_color('red')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')
plt.gca().invert_yaxis()
plt.title('Coarse-Grained Velocity Field')

# Plot streamlines
plt.figure(4)
m, n = ux0.shape
x, y = np.meshgrid(np.arange(1, n+1), np.arange(1, m+1))
dn = 10
dm = 10
sx, sy = np.meshgrid(np.arange(1, n+1, dn), np.arange(1, m+1, dm))
h = plt.streamplot(x, y, ux0, uy0, color='blue')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')
plt.gca().invert_yaxis()
plt.title('Coarse-Grained Streamlines')

# Plot the original images
plt.figure(10)
plt.imshow(Im1, cmap='gray')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')
plt.title('Image 1')

plt.figure(11)
plt.imshow(Im2, cmap='gray')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')
plt.title('Image 2')

# Plot refined velocity vector field
plt.figure(12)
gx = 50
offset = 1
h = vis_flow(ux, uy, gx, offset, 5, 'm')
for line in h:
    line.set_color('red')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')
plt.gca().invert_yaxis()
plt.title('Refined Velocity Field')

# Plot streamlines
plt.figure(13)
m, n = ux.shape
x, y = np.meshgrid(np.arange(1, n+1), np.arange(1, m+1))
dn = 10
dm = 10
sx, sy = np.meshgrid(np.arange(1, n+1, dn), np.arange(1, m+1, dm))
h = plt.streamplot(x, y, ux, uy, color='blue')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.axis('image')
plt.gca().invert_yaxis()
plt.title('Refined Streamlines')

plt.show()
