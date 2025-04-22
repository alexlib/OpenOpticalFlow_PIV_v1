import numpy as np
import matplotlib.pyplot as plt

def vis_flow(vx, vy, gx=30, offset=1, mag=1, units='m'):
    """
    Visualize flow field using quiver plot.
    Direct translation of MATLAB's vis_flow function.
    
    Parameters:
        vx (numpy.ndarray): x-component of velocity field
        vy (numpy.ndarray): y-component of velocity field
        gx (int): Grid spacing parameter
        offset (int): Offset for sampling grid points
        mag (float): Magnitude scaling factor
        units (str): Units for the velocity field
    
    Returns:
        h: Handle to the quiver plot (for compatibility with MATLAB-style scripts)
    """
    sy, sx = vx.shape
    
    # Calculate jump size based on grid spacing
    if gx == 0:
        jmp = 1
    else:
        jmp = max(1, sx // gx)
    
    # Create grid points
    indx = np.arange(offset, sx, jmp)
    indy = np.arange(offset, sy, jmp)
    X, Y = np.meshgrid(indx, indy)
    
    # Extract velocity components at the sampled points
    U = vx[indy][:, indx]
    V = vy[indy][:, indx]
    
    # Calculate velocity magnitude for scaling
    vel_magnitude = np.sqrt(U**2 + V**2)
    max_vel = np.max(vel_magnitude)
    
    # Set scale factor (similar to MATLAB's quiver scaling)
    if max_vel > 0:
        scale = max_vel / mag
    else:
        scale = 1.0
    
    # Plot quiver
    h = plt.quiver(X, Y, U, V, 
                  scale=scale,
                  scale_units='xy',
                  angles='xy',
                  color='red')
    
    return h

def plot_streamlines(x, y, vx, vy, density=1.5):
    """
    Plot streamlines of a vector field.
    Similar to MATLAB's streamslice function.
    
    Parameters:
        x (numpy.ndarray): x-coordinates grid
        y (numpy.ndarray): y-coordinates grid
        vx (numpy.ndarray): x-component of velocity field
        vy (numpy.ndarray): y-component of velocity field
        density (float): Controls the density of streamlines
    
    Returns:
        h: Handle to the streamlines plot
    """
    h = plt.streamplot(x, y, vx, vy, density=density, color='blue')
    return h
