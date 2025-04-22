import numpy as np
import matplotlib.pyplot as plt
from openopticalflow.vorticity import vorticity
from openopticalflow.invariant2_factor import invariant2_factor
from visualization.vis_flow import vis_flow

def plots_set_2(ux, uy):
    """
    Create plots similar to MATLAB's plots_set_2.m script.

    Parameters:
        ux (np.ndarray): x-component of velocity field
        uy (np.ndarray): y-component of velocity field
    """
    # Calculate the velocity magnitude
    u_mag = np.sqrt(ux**2 + uy**2)
    u_max = np.max(u_mag)
    u_mag = u_mag / u_max

    # Calculate vorticity
    vor = vorticity(ux, uy)
    vor_max = np.max(np.abs(vor))
    vor = vor / vor_max

    # Calculate the 2nd invariant
    Q = invariant2_factor(ux, uy, 1, 1)

    # Plot velocity magnitude field with streamlines
    plt.figure()
    ulims = [0, 1]
    plt.imshow(u_mag, vmin=ulims[0], vmax=ulims[1], cmap='jet')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()  # Equivalent to MATLAB's set(gca,'YDir','reverse')
    plt.title('Velocity Magnitude Field')
    plt.colorbar()

    # Add streamlines to the same figure
    m, n = ux.shape
    x, y = np.meshgrid(np.arange(n), np.arange(m))
    dn = 10
    dm = 10
    # Note: Python's streamplot is similar to MATLAB's streamslice
    h = plt.streamplot(x, y, ux, uy, density=1.5, color='yellow')

    # Plot Vorticity field with streamlines
    plt.figure()
    vlims = [-1, 1]
    plt.imshow(vor, vmin=vlims[0], vmax=vlims[1], cmap='RdBu_r')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.title('Vorticity Field')
    plt.colorbar()

    # Add streamlines to the same figure
    h = plt.streamplot(x, y, ux, uy, density=1.5, color='blue')

    # Plot Vorticity field with velocity vectors
    plt.figure()
    plt.imshow(vor, vmin=vlims[0], vmax=vlims[1], cmap='RdBu_r')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.title('Vorticity Field')
    plt.colorbar()

    # Add velocity vectors to the same figure
    gx = 50
    offset = 1
    h = vis_flow(ux, uy, gx, offset, 3, 'm')
    plt.setp(h, color='black')

    # Plot Velocity magnitude field with velocity vectors
    plt.figure()
    vlims = [0, 1]
    plt.imshow(u_mag, vmin=vlims[0], vmax=vlims[1], cmap='jet')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.title('Velocity Magnitude Field')
    plt.colorbar()

    # Add velocity vectors to the same figure
    gx = 50
    offset = 1
    h = vis_flow(ux, uy, gx, offset, 3, 'm')
    plt.setp(h, color='black')

    # Plot Q field
    plt.figure()
    Qlims = [0, 0.1]
    plt.imshow(Q, vmin=Qlims[0], vmax=Qlims[1], cmap='jet')
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.axis('image')
    plt.gca().invert_yaxis()
    plt.title('Q Field')
    plt.colorbar()