import numpy as np
import matplotlib.pyplot as plt

def plot_streamlines(VVx, VVy, density=1.0, color='b'):
    """
    Visualize flow field using streamlines.

    Parameters:
        VVx (numpy.ndarray): x-component of velocity field
        VVy (numpy.ndarray): y-component of velocity field
        density (float): Density of the streamlines
        color (str): Color of the streamlines

    Returns:
        strm: Handle to the streamplot
    """
    sy, sx = VVx.shape
    y, x = np.mgrid[:sy, :sx]

    # Create streamplot
    strm = plt.streamplot(x, y, VVx, VVy, density=density, color=color)
    plt.axis([0, sx, 0, sy])

    return strm

def vis_flow(VVx, VVy, gx=25, offset=0, mag=1, col='b'):
    """
    Visualize flow field using quiver plot.
    Direct translation of MATLAB's vis_flow function.

    Parameters:
        VVx (numpy.ndarray): x-component of velocity field
        VVy (numpy.ndarray): y-component of velocity field
        gx (int): Grid spacing parameter
        offset (int): Offset for sampling grid points
        mag (float): Magnitude scaling factor
        col (str): Color of the arrows

    Returns:
        H: Handle to the quiver plot (for compatibility with MATLAB-style scripts)
        or (Vx, Vy) if two outputs are requested
    """
    # function [Ox,Oy] = vis_flow (VVx, VVy, gx, offset, mag, col);
    #
    # if (nargin<3)
    #     gx = 25;
    # end
    # if (nargin<4)
    #     offset = 0;
    # end
    # if (nargin<5)
    #     mag = 1;
    # end
    # if (nargin<6)
    #     col = 'b';
    # end

    # Default parameters are handled by Python's function signature

    # [sy sx] = size(VVx);
    # if (gx==0)
    #     jmp = 1;
    # else
    #     jmp = floor(sx/gx);
    #     jmp = jmp + (jmp==0);
    # end

    sy, sx = VVx.shape
    if gx == 0:
        jmp = 1
    else:
        jmp = int(sx / gx)  # floor division in Python
        jmp = jmp + (jmp == 0)  # Ensure jmp is at least 1

    # indx = (offset+1):jmp:sx;
    # c = 1;
    # CX = [];
    # CY = [];

    indx = np.arange(offset+1, sx+1, jmp)  # +1 for MATLAB 1-based indexing
    indx = indx - 1  # Convert back to Python 0-based indexing
    c = 0  # Python uses 0-based indexing

    # for j=(1+offset):jmp:sy
    #     Vx(c,:) = VVx(j,indx);
    #     Vy(c,:) = VVy(j,indx);
    #     CX(c,:) = indx;
    #     %CY(c,:) = ones(size(indx)).*(sy-j+1);
    #     CY(c,:) = ones(size(indx)).*j;
    #     c = c+1;
    # end

    # Initialize arrays
    rows = len(range(1+offset, sy+1, jmp))
    cols = len(indx)
    Vx = np.zeros((rows, cols))
    Vy = np.zeros((rows, cols))
    CX = np.zeros((rows, cols))
    CY = np.zeros((rows, cols))

    # Fill arrays
    for j in range(1+offset, sy+1, jmp):
        j_idx = j - 1  # Convert to Python 0-based indexing
        Vx[c, :] = VVx[j_idx, indx]
        Vy[c, :] = VVy[j_idx, indx]
        CX[c, :] = indx
        CY[c, :] = np.ones(indx.shape) * j
        c += 1

    # if (isnan(Vx(1,1)))
    #     Vx(1,1) = 1;
    #     Vy(1,1) = 0;
    #     CX(1,1) = 1;
    #     CY(1,1) = 1;
    # end

    if np.isnan(Vx[0, 0]):
        Vx[0, 0] = 1
        Vy[0, 0] = 0
        CX[0, 0] = 1
        CY[0, 0] = 1

    # M = ~isnan(Vx) & ~isnan(Vy);
    # H = quiver (CX(M), CY(M), Vx(M), Vy(M), mag);
    # s = size(VVx);
    # axis ([0 s(2) 0 s(1)]);
    # set (H, 'Color', col);

    M = ~np.isnan(Vx) & ~np.isnan(Vy)
    H = plt.quiver(CX[M], CY[M], Vx[M], Vy[M], scale=1.0/mag,
                  scale_units='xy', angles='xy', color=col)

    plt.axis([0, sx, 0, sy])

    # switch nargout
    #     case 0
    #         clear Ox;
    #         clear Oy;
    #     case 1
    #         Ox = H;
    #     otherwise
    #         Ox = Vx;
    #         Oy = Vy;
    # end

    # In Python, we'll just return H by default
    return H