"""
function [cc] = dcn(X1,X2,MaxD)
% computes cross-correlation using discrete convolution
Nx = size(X1,2);
Ny = size(X1,1);
cc = zeros(Ny,Nx);
% create variables defining where is cc(0,0)
dx0 = Nx/2;
dy0 = Ny/2;
if rem(Nx,2) == 0
    dx0 = dx0+1;
else
    dx0 = dx0+0.5;
end
if rem(Ny,2) == 0
    dy0 = dy0+1;
else
    dy0 = dy0+0.5;
end
% pad IAs
X1p = zeros(Ny+2*MaxD,Nx+2*MaxD);
X2p = zeros(Ny+2*MaxD,Nx+2*MaxD);
X1p(MaxD+1:MaxD+Ny,MaxD+1:MaxD+Nx) = X1;
X2p(MaxD+1:MaxD+Ny,MaxD+1:MaxD+Nx) = X2;
% convolve
for kx = -MaxD:MaxD
    for ky = -MaxD:MaxD
        if abs(kx)+abs(ky)>MaxD, continue; end
        cc(dy0+ky,dx0+kx) = sum(sum(...
            X2p(ky+MaxD+1 : ky+MaxD+Ny,  kx+MaxD+1 : kx+MaxD+Nx) .* ...
            X1p(   MaxD+1 : MaxD+Ny,        MaxD+1 : MaxD+Nx)));
    end
end

end
"""

import numpy as np
#from numba import njit

# @njit
def dcn(X1, X2, MaxD):
    """
    Computes cross-correlation using discrete convolution.
    """
    Nx = X1.shape[1]
    Ny = X1.shape[0]
    cc = np.zeros((Ny, Nx))

    # create variables defining where is cc(0,0)
    dx0 = Nx / 2
    dy0 = Ny / 2
    if Nx % 2 == 0:
        dx0 += 1
    else:
        dx0 += 0.5
    if Ny % 2 == 0:
        dy0 += 1
    else:
        dy0 += 0.5

    # pad IAs
    X1p = np.zeros((Ny + 2 * MaxD, Nx + 2 * MaxD))
    X2p = np.zeros((Ny + 2 * MaxD, Nx + 2 * MaxD))
    X1p[MaxD:MaxD + Ny, MaxD:MaxD + Nx] = X1
    X2p[MaxD:MaxD + Ny, MaxD:MaxD + Nx] = X2

    # convolve
    for kx in range(-MaxD, MaxD + 1):
        for ky in range(-MaxD, MaxD + 1):
            if abs(kx) + abs(ky) > MaxD:
                continue
            cc[int(dy0 + ky), int(dx0 + kx)] = np.sum(
                X2p[ky + MaxD:ky + MaxD + Ny, kx + MaxD:kx + MaxD + Nx] *
                X1p[MaxD:MaxD + Ny, MaxD:MaxD + Nx]
            )

    return cc