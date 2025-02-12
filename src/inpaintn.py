import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.ndimage import distance_transform_edt as bwdist
import matplotlib.pyplot as plt

def inpaintn(x, n=100, y0=None):
    """
    INPAINTN Inpaint over missing data in N-D array
    Y = INPAINTN(X) replaces the missing data in X by extra/interpolating
    the non-missing elements. The non finite values (NaN or Inf) in X are
    considered as missing data. X can be any N-D array.
    """
    x = x.astype(float)
    sizx = x.shape
    d = x.ndim
    Lambda = np.zeros(sizx)
    for i in range(d):
        siz0 = np.ones(d, dtype=int)
        siz0[i] = sizx[i]
        Lambda += np.cos(np.pi * (np.reshape(np.arange(1, sizx[i] + 1), siz0) - 1) / sizx[i])
    Lambda = -2 * (d - Lambda)

    W = np.isfinite(x)
    if y0 is not None:
        y = y0
        s0 = 0
    else:
        if np.any(~W):
            y, s0 = InitialGuess(x, np.isfinite(x))
        else:
            return x
    x[~W] = 0

    s = np.logspace(s0, -3, n)
    RF = 2  # relaxation factor
    Lambda = Lambda ** 2

    for i in range(n):
        Gamma = 1 / (1 + s[i] * Lambda)
        y = RF * idctn(Gamma * dctn(W * (x - y) + y)) + (1 - RF) * y

    y[W] = x[W]
    return y

def InitialGuess(y, I):
    if 'scipy.ndimage' in sys.modules:
        z, L = bwdist(I, return_indices=True)
        z = y.copy()
        z[~I] = y[tuple(L[:, ~I])]
        s0 = 3
    else:
        print('Warning: BWDIST (Image Processing Toolbox) does not exist. The initial guess may not be optimal; additional iterations can thus be required to ensure complete convergence. Increase N value if necessary.')
        z = y.copy()
        z[~I] = np.mean(y[I])
        s0 = 6
    return z, s0

def RunTheExample():
    from scipy.io import loadmat
    data = loadmat('wind.mat')
    x, y, z = data['x'], data['y'], data['z']
    u, v, w = data['u'], data['v'], data['w']
    xmin, xmax = np.min(x), np.max(x)
    zmin, ymax = np.min(z), np.max(y)
    vel0 = interp3(np.sqrt(u**2 + v**2 + w**2), 1, 'cubic')
    x = interp3(x, 1)
    y = interp3(y, 1)
    z = interp3(z, 1)
    I = np.random.permutation(vel0.size)
    velNaN = vel0.copy()
    velNaN.flat[I[:round(vel0.size * 0.9)]] = np.nan
    vel = inpaintn(velNaN)
    plt.subplot(221)
    plt.imshow(velNaN[:, :, 15], aspect='equal')
    plt.axis('off')
    plt.title('Corrupt plane, z = 15')
    plt.subplot(222)
    plt.imshow(vel[:, :, 15], aspect='equal')
    plt.axis('off')
    plt.title('Reconstructed plane, z = 15')
    plt.subplot(223)
    plt.contourf(x, y, z, vel0, levels=[xmin, 100, xmax], cmap='viridis')
    plt.title('Original data compared with...')
    plt.subplot(224)
    plt.contourf(x, y, z, vel, levels=[xmin, 100, xmax], cmap='viridis')
    plt.title('... reconstructed data')
    plt.show()

if __name__ == "__main__":
    RunTheExample()
