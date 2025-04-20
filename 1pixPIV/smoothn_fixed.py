import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import fminbound
import math

def smoothn(y, w=None, s=None, robust=False, MaxIter=100, TolZ=1e-3, axis=None):
    """
    Smooths noisy data.
    
    Parameters
    ----------
    y : array_like
        Data to be smoothed.
    w : array_like, optional
        Weights for each element in y. 1/w defines the variance of the
        observations. If w is not specified, all weights are set to 1.
    s : float, optional
        Smoothing parameter. If not given, s is determined automatically.
    robust : bool, optional
        If True, robust smoothing is performed. This option is useful when
        the data contains outliers.
    MaxIter : int, optional
        Maximum number of iterations allowed.
    TolZ : float, optional
        Convergence tolerance.
    axis : int or sequence of ints, optional
        Axis along which the smooth operation is applied. If None, all
        dimensions are smoothed.
    
    Returns
    -------
    z : ndarray
        Smoothed data.
    s : float
        Smoothing parameter used.
    exitflag : bool
        True if convergence was reached.
    """
    # Check input arguments
    if axis is not None:
        raise ValueError('Axis option not implemented yet.')
    
    y = np.asarray(y, dtype=float)
    sizy = y.shape
    
    # Default weights
    if w is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != y.shape:
            raise ValueError('Y and W must have the same size.')
    
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.isfinite(y)
    nof = np.count_nonzero(IsFinite)  # number of finite elements
    W = w * IsFinite
    if np.any(W < 0):
        raise ValueError('Weights must all be >=0')
    
    # Weighted or missing data?
    isweighted = np.any(W != 1)
    
    # Robust smoothing?
    RobustIterativeProcess = False
    if robust:
        RobustIterativeProcess = True
        RobustStep = 1
        nit = 0
        robustFlag = 1
        oneMinusRobFrac = 1 - 1e-6
    
    # Create initial conditions
    z = initial_guess(y, IsFinite)
    
    # Smoothness parameter
    N = np.sum(np.array(sizy) > 1)  # tensor rank of the y-array
    lambda_default = 10 ** (-N - 1)  # Default regularization parameter
    if s is None:
        s = lambda_default
    
    # Upper and lower bound for the smoothness parameter
    sMinBnd = np.spacing(1)  # smallest positive spacing
    sMaxBnd = 0.1
    
    # Initialize difference, weights, etc.
    DiffZ = np.zeros_like(y)
    if isweighted:
        Wtot = W
    else:
        Wtot = np.ones_like(y)
    
    # Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted
    
    # Initialize zero quantities
    Lambda = np.zeros(y.ndim)
    for i in range(y.ndim):
        siz0 = np.ones(y.ndim, dtype=int)
        siz0[i] = sizy[i]
        Lambda_i = np.cos(np.pi * np.arange(sizy[i]) / sizy[i])
        Lambda[i] = Lambda_i.reshape(siz0)
    
    # Tensor product of Lambda_i
    Lambda = np.sum(Lambda, axis=0) - y.ndim
    
    # Initial condition
    W = W.reshape(-1)
    Wtot = Wtot.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    
    # Initialization of DCT
    siz = sizy
    d = y.ndim
    
    # Permute and reshape variables to speed up Matlab's fft
    y = y.reshape(sizy)
    z = z.reshape(sizy)
    Wtot = Wtot.reshape(sizy)
    
    # Compute initial DCT and a few constants
    DCTy = dctn(Wtot * y)
    
    if isweighted:
        # Normalization
        y = y / np.sqrt(Wtot)
        z = z / np.sqrt(Wtot)
        DCTy = DCTy / np.sqrt(Wtot)
    
    # Estimate the noise standard deviation
    if np.all(IsFinite):
        # No weighted/missing data
        res = y - z
        sigma2 = np.mean(res ** 2)
    else:
        # Weighted/missing data
        res = (y - z) * np.sqrt(Wtot)
        sigma2 = np.mean(res[IsFinite] ** 2)
    
    # Main loop
    exitflag = False
    nit = 0
    
    if RobustIterativeProcess:
        # Robust smoothing: iteratively re-weighted process
        # Compute weights for robust smoothing
        nof = np.sum(IsFinite)
        noe = nof
        
        # Smoothing parameter
        s = 1.5 * lambda_default
        
        # Low iteration count for first passes
        MaxIter = 3
        
        # Robustness parameter
        h = 0.5
        
        # Automatic smoothing?
        if s is None:
            s = lambda_default
        
        # Main iterative process
        tol = 1
        robustIt = 0
        robustMaxIter = 100
        
        # Robust iterative process
        while RobustIterativeProcess and robustIt < robustMaxIter:
            robustIt += 1
            DCTy = dctn(Wtot * y)
            
            while tol > TolZ and nit < MaxIter:
                nit += 1
                DCTz = DCTy / (1 + s * Lambda ** 2)
                z = idctn(DCTz)
                
                # Residual
                res = y - z
                
                # Compute new weights
                if robustFlag:
                    # Compute new weights based on computed residuals
                    h = 1.5 * np.median(np.abs(res[IsFinite]))
                    h = max(h, 1e-6)
                    
                    # Get weights
                    W = robust_weights(res, IsFinite, h, 'bisquare')
                    
                    # Recompute weighted residuals
                    Wtot = W * w
                    Wtot = Wtot / np.mean(Wtot[IsFinite])
                    DCTy = dctn(Wtot * y)
                
                # Check convergence
                if np.sum(np.abs(Wtot * res)) < TolZ:
                    tol = 0
                else:
                    tol = np.sum(np.abs(DCTz - DCTz0)) / np.sum(np.abs(DCTz))
                
                DCTz0 = DCTz
            
            # Re-initialize for next iteration
            MaxIter = 100
            TolZ = 1e-5
            tol = 1
            
            # Stop the process?
            if robustIt >= 1:
                # Smoothness parameter
                aow = np.sum(Wtot) / noe  # 0 < aow <= 1
                
                # Compute the smoothness parameter (Generalized Cross Validation)
                s = 10 ** fminbound(gcv, np.log10(sMinBnd), np.log10(sMaxBnd), xtol=1e-6)
                
                # End of the robust iterative process?
                if robustIt >= 3:
                    RobustIterativeProcess = False
    else:
        # Not robust
        # Smoothing parameter
        if s is None:
            # GCV method
            s = 10 ** fminbound(gcv, np.log10(sMinBnd), np.log10(sMaxBnd), xtol=1e-6)
        
        # Compute the influence of the input parameters
        Gamma = 1 / (1 + s * Lambda ** 2)
        
        # Filter
        z = idctn(Gamma * DCTy)
    
    # Reshape z
    z = z.reshape(sizy)
    
    # Compute additional output arguments if needed
    exitflag = nit < MaxIter
    
    if not exitflag:
        print(f'Warning: Maximum number of iterations ({MaxIter}) has been exceeded. Increase MaxIter option or decrease TolZ value.')
    
    return z, s, exitflag

def robust_weights(r, I, h, wstr):
    """
    Weights for robust smoothing.
    
    Parameters
    ----------
    r : array_like
        Residuals.
    I : array_like
        Boolean array indicating finite values in r.
    h : float
        Average leverage.
    wstr : str
        Weighting function ('cauchy', 'talworth', or 'bisquare').
    
    Returns
    -------
    W : ndarray
        Robust weights.
    """
    MAD = np.median(np.abs(r[I] - np.median(r[I])))  # median absolute deviation
    u = np.abs(r / (1.4826 * MAD) / np.sqrt(1 - h))  # studentized residuals
    
    if wstr == 'cauchy':
        c = 2.385
        W = 1 / (1 + (u / c) ** 2)  # Cauchy weights
    elif wstr == 'talworth':
        c = 2.795
        W = u < c  # Talworth weights
    elif wstr == 'bisquare':
        c = 4.685
        W = (1 - (u / c) ** 2) ** 2 * ((u / c) < 1)  # bisquare weights
    else:
        raise ValueError('A valid weighting function must be chosen')
    
    W[np.isnan(W)] = 0
    return W

def initial_guess(y, I):
    """
    Initial guess with weighted/missing data.
    
    Parameters
    ----------
    y : array_like
        Input array.
    I : array_like
        Boolean array indicating finite values in y.
    
    Returns
    -------
    z : ndarray
        Initial guess array.
    """
    if not np.all(I):
        try:
            # Nearest neighbor interpolation (in case of missing values)
            _, L = distance_transform_edt(I, return_indices=True)
            z = y.copy()
            z[~I] = y[tuple(L[:, ~I])]
        except ImportError:
            # If distance_transform_edt does not exist, NaN values are all replaced with the same scalar.
            z = y.copy()
            z[~I] = np.mean(y[I])
            print('Warning: distance_transform_edt (from scipy.ndimage) does not exist. '
                  'The initial guess may not be optimal; additional iterations can thus be required '
                  'to ensure complete convergence. Increase "MaxIter" criterion if necessary.')
    else:
        z = y.copy()
    
    # Coarse fast smoothing using one-tenth of the DCT coefficients
    siz = z.shape
    z, w = dctn(z)
    for k in range(z.ndim):
        z = np.moveaxis(z, k, 0)
        z[math.ceil(siz[k] / 10):] = 0
        z = np.moveaxis(z, 0, k)
    z, _ = idctn(z, w=w)
    
    return z

def dctn(y, dim=None, w=None):
    """
    N-D discrete cosine transform.
    Y = DCTN(X) returns the discrete cosine transform of X. The array Y is
    the same size as X and contains the discrete cosine transform
    coefficients. This transform can be inverted using IDCTN.

    DCTN(X, DIM) applies the DCTN operation across the dimension DIM.

    Parameters
    ----------
    y : array_like
        Input array.
    dim : int, optional
        Dimension along which the DCT is applied. If not given, the DCT is applied along all dimensions.
    w : list of array_like, optional
        Weights used by the program. If DCTN is required for several large arrays of same size, the weights can be reused to make the algorithm faster.

    Returns
    -------
    y : ndarray
        The transformed array.
    w : list of array_like
        The weights used by the program.
    """
    y = np.asarray(y, dtype=float)
    sizy = y.shape

    if dim is not None:
        assert isinstance(dim, int) and dim > 0, 'DIM must be a positive integer scalar within indexing range.'

    if dim is None:
        y = np.squeeze(y)
    dimy = y.ndim

    if y.ndim == 1:
        dimy = 1
        if y.shape[0] == 1:
            if dim == 1:
                return y, w
            elif dim == 2:
                dim = 1
            y = y.T
        elif dim == 2:
            return y, w

    if w is None:
        w = [None] * dimy
        for d in range(dimy):
            if dim is not None and d != dim - 1:
                continue
            n = y.shape[d] if dimy > 1 else y.size
            w[d] = np.exp(1j * np.arange(n) * np.pi / (2 * n))

    if not np.isrealobj(y):
        y = dctn(y.real, dim, w) + 1j * dctn(y.imag, dim, w)
    else:
        for d in range(dimy):
            if dim is not None and d != dim - 1:
                y = np.moveaxis(y, 0, -1)
                continue
            siz = y.shape
            n = siz[0]
            y = y.reshape(n, -1)
            y = y * w[d]
            y[0, :] /= np.sqrt(2)
            y = np.fft.fft(y, axis=0).real / np.sqrt(2 * n)
            I = np.arange(1, n + 1) * 0.5 + 0.5
            I[1::2] = n - I[0::2] + 1
            y = y[I.astype(int) - 1, :]
            y = y.reshape(siz)
            y = np.moveaxis(y, 0, -1)

    return y.reshape(sizy), w

def idctn(y, dim=None, w=None):
    """
    N-D inverse discrete cosine transform.
    X = IDCTN(Y) inverts the N-D DCT transform, returning the original
    array if Y was obtained using Y = DCTN(X).

    IDCTN(X, DIM) applies the IDCTN operation across the dimension DIM.

    Parameters
    ----------
    y : array_like
        Input array.
    dim : int, optional
        Dimension along which the IDCT is applied. If not given, the IDCT is applied along all dimensions.
    w : list of array_like, optional
        Weights used by the program. If IDCTN is required for several large arrays of same size, the weights can be reused to make the algorithm faster.

    Returns
    -------
    y : ndarray
        The transformed array.
    w : list of array_like
        The weights used by the program.
    """
    y = np.asarray(y, dtype=float)
    sizy = y.shape

    if dim is not None:
        assert isinstance(dim, int) and dim > 0, 'DIM must be a positive integer scalar within indexing range.'

    if dim is None:
        y = np.squeeze(y)
    dimy = y.ndim

    if y.ndim == 1:
        dimy = 1
        if y.shape[0] == 1:
            if dim == 1:
                return y, w
            elif dim == 2:
                dim = 1
            y = y.T
        elif dim == 2:
            return y, w

    if w is None:
        w = [None] * dimy
        for d in range(dimy):
            if dim is not None and d != dim - 1:
                continue
            n = y.shape[d] if dimy > 1 else y.size
            w[d] = np.exp(1j * np.arange(n) * np.pi / (2 * n))

    if not np.isrealobj(y):
        y = idctn(y.real, dim, w) + 1j * idctn(y.imag, dim, w)
    else:
        for d in range(dimy):
            if dim is not None and d != dim - 1:
                y = np.moveaxis(y, 0, -1)
                continue
            siz = y.shape
            n = siz[0]
            y = y.reshape(n, -1)
            y = y * w[d]
            y[0, :] /= np.sqrt(2)
            y = np.fft.ifft(y, axis=0).real * np.sqrt(2 * n)
            I = np.arange(1, n + 1) * 0.5 + 0.5
            I[1::2] = n - I[0::2] + 1
            y = y[I.astype(int) - 1, :]
            y = y.reshape(siz)
            y = np.moveaxis(y, 0, -1)

    return y.reshape(sizy), w

def gcv(p, Lambda=None, DCTy=None, Wtot=None, IsFinite=None, y=None, nof=None, noe=None, m=2):
    """
    Search the smoothing parameter s that minimizes the GCV score.
    
    Parameters
    ----------
    p : float
        Log10 of the smoothing parameter.
    Lambda, DCTy, Wtot, IsFinite, y, nof, noe : array_like
        Variables needed for the GCV computation.
    m : int, optional
        Order of the difference of the penalty (default is 2).
    
    Returns
    -------
    GCVscore : float
        The GCV score.
    """
    s = 10**p
    Gamma = 1 / (1 + s * Lambda**m)
    
    # RSS = Residual sum-of-squares
    if np.sum(Wtot) / noe > 0.9:  # aow = 1 means that all of the data are equally weighted
        # Very much faster: does not require any inverse DCT
        RSS = np.linalg.norm(DCTy * (Gamma - 1))**2
    else:
        # Take account of the weights to calculate RSS
        yhat = idctn(Gamma * DCTy)[0]
        RSS = np.linalg.norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite]))**2
    
    # Generalized Cross-Validation score
    TrH = np.sum(Gamma)
    GCVscore = RSS / nof / (1 - TrH / noe)**2
    
    return GCVscore
