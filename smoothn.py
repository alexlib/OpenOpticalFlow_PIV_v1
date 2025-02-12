def smoothn(*args):

    # %SMOOTHN Robust spline smoothing for 1-D to N-D data.
    # %   SMOOTHN provides a fast, automatized and robust discretized smoothing
    # %   spline for data of arbitrary dimension.
    # %
    # %   Z = SMOOTHN(Y) automatically smoothes the uniformly-sampled array Y. Y
    # %   can be any N-D noisy array (time series, images, 3D data,...). Non
    # %   finite data (NaN or Inf) are treated as missing values.
    # %
    # %   Z = SMOOTHN(Y,S) smoothes the array Y using the smoothing parameter S.
    # %   S must be a real positive scalar. The larger S is, the smoother the
    # %   output will be. If the smoothing parameter S is omitted (see previous
    # %   option) or empty (i.e. S = []), it is automatically determined by
    # %   minimizing the generalized cross-validation (GCV) score.
    # %
    # %   Z = SMOOTHN(Y,W) or Z = SMOOTHN(Y,W,S) smoothes Y using a weighting
    # %   array W of positive values, that must have the same size as Y. Note
    # %   that a nil weight corresponds to a missing value.
    # %
    # %   Robust smoothing
    # %   ----------------
    # %   Z = SMOOTHN(...,'robust') carries out a robust smoothing that minimizes
    # %   the influence of outlying data.
    # %
    # %   [Z,S] = SMOOTHN(...) also returns the calculated value for the
    # %   smoothness parameter S so that you can fine-tune the smoothing
    # %   subsequently if needed.
    # %
    # %   An iteration process is used in the presence of weighted and/or missing
    # %   values. Z = SMOOTHN(...,OPTION_NAME,OPTION_VALUE) smoothes with the
    # %   termination parameters specified by OPTION_NAME and OPTION_VALUE. They
    # %   can contain the following criteria:
    # %       -----------------
    # %       TolZ:       Termination tolerance on Z (default = 1e-3)
    # %                   TolZ must be in ]0,1[
    # %       MaxIter:    Maximum number of iterations allowed (default = 100)
    # %       Initial:    Initial value for the iterative process (default =
    # %                   original data)
    # %       Weights:    Weighting function for robust smoothing:
    # %                   'bisquare' (default), 'talworth' or 'cauchy'
    # %       -----------------
    # %   Syntax: [Z,...] = SMOOTHN(...,'MaxIter',500,'TolZ',1e-4,'Initial',Z0);
    # %
    # %   [Z,S,EXITFLAG] = SMOOTHN(...) returns a boolean value EXITFLAG that
    # %   describes the exit condition of SMOOTHN:
    # %       1       SMOOTHN converged.
    # %       0       Maximum number of iterations was reached.
    # %
    # %   Notes
    # %   -----
    # %   The N-D (inverse) discrete cosine transform functions <a
    # %   href="matlab:web('http://www.biomecardio.com/matlab/dctn.html')"
    # %   >DCTN</a> and <a
    # %   href="matlab:web('http://www.biomecardio.com/matlab/idctn.html')"
    # %   >IDCTN</a> are required.
    # %
    # %   Reference
    # %   --------- 
    # %   Garcia D, Robust smoothing of gridded data in one and higher dimensions
    # %   with missing values. Computational Statistics & Data Analysis, 2010. 
    # %   <a
    # %   href="matlab:web('http://www.biomecardio.com/pageshtm/publi/csda10.pdf')">PDF download</a>
    # %
    # %   Examples:
    # %   --------
    # %   % 1-D example
    # %   x = linspace(0,100,2^8);
    # %   y = cos(x/10)+(x/50).^2 + randn(size(x))/10;
    # %   y([70 75 80]) = [5.5 5 6];
    # %   z = smoothn(y); % Regular smoothing
    # %   zr = smoothn(y,'robust'); % Robust smoothing
    # %   subplot(121), plot(x,y,'r.',x,z,'k','LineWidth',2)
    # %   axis square, title('Regular smoothing')
    # %   subplot(122), plot(x,y,'r.',x,zr,'k','LineWidth',2)
    # %   axis square, title('Robust smoothing')
    # %
    # %   % 2-D example
    # %   xp = 0:.02:1;
    # %   [x,y] = meshgrid(xp);
    # %   f = exp(x+y) + sin((x-2*y)*3);
    # %   fn = f + randn(size(f))*0.5;
    # %   fs = smoothn(fn);
    # %   subplot(121), surf(xp,xp,fn), zlim([0 8]), axis square
    # %   subplot(122), surf(xp,xp,fs), zlim([0 8]), axis square
    # %
    # %   % 2-D example with missing data
    # %   n = 256;
    # %   y0 = peaks(n);
    # %   y = y0 + randn(size(y0))*2;
    # %   I = randperm(n^2);
    # %   y(I(1:n^2*0.5)) = NaN; % lose 1/2 of data
    # %   y(40:90,140:190) = NaN; % create a hole
    # %   z = smoothn(y); % smooth data
    # %   subplot(2,2,1:2), imagesc(y), axis equal off
    # %   title('Noisy corrupt data')
    # %   subplot(223), imagesc(z), axis equal off
    # %   title('Recovered data ...')
    # %   subplot(224), imagesc(y0), axis equal off
    # %   title('... compared with original data')
    # %
    # %   % 3-D example
    # %   [x,y,z] = meshgrid(-2:.2:2);
    # %   xslice = [-0.8,1]; yslice = 2; zslice = [-2,0];
    # %   vn = x.*exp(-x.^2-y.^2-z.^2) + randn(size(x))*0.06;
    # %   subplot(121), slice(x,y,z,vn,xslice,yslice,zslice,'cubic')
    # %   title('Noisy data')
    # %   v = smoothn(vn);
    # %   subplot(122), slice(x,y,z,v,xslice,yslice,zslice,'cubic')
    # %   title('Smoothed data')
    # %
    # %   % Cardioid
    # %   t = linspace(0,2*pi,1000);
    # %   x = 2*cos(t).*(1-cos(t)) + randn(size(t))*0.1;
    # %   y = 2*sin(t).*(1-cos(t)) + randn(size(t))*0.1;
    # %   z = smoothn(complex(x,y));
    # %   plot(x,y,'r.',real(z),imag(z),'k','linewidth',2)
    # %   axis equal tight
    # %
    # %   % Cellular vortical flow
    # %   [x,y] = meshgrid(linspace(0,1,24));
    # %   Vx = cos(2*pi*x+pi/2).*cos(2*pi*y);
    # %   Vy = sin(2*pi*x+pi/2).*sin(2*pi*y);
    # %   Vx = Vx + sqrt(0.05)*randn(24,24); % adding Gaussian noise
    # %   Vy = Vy + sqrt(0.05)*randn(24,24); % adding Gaussian noise
    # %   I = randperm(numel(Vx));
    # %   Vx(I(1:30)) = (rand(30,1)-0.5)*5; % adding outliers
    # %   Vy(I(1:30)) = (rand(30,1)-0.5)*5; % adding outliers
    # %   Vx(I(31:60)) = NaN; % missing values
    # %   Vy(I(31:60)) = NaN; % missing values
    # %   Vs = smoothn(complex(Vx,Vy),'robust'); % automatic smoothing
    # %   subplot(121), quiver(x,y,Vx,Vy,2.5), axis square
    # %   title('Noisy velocity field')
    # %   subplot(122), quiver(x,y,real(Vs),imag(Vs)), axis square
    # %   title('Smoothed velocity field')
    # %
    # %   See also DCTSMOOTH, DCTN, IDCTN.
    # %
    # %   -- Damien Garcia -- 2009/03, revised 2012/04
    # %   website: <a
    # %   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>

    # % Check input arguments
    # % narginchk(1,12);

    # Test & prepare the variables
    #---
    k = 0
    while k < len(args) and not isinstance(args[k], str):
        k += 1
    #---
    # y = array to be smoothed
    y = np.array(args[0], dtype=float)
    sizy = y.shape
    noe = np.prod(sizy)  # number of elements
    if noe < 2:
        z = y
        s = None
        exitflag = True
        return z, s, exitflag
    #---
    # Smoothness parameter and weights
    W = np.ones(sizy)
    s = None
    if k == 2:
        if args[1] is None or np.isscalar(args[1]):  # smoothn(y,s)
            s = args[1]  # smoothness parameter
        else:  # smoothn(y,W)
            W = np.array(args[1])  # weight array
    elif k == 3:  # smoothn(y,W,s)
        W = np.array(args[1])  # weight array
        s = args[2]  # smoothness parameter
    if W.shape != sizy:
        raise ValueError('Arrays for data and weights (Y and W) must have same size.')
    elif s is not None and (not np.isscalar(s) or s < 0):
        raise ValueError('The smoothing parameter S must be a scalar >=0')
    #---
    # "Maximal number of iterations" criterion
    try:
        MaxIter = args[args.index('MaxIter') + 1]
    except (ValueError, IndexError):
        MaxIter = 100  # default value for MaxIter
    if not isinstance(MaxIter, int) or MaxIter < 1:
        raise ValueError('MaxIter must be an integer >=1')
    #---
    # "Tolerance on smoothed output" criterion
    try:
        TolZ = args[args.index('TolZ') + 1]
    except (ValueError, IndexError):
        TolZ = 1e-3  # default value for TolZ
    if not isinstance(TolZ, float) or TolZ <= 0 or TolZ >= 1:
        raise ValueError('TolZ must be in ]0,1[')
    #---
    # "Initial Guess" criterion
    try:
        z0 = args[args.index('Initial') + 1]
        isinitial = True
    except (ValueError, IndexError):
        isinitial = False
    if isinitial and (not isinstance(z0, np.ndarray) or z0.shape != sizy):
        raise ValueError('Z0 must be a valid initial guess for Z')
    #---
    # "Weighting function" criterion (for robust smoothing)
    try:
        weightstr = args[args.index('Weights') + 1].lower()
    except (ValueError, IndexError):
        weightstr = 'bisquare'  # default weighting function
    if not isinstance(weightstr, str):
        raise ValueError('A valid weighting function must be chosen')
    #---
    # Weights. Zero weights are assigned to not finite values (Inf or NaN),
    # (Inf/NaN values = missing data).
    IsFinite = np.isfinite(y)
    nof = np.count_nonzero(IsFinite)  # number of finite elements
    W = W * IsFinite
    if np.any(W < 0):
        raise ValueError('Weights must all be >=0')
    else:
        W = W / np.max(W)
    #---
    # Weighted or missing data?
    isweighted = np.any(W < 1)
    #---
    # Robust smoothing?
    isrobust = 'robust' in args
    #---
    # Automatic smoothing?
    isauto = s is None
    #---
    # DCTN and IDCTN are required
    # test4DCTNandIDCTN

    # Create the Lambda tensor
    #---
    # Lambda contains the eigenvalues of the difference matrix used in this
    # penalized least squares process (see CSDA paper for details)
    d = y.ndim
    m = 2
    Lambda = np.zeros(sizy)
    for i in range(d):
        siz0 = np.ones(d, dtype=int)
        siz0[i] = sizy[i]
        Lambda += np.cos(np.pi * (np.reshape(np.arange(1, sizy[i] + 1), siz0) - 1) / sizy[i])
    Lambda = 2 * (d - Lambda)
    if not isauto:
        Gamma = 1 / (1 + s * Lambda**m)

    # Upper and lower bound for the smoothness parameter
    # The average leverage (h) is by definition in [0 1]. Weak smoothing occurs
    # if h is close to 1, while over-smoothing appears when h is near 0. Upper
    # and lower bounds for h are given to avoid under- or over-smoothing. See
    # equation relating h to the smoothness parameter (Equation #12 in the
    # referenced CSDA paper).
    N = np.sum(np.array(sizy) != 1)  # tensor rank of the y-array
    hMin = 1e-6
    hMax = 0.99
    sMinBnd = (((1 + np.sqrt(1 + 8 * hMax**(2 / N))) / 4 / hMax**(2 / N))**2 - 1) / 16
    sMaxBnd = (((1 + np.sqrt(1 + 8 * hMin**(2 / N))) / 4 / hMin**(2 / N))**2 - 1) / 16

    # Initialize before iterating
    #---
    Wtot = W
    #--- Initial conditions for z
    if isweighted:
        #--- With weighted/missing data
        # An initial guess is provided to ensure faster convergence. For that
        # purpose, a nearest neighbor interpolation followed by a coarse
        # smoothing are performed.
        #---
        if isinitial:  # an initial guess (z0) has been already given
            z = z0
        else:
            z = initial_guess(y, IsFinite)
    else:
        z = np.zeros(sizy)
    #---
    z0 = z
    y[~IsFinite] = 0  # arbitrary values for missing y-data
    #---
    tol = 1
    RobustIterativeProcess = True
    RobustStep = 1
    nit = 0
    #--- Error on p. Smoothness parameter s = 10^p
    errp = 0.001
    #--- Relaxation factor RF: to speedup convergence
    RF = 1 + 0.75 * isweighted

    # Main iterative process
    #---
    while RobustIterativeProcess:
        #--- "amount" of weights (see the function GCVscore)
        aow = np.sum(Wtot) / noe  # 0 < aow <= 1
        #---
        while tol > TolZ and nit < MaxIter:
            nit += 1
            DCTy = dctn(Wtot * (y - z) + z)
            if isauto and not np.log2(nit) % 1:
                #---
                # The generalized cross-validation (GCV) method is used.
                # We seek the smoothing parameter S that minimizes the GCV
                # score i.e. S = Argmin(GCVscore).
                # Because this process is time-consuming, it is performed from
                # time to time (when the step number - nit - is a power of 2)
                #---
                s = 10**fminbound(gcv, np.log10(sMinBnd), np.log10(sMaxBnd), xtol=errp)
                Gamma = 1 / (1 + s * Lambda**m)
            z = RF * idctn(Gamma * DCTy) + (1 - RF) * z

            # if no weighted/missing data => tol=0 (no iteration)
            tol = isweighted * np.linalg.norm(z0 - z) / np.linalg.norm(z)

            z0 = z  # re-initialization
        exitflag = nit < MaxIter

        if isrobust:  #-- Robust Smoothing: iteratively re-weighted process
            #--- average leverage
            h = np.sqrt(1 + 16 * s)
            h = np.sqrt(1 + h) / np.sqrt(2) / h
            h = h**N
            #--- take robust weights into account
            Wtot = W * robust_weights(y - z, IsFinite, h, weightstr)
            #--- re-initialize for another iterative weighted process
            isweighted = True
            tol = 1
            nit = 0
            #---
            RobustStep += 1
            RobustIterativeProcess = RobustStep < 4  # 3 robust steps are enough.
        else:
            RobustIterativeProcess = False  # stop the whole process

    # Warning messages
    #---
    if isauto:
        if abs(np.log10(s) - np.log10(sMinBnd)) < errp:
            pass
            # print(f'Warning: S = {s:.3e}: the lower bound for S has been reached. Put S as an input variable if required.')
        elif abs(np.log10(s) - np.log10(sMaxBnd)) < errp:
            print(f'Warning: S = {s:.3e}: the upper bound for S has been reached. Put S as an input variable if required.')
    if not exitflag:
        print(f'Warning: Maximum number of iterations ({MaxIter}) has been exceeded. Increase MaxIter option or decrease TolZ value.')


%% GCV score
#---
def gcv(p):
    """
    Search the smoothing parameter s that minimizes the GCV score.
    """
    s = 10**p
    Gamma = 1 / (1 + s * Lambda**m)
    #--- RSS = Residual sum-of-squares
    if aow > 0.9:  # aow = 1 means that all of the data are equally weighted
        # very much faster: does not require any inverse DCT
        RSS = np.linalg.norm(DCTy * (Gamma - 1))**2
    else:
        # take account of the weights to calculate RSS:
        yhat = idctn(Gamma * DCTy)
        RSS = np.linalg.norm(np.sqrt(Wtot[IsFinite]) * (y[IsFinite] - yhat[IsFinite]))**2
    #---
    TrH = np.sum(Gamma)
    GCVscore = RSS / nof / (1 - TrH / noe)**2
    return GCVscore

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

% % Test for DCTN and IDCTN
% function commented as dctn and idctn functions were inserted directly to this m-file
% function test4DCTNandIDCTN
%     if ~exist('dctn','file')
%         error('MATLAB:smoothn:MissingFunction',...
%             ['DCTN and IDCTN are required. Download DCTN <a href="matlab:web(''',...
%             'http://www.biomecardio.com/matlab/dctn.html'')">here</a>.'])
%     elseif ~exist('idctn','file')
%         error('MATLAB:smoothn:MissingFunction',...
%             ['DCTN and IDCTN are required. Download IDCTN <a href="matlab:web(''',...
%             'http://www.biomecardio.com/matlab/idctn.html'')">here</a>.'])
%     end
% end

import numpy as np
from scipy.ndimage import distance_transform_edt

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
        z[ceil(siz[k] / 10):] = 0
        z = np.moveaxis(z, 0, k)
    z, _ = idctn(z, w=w)
    
    return z

%% N-D Discrete cosine transform

import numpy as np
from scipy.fftpack import dct, idct

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


import numpy as np
from scipy.fftpack import dct, idct

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


