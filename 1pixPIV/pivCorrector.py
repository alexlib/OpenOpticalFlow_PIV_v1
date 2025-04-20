import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from piv_parameters import PIVParameters

def inpaint_nans(arr):
    # Simple inpainting function to replace NaNs with the mean of the array
    arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    return arr

def piv_corrector(piv_data, piv_data0, piv_par):
    # piv_corrector - correct the changes to the velocity field for attenuation

    piv_data_out = piv_data.copy()

    # Convert piv_par to PIVParameters if it's not already
    piv_par = PIVParameters.from_tuple_or_dict(piv_par)

    # exit, if corrector is not required, or if initial velocity data are not available
    if (not hasattr(piv_par, 'crAmount') or \
            piv_par.crAmount == 0 or \
            'U' not in piv_data0 or \
            'V' not in piv_data0 or \
            'X' not in piv_data0 or \
            'Y' not in piv_data0):
        return piv_data_out

    # simplify names
    X = piv_data['X']
    Y = piv_data['Y']
    U = piv_data['U']
    V = piv_data['V']
    W0 = piv_data['ccW']

    # create interpolators for initial and actual deformation fields
    U0_estimator = RegularGridInterpolator((piv_data0['X'], piv_data0['Y']), inpaint_nans(piv_data0['U']), method='spline')
    V0_estimator = RegularGridInterpolator((piv_data0['X'], piv_data0['Y']), inpaint_nans(piv_data0['V']), method='spline')
    U_estimator = RegularGridInterpolator((piv_data['X'], piv_data['Y']), inpaint_nans(piv_data['U']), method='spline')
    V_estimator = RegularGridInterpolator((piv_data['X'], piv_data['Y']), inpaint_nans(piv_data['V']), method='spline')

    # compute for all image pixels
    XX, YY = np.meshgrid(np.arange(1, piv_data['imSizeX'] + 1), np.arange(1, piv_data['imSizeY'] + 1), indexing='ij')
    U0_im = U0_estimator((XX, YY))
    V0_im = V0_estimator((XX, YY))
    U_im = U_estimator((XX, YY))
    V_im = V_estimator((XX, YY))

    # get image mask
    auxM1 = piv_data.get('imMaskArray1', np.ones_like(XX))
    auxM2 = piv_data.get('imMaskArray2', np.ones_like(XX))
    M = auxM1 * auxM2

    # interpolate initial guess for final grid
    U0 = U0_estimator((X, Y))
    V0 = V0_estimator((X, Y))

    # loop over all IAs and correct the velocity
    for kx in range(X.shape[1]):
        for ky in range(Y.shape[0]):
            # get mask of the current IA
            auxM = M[
                Y[ky, kx] - (piv_par.get_parameter('iaSizeY') - 1) // 2:Y[ky, kx] + (piv_par.get_parameter('iaSizeY') - 1) // 2 + 1,
                X[ky, kx] - (piv_par.get_parameter('iaSizeX') - 1) // 2:X[ky, kx] + (piv_par.get_parameter('iaSizeX') - 1) // 2 + 1
            ]
            # get actual weighting function, respecting the mask
            W = W0 * auxM
            # apply corrector only if less than 75% of "weighted" pixels are masked
            if np.sum(W) > 0.25 * np.sum(W0):
                # normalize weighting function
                W = W / np.sum(W)
                # get the weighted averages of velocity estimates in the IA
                U0_avg = np.sum(U0_im[
                    Y[ky, kx] - (piv_par.get_parameter('iaSizeY') - 1) // 2:Y[ky, kx] + (piv_par.get_parameter('iaSizeY') - 1) // 2 + 1,
                    X[ky, kx] - (piv_par.get_parameter('iaSizeX') - 1) // 2:X[ky, kx] + (piv_par.get_parameter('iaSizeX') - 1) // 2 + 1
                ] * W)
                V0_avg = np.sum(V0_im[
                    Y[ky, kx] - (piv_par.get_parameter('iaSizeY') - 1) // 2:Y[ky, kx] + (piv_par.get_parameter('iaSizeY') - 1) // 2 + 1,
                    X[ky, kx] - (piv_par.get_parameter('iaSizeX') - 1) // 2:X[ky, kx] + (piv_par.get_parameter('iaSizeX') - 1) // 2 + 1
                ] * W)
                U_avg = np.sum(U_im[
                    Y[ky, kx] - (piv_par.get_parameter('iaSizeY') - 1) // 2:Y[ky, kx] + (piv_par.get_parameter('iaSizeY') - 1) // 2 + 1,
                    X[ky, kx] - (piv_par.get_parameter('iaSizeX') - 1) // 2:X[ky, kx] + (piv_par.get_parameter('iaSizeX') - 1) // 2 + 1
                ] * W)
                V_avg = np.sum(V_im[
                    Y[ky, kx] - (piv_par.get_parameter('iaSizeY') - 1) // 2:Y[ky, kx] + (piv_par.get_parameter('iaSizeY') - 1) // 2 + 1,
                    X[ky, kx] - (piv_par.get_parameter('iaSizeX') - 1) // 2:X[ky, kx] + (piv_par.get_parameter('iaSizeX') - 1) // 2 + 1
                ] * W)
                # correct the velocity change
                piv_data_out['U'][ky, kx] = U[ky, kx] + piv_par.crAmount * ((U[ky, kx] - U0[ky, kx]) - (U_avg - U0_avg))
                piv_data_out['V'][ky, kx] = V[ky, kx] + piv_par.crAmount * ((V[ky, kx] - V0[ky, kx]) - (V_avg - V0_avg))

    return piv_data_out