import numpy as np
from scipy.ndimage import gaussian_filter
from smoothn import smoothn  # Assuming you have a Python equivalent of smoothn.m

def piv_smooth(piv_data, piv_par):
    """
    piv_smooth - smooth displacement field in a set of PIV data

    Usage:
        piv_data = piv_smooth(piv_data, piv_par)

    Inputs:
        piv_data ... (dict) structure containing more detailed results. Required fields are
            X, Y ... position, at which velocity/displacement is calculated
            U, V ... displacements in x and y direction
            Status ... matrix describing status of velocity vectors (for values, see Outputs section)
        piv_par ... (dict) parameters defining the evaluation. Following fields are considered:
                    smMethod ... defines smoothing method. Possible values are:
                        'none' ... do not perform smoothing
                        'smoothn' ... uses smoothn function
                        'gauss' ... uses Gaussian kernel
                    smSigma ... amount of smoothing
                    smSize ... size of filter (applies only to Gaussian filter)

    Outputs:
        piv_data  ... (dict) structure containing more detailed results. If piv_data was non-empty at the input, its
                  fields are preserved. Following field is modified:
            U, V ... x and y components of the velocity/displacement vector (with replaced NaN's)
            smSu, smSv ... smoothing parameter (if 'smoothn' method is used)
            Status ... matrix with status of velocity vectors (uint8). Bits have this coding:
                1 ... masked (set by pivInterrogate)
                2 ... cross-correlation failed (set by pivCrossCorr)
                4 ... peak detection failed (set by pivCrossCorr)
                8 ... indicated as spurious by median test (set by pivValidate)
               16 ... interpolated (set by pivReplaced)
               32 ... smoothed (set by pivSmooth)
    """
    is_single = piv_data['U'].dtype == np.float32

    Uin = piv_data['U'].astype(np.float64)
    Vin = piv_data['V'].astype(np.float64)
    status = piv_data['Status']
    if Uin.ndim > 2:
        method = piv_par['smMethodSeq']
        sigma = piv_par['smSigmaSeq']
        bit = 9
    else:
        method = piv_par['smMethod']
        sigma = piv_par['smSigma']
        bit = 6

    if method.lower() == 'none':
        U = Uin
        V = Vin
        smSu = np.nan
        smSv = np.nan
    elif method.lower() == 'smoothn':
        aux_nan_u = np.isnan(Uin)
        aux_nan_v = np.isnan(Vin)
        if not np.isnan(sigma):
            U, smSu = smoothn(Uin, sigma, robust=True)
            V, smSv = smoothn(Vin, sigma, robust=True)
        else:
            U, smSu = smoothn(Uin)
            V, smSv = smoothn(Vin)
        U[aux_nan_u] = np.nan
        V[aux_nan_v] = np.nan
        status[~(aux_nan_u | aux_nan_v)] = np.bitwise_or(status[~(aux_nan_u | aux_nan_v)], bit)
    elif method.lower() == 'gauss':
        if Uin.ndim > 2:
            print('Error (pivSmooth): smMethod "Gauss" is allowed only for data on image pair, not on sequences.')
            return
        h = gaussian_filter(Uin, sigma)
        U = gaussian_filter(Uin, sigma)
        V = gaussian_filter(Vin, sigma)
        status[~np.isnan(U) | ~np.isnan(V)] = np.bitwise_or(status[~np.isnan(U) | ~np.isnan(V)], bit)
        aux_nan_u = np.isnan(U) & ~np.isnan(Uin)
        aux_nan_v = np.isnan(V) & ~np.isnan(Vin)
        U[aux_nan_u] = Uin[aux_nan_u]
        V[aux_nan_v] = Vin[aux_nan_v]
        smSu = np.nan
        smSv = np.nan
    else:
        print('warning: unknown smoothing method')
        smSu = np.nan
        smSv = np.nan

    if is_single:
        piv_data['U'] = U.astype(np.float32)
        piv_data['V'] = V.astype(np.float32)
    else:
        piv_data['U'] = U
        piv_data['V'] = V

    piv_data['Status'] = status.astype(np.uint16)
    piv_data['smU'] = smSu
    piv_data['smV'] = smSv

    return piv_data
