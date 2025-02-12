import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from inpaint import inpaint_nans, inpaint_nans3
from inpaintn import inpaintn

def piv_replace(piv_data, piv_par):
    """
    Replace displacement vectors containing NaN with values coherent with their neighborhood.
    
    Parameters:
        piv_data (dict): Dictionary containing detailed results.
            Required fields are:
                X, Y: Position at which velocity/displacement is calculated.
                U, V: Displacements in x and y direction.
                Status: Matrix describing status of velocity vectors.
        piv_par (dict): Parameters defining the evaluation.
            Required field is:
                rpMethod: Specifies how the spurious vectors are replaced.
    
    Returns:
        piv_data (dict): Updated dictionary with replaced NaN values.
    """
    
    single_type = piv_data['U'].dtype == np.float32

    X0 = piv_data['X'].astype(np.float64)
    Y0 = piv_data['Y'].astype(np.float64)
    U = piv_data['U'].astype(np.float64)
    V = piv_data['V'].astype(np.float64)
    status = piv_data['Status']

    method = piv_par['rpMethod']

    if U.shape[2] == 1 and method.lower() in ['lineart', 'naturalt', 'inpaintt']:
        method = method[:-1]

    contains_nan = np.isnan(U) | np.isnan(V)
    masked = (status & 1).astype(bool)

    if method.lower() == 'none':
        filled_u = U
        filled_v = V
    elif method.lower() in ['linear', 'natural']:
        X = X0
        Y = Y0
        filled_u = np.full_like(U, np.nan)
        filled_v = np.full_like(V, np.nan)
        for kt in range(U.shape[2]):
            flat_x = X.flatten()
            flat_y = Y.flatten()
            flat_u = U[:, :, kt].flatten()
            flat_v = V[:, :, kt].flatten()
            ok = ~np.isnan(flat_u) & ~np.isnan(flat_v)
            flat_x = flat_x[ok]
            flat_y = flat_y[ok]
            flat_u = flat_u[ok]
            flat_v = flat_v[ok]
            if method.lower() == 'linear':
                interp_u = LinearNDInterpolator((flat_x, flat_y), flat_u)
                interp_v = LinearNDInterpolator((flat_x, flat_y), flat_v)
            elif method.lower() == 'natural':
                interp_u = NearestNDInterpolator((flat_x, flat_y), flat_u)
                interp_v = NearestNDInterpolator((flat_x, flat_y), flat_v)
            filled_u[:, :, kt] = interp_u(X, Y)
            filled_v[:, :, kt] = interp_v(X, Y)
    elif method.lower() in ['lineart', 'naturalt']:
        X = np.full_like(U, np.nan)
        Y = np.full_like(U, np.nan)
        T = np.full_like(U, np.nan)
        if 'imPairNo' in piv_data:
            for kt in range(T.shape[2]):
                T[:, :, kt] = piv_data['imPairNo'][kt]
                X[:, :, kt] = X0
                Y[:, :, kt] = Y0
        else:
            for kt in range(T.shape[2]):
                T[:, :, kt] = kt
                X[:, :, kt] = X0
                Y[:, :, kt] = Y0
        flat_x = X.flatten()
        flat_y = Y.flatten()
        flat_t = T.flatten()
        flat_u = U.flatten()
        flat_v = V.flatten()
        ok = ~np.isnan(flat_u) & ~np.isnan(flat_v)
        flat_x = flat_x[ok]
        flat_y = flat_y[ok]
        flat_t = flat_t[ok]
        flat_u = flat_u[ok]
        flat_v = flat_v[ok]
        if method.lower() == 'lineart':
            interp_u = LinearNDInterpolator((flat_x, flat_y, flat_t), flat_u)
            interp_v = LinearNDInterpolator((flat_x, flat_y, flat_t), flat_v)
        elif method.lower() == 'naturalt':
            interp_u = NearestNDInterpolator((flat_x, flat_y, flat_t), flat_u)
            interp_v = NearestNDInterpolator((flat_x, flat_y, flat_t), flat_v)
        filled_u = interp_u(X, Y, T)
        filled_v = interp_v(X, Y, T)
    elif method.lower() == 'inpaint':
        filled_u = np.full_like(U, np.nan)
        filled_v = np.full_like(V, np.nan)
        for kt in range(U.shape[2]):
            filled_u[:, :, kt] = inpaint_nans(U[:, :, kt], 4)
            filled_v[:, :, kt] = inpaint_nans(V[:, :, kt], 4)
    elif method.lower() == 'inpaintt':
        filled_u = inpaint_nans3(U, 1)
        filled_v = inpaint_nans3(V, 1)
    elif method.lower() == 'inpaintgarcia':
        filled_u = np.full_like(U, np.nan)
        filled_v = np.full_like(V, np.nan)
        for kt in range(U.shape[2]):
            filled_u[:, :, kt] = inpaintn(U[:, :, kt])
            filled_v[:, :, kt] = inpaintn(V[:, :, kt])
    elif method.lower() == 'inpaintgarciat':
        filled_u = inpaintn(U)
        filled_v = inpaintn(V)
    else:
        print('Error (piv_replace): Unknown replacement method.')
        filled_u = U
        filled_v = V

    U = filled_u
    V = filled_v

    U[masked] = np.nan
    V[masked] = np.nan

    replaced = contains_nan & ~masked
    if single_type:
        piv_data['U'] = U.astype(np.float32)
        piv_data['V'] = V.astype(np.float32)
    else:
        piv_data['U'] = U.astype(np.float64)
        piv_data['V'] = V.astype(np.float64)

    if piv_data['U'].shape[2] == 1:
        status[replaced] = np.bitwise_or(status[replaced], 16)
        piv_data['replacedN'] = np.sum(replaced)
    else:
        status[replaced] = np.bitwise_or(status[replaced], 128)
        piv_data['replacedN'] = [np.sum(replaced[:, :, kt]) for kt in range(piv_data['U'].shape[2])]
    
    piv_data['Status'] = status.astype(np.uint16)

    return piv_data
