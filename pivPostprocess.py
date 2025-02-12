import numpy as np
import time

def piv_postprocess(action, *args):
    """
    piv_postprocess - postprocess PIV data

    Usage:
        piv_data_out = piv_postprocess('vorticity', piv_data)
        piv_data_seq = piv_postprocess('vorticity', piv_data_seq)
        Computes the vorticity field. The vorticity computation follows the eqs. (9.8) and (9.11) in Adrian
        & Westerweel, p. 432

    Inputs:
        action ... defines how the data will be manipulated. Recognized actions (case-insensitive) are
            'vorticity'.
            See "Usage" for details.
        piv_data ... structure containing results of pivAnalyzeImagePair.m. In this structure, velocity fields .U
            and .V have ny x nx elements (where ny and nx is the number of interrogation areas). 
        piv_data_seq ... structure containing results of pivAnalyzeImageSequence.m. In this structure, velocity
            fields .U and .V have ny x nx x nt elements (where ny and nx is the number of interrogation areas,
            nt is the number of image pairs). .ccPeak and .ccPeakSecondary have the same size. Fields
            .spuriousU, .spuriousV, .spuriousX, .spuriousY are missing.

    Outputs:
        piv_data, piv_data_seq ... see inputs for their meaning.
    """

    action = action.lower()

    if action == 'vorticity':
        print('Calculating the vorticity... ')
        start_time = time.time()
        piv_data_c = args[0]
        vort = np.full_like(piv_data_c['U'], np.nan)
        for ki in range(piv_data_c['U'].shape[2]):
            # following eq. (9.8) and (9.11) in Adrian & Westerweel, p. 432
            # compute filtered velocity fields
            Ufilt = np.full((piv_data_c['U'].shape[0] + 2, piv_data_c['U'].shape[1] + 2), np.nan)
            Ufilt[1:-1, 1:-1] = 0.5 * piv_data_c['U'][:, :, ki]
            Ufilt[1:-1, :-2] += 0.25 * piv_data_c['U'][:, :, ki]
            Ufilt[1:-1, 2:] += 0.25 * piv_data_c['U'][:, :, ki]
            Vfilt = np.full((piv_data_c['U'].shape[0] + 2, piv_data_c['U'].shape[1] + 2), np.nan)
            Vfilt[1:-1, 1:-1] = 0.5 * piv_data_c['V'][:, :, ki]
            Vfilt[:-2, 1:-1] += 0.25 * piv_data_c['V'][:, :, ki]
            Vfilt[2:, 1:-1] += 0.25 * piv_data_c['V'][:, :, ki]
            # get grid spacing
            dX = piv_data_c['X'][0, 1] - piv_data_c['X'][0, 0]
            dY = piv_data_c['Y'][1, 0] - piv_data_c['Y'][0, 0]
            # compute vorticity
            vort[:, :, ki] = (Vfilt[1:-1, 2:] - Vfilt[1:-1, :-2]) / (2 * dX) - \
                             (Ufilt[2:, 1:-1] - Ufilt[:-2, 1:-1]) / (2 * dY)
        piv_data_c['vorticity'] = vort.astype(np.float32)
        print(f'finished in {time.time() - start_time:.2f} s.')
        piv_data_out = piv_data_c

    else:
        print(f'Error: action {action} is not recognized by piv_postprocess.')

    return piv_data_out
