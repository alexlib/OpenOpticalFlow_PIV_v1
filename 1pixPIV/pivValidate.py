# function [pivData] = pivValidate_V081(pivData,pivPar)
# % pivValidate - validates displacement vectors in PIV data
# %
# % Usage:
# %     [pivData] = pivValidate(pivData,pivPar)
# %
# % Inputs:
# %     pivData ... (struct) structure containing more detailed results. Required field is
# %        X, Y ... position, at which velocity/displacement is calculated
# %        U, V ... displacements in x and y direction
# %        status ... matrix describing status of velocity vectors (for values, see Outputs section)
# %     pivPar ... (struct) parameters defining the evaluation. Following fields are considered:
# %         vlTresh, vlEps ... Define treshold for the median test. To accepted, the difference of actual vector
# %                            from the median (of vectors in the neighborhood) should be at most
# %                            vlTresh *(vlEps + (neighborhood vectors) - (their median)).
# %         vlDist ... to what distance median test is performed (if vlDist = 1, kernel has  size 3x3; for
# %                    vlDist = 2, kernel is 5x5, and so on).
# %          vlDistSeq ... to what distance (in time) the median test is performed. If vlDistSeq == 1, medina test
# %              will be based on one previous and one subsequent time slices. This parameter is taken in
# %              account only if pivValidate.m is applied on pivData, in which contains data for image sequence.
# %              If vlDist is not specified and if pivData contains results for an image sequence, vlDistSeq = 0
# %              is assumed (that is, median test is based only on current time slice).
# %         vlPasses ... number of passes of the median test
# %
# % Outputs:
# %     pivData  ... (struct) structure containing more detailed results. If pivData was non-empty at the input, its
# %              fields are preserved. Following fields are added or updated:
# %        U, V ... x and y components of the velocity/displacement vector (spurious vectors replaced by NaNs)
# %        Status ... matrix with statuis of velocity vectors (uint8). Bits have this coding:
# %            1 (bit  1) ... masked (set by pivInterrogate)
# %            2 (bit  2) ... cross-correlation failed (set by pivCrossCorr)
# %            4 (bit  3) ... peak detection failed (set by pivCrossCorr)
# %            8 (bit  4) ... indicated as spurious by median test based on image pair (set by pivValidate)
# %           16 (bit  5) ... interpolated (set by pivReplaced)
# %           32 (bit  6) ... smoothed (set by pivSmooth)
# %           64 (bit  7) ... indicated as spurious by median test based on image sequence (set by pivValidate)
# %          128 (bit  8) ... interpolated within image sequence (set by pivReplaced)
# %          256 (bit  9) ... smoothed within an image sequence (set by pivSmooth)
# %        spuriousN ... number of spurious vectors
# %        spuriousX, spuriousY ... positions, at which the velocity is spurious
# %        spuriousU, spuriousV ... components of the velocity/displacement vectors, which were indicated as
# %                             spurious
# %
# %
# % Outputs:
# %    pivData  ... (struct) structure containing more detailed results. If some fiels were present in pivData
# %           at the input, they are repeated. Followinf fields are added:
# %        imFilename1, imFilename2 ... path and filename of image files (stored only if im1 and im2 are
# %              filenames)
# %        imMaskFilename1, imMaskFilename2 ... path and filename of masking files (stored only if imMask1 and
# %              imMask2 are filenames)
# %        N ... number of interrogation area (= of velocity vectors)
# %        X, Y ... matrices with centers of interrogation areas (positions of velocity vectors)
# %        U, V ... components of velocity vectors
# %        Status ... matrix with statuis of velocity vectors (uint8). Bits have this coding:
# %            1 (bit 1) ... masked (set by pivInterrogate)
# %            2 (bit 2) ... cross-correlation failed (set by pivCrossCorr)
# %            4 (bit 3) ... peak detection failed (set by pivCrossCorr)
# %            8 (bit 4) ... indicated as spurious by median test based on image pair (set by pivValidate)
# %           16 (bit 5) ... interpolated (set by pivReplaced)
# %           32 (bit 6) ... smoothed (set by pivSmooth)
# %           64 (bit 7) ... indicated as spurious by median test based on image sequence (set by pivValidate);
# %              this flag cannot be set when working with a single image pair
# %          128 (bit 8) ... interpolated within image sequence (set by pivReplaced); this flag cannot be set
# %              when working with a single image pair
# %          256 (bit 9) ... smoothed within an image sequence (set by pivSmooth); this flag cannot be set when
# %              working with a single image pair
# %           (example: if Status for a particulat point is 56 = 32 + 16 + 8, the velocity vector in this point
# %            was indicated as spurious, was replaced by interpolating neighborhood values and was then
# %            adjusted by smoothing.)
# %        iaSizeX, iaSizeY, iaStepX, iaStepY ... copy of dorresponding fields in pivPar input
# %        imSizeX, imSizeY ... image size in pixels
# %        imFilename1, imFilename2 ... path and filename of image files (stored only if im1 and im2 are
# %            filenames)
# %        imMaskFilename1, imMaskFilename2 ... path and filename of masking files (stored only if imMask1 and
# %            imMask2 are filenames)
# %        imNo1, imNo2, imPairNo ... image number and number of image pair (stored only if im1 and im2 are
# %            string with filenames of images). For example, if im1 and im2 are 'Img000005.bmp' and
# %            'Img000006.bmp', value will be imNo1 = 5, imNo2 = 6, and imPairNo = 5.5.
# %        ccPeak ... table with values of cross-correlation peak
# %        ccPeakSecondary ... table with values of secondary cross-correlation peak (maximum of
# %                            crosscorrelation, if 5x5 neighborhood of primary peak is removed)
# %        ccFailedN ... number of vectors for which cross-correlation failed
# %            at distance larger than ccMaxDisplacement*(iaSizeX,iaSizeY) )
# %        ccSubpxFailedN ... number of vectors for which subpixel interpolation failed
# %        spuriousN ... number of spurious vectors (status 1)
# %        spuriousX, spuriousY ... positions, at which the velocity is spurious
# %        spuriousU, spuriousV ... components of the velocity/displacement vectors, which were indicated as
# %                             spurious
# %        replacedN ... number of interpolated vectors (status 2)
# %        replacedX,replacedY ... positions, at which velocity/displacement vectors were replaced
# %        replacedU,replacedV ... components of the velocity/displacement vectors, which were replaced
# %        validN ... number of original and vectors
# %        validX,validY ... positions, at which velocity/displacement vectors is original and valid
# %        validU,validV ... original and valid components of the velocity/displacement vector
# %        infCompTime ... 1D array containing computational time of individual passes (in seconds)
# %    ccFunction ... returns cross-correlation function (in form of an expanded image); see pivCrossCorr
# %
# %%
# % This subroutine is a part of
# %
# % =========================================
# %               PIVsuite
# % =========================================
# %
# % PIVsuite is a set of subroutines intended for processing of data acquired with PIV (particle image
# % velocimetry) within Matlab environment.
# %
# % Written by Jiri Vejrazka, Institute of Chemical Process Fundamentals, Prague, Czech Republic
# %
# % For the use, see files example_XX_xxxxxx.m, which acompany this file. PIVsuite was tested with
# % Matlab 8.2 (R2013b).
# %
# % In the case of a bug, please, contact me: vejrazka (at) icpf (dot) cas (dot) cz
# %
# %
# % Requirements:
# %     Image Processing Toolbox
# %         (required only if pivPar.smMethod is set to 'gaussian')
# %
# %     inpaint_nans.m
# %         subroutine by John D'Errico, available at http://www.mathworks.com/matlabcentral/fileexchange/4551
# %
# %     smoothn.m
# %         subroutine by Damien Garcia, available at
# %         http://www.mathworks.com/matlabcentral/fileexchange/274-smooth
# %
# % Credits:
# %    PIVsuite is a redesigned version of a part of PIVlab software [3], developped by W. Thielicke and
# %    E. J. Stamhuis. Some parts of this code are copied or adapted from it (especially from its
# %    piv_FFTmulti.m subroutine).
# %
# %    PIVsuite uses 3rd party software:
# %        inpaint_nans.m, by J. D'Errico, [2]
# %        smoothn.m, by Damien Garcia, [5]
# %
# % References:
# %   [1] Adrian & Whesterweel, Particle Image Velocimetry, Cambridge University Press, 2011
# %   [2] John D'Errico, inpaint_nans subroutine, http://www.mathworks.com/matlabcentral/fileexchange/4551
# %   [3] W. Thielicke and E. J. Stamhuid, PIVlab 1.31, http://pivlab.blogspot.com
# %   [4] Raffel, Willert, Wereley & Kompenhans, Particle Image Velocimetry: A Practical Guide. 2nd edition,
# %       Springer, 2007
# %   [5] Damien Garcia, smoothn subroutine, http://www.mathworks.com/matlabcentral/fileexchange/274-smooth
# %


# %% Velocity field validation by median test

# % initialize fields
import numpy as np
import time
from .piv_parameters import PIVParameters

def piv_validate(piv_data, piv_par):
    def medianfast(in_array):
        in_array = in_array.flatten()
        in_array = in_array[~np.isnan(in_array)]
        if in_array.size == 0:
            return np.nan
        in_array = np.sort(in_array)
        N = in_array.size
        if N % 2 == 0:
            return (in_array[N//2 - 1] + in_array[N//2]) / 2
        else:
            return in_array[N//2]

    # Check if U is a 3D array
    if len(piv_data['U'].shape) < 3:
        # If U is a 2D array, create 3D arrays for the medians and RMS
        vl_med_u = np.copy(piv_data['U']).reshape(piv_data['U'].shape[0], piv_data['U'].shape[1], 1) * np.nan
        vl_med_v = np.copy(vl_med_u)
        vl_rms_u = np.copy(vl_med_u)
        vl_rms_v = np.copy(vl_med_u)
    else:
        # If U is a 3D array, create 3D arrays for the medians and RMS
        vl_med_u = np.copy(piv_data['U']) * np.nan  # velocity median in the neighborhood of given IA
        vl_med_v = np.copy(vl_med_u)
        vl_rms_u = np.copy(vl_med_u)  # rms (from median) in the neighborhood of given IA
        vl_rms_v = np.copy(vl_med_u)
    X = piv_data['X']
    Y = piv_data['Y']
    U = piv_data['U']
    V = piv_data['V']
    status = piv_data['Status']
    # Convert piv_par to PIVParameters if it's not already
    piv_par = PIVParameters.from_tuple_or_dict(piv_par)

    # Extract parameters from piv_par
    dist_t = piv_par.vlDistTSeq
    dist_xy = piv_par.get_parameter('vlDist')
    passes = piv_par.get_parameter('vlPasses')
    tresh = piv_par.vlTresh
    epsi = piv_par.vlEps
    min_cc = piv_par.vlMinCC
    dist_xy_seq = piv_par.vlDistSeq
    passes_seq = piv_par.vlPassesSeq
    tresh_seq = piv_par.vlTreshSeq
    epsi_seq = piv_par.vlEpsSeq
    exp_name = getattr(piv_par, 'expName', '???')
    an_lock_file = getattr(piv_par, 'anLockFile', '')
    # Check if U is a 3D array
    if len(U.shape) < 3:
        # If U is a 2D array, reshape it to a 3D array with a single time slice
        U = U.reshape(U.shape[0], U.shape[1], 1)
        V = V.reshape(V.shape[0], V.shape[1], 1)
        status = status.reshape(status.shape[0], status.shape[1], 1)
        dist_t = 0
    elif U.shape[2] == 1:
        dist_t = 0

    # choose parameters, depending if Pair or Sequence is validated
    if U.shape[2] == 1:
        # Already set above, no need to override
        statusbit = 4
    else:
        # Use sequence parameters instead
        dist_xy = dist_xy_seq
        passes = passes_seq
        tresh = tresh_seq
        epsi = epsi_seq
        statusbit = 7

    # Validation based on ccPeak: Anything with ccPeak < vlMinCC mark as invalid.
    if min_cc > 0:
        # Check if ccPeak is a 3D array
        if len(piv_data['ccPeak'].shape) < 3:
            # If ccPeak is a 2D array, reshape it to a 3D array with a single time slice
            aux_low_cc = (piv_data['ccPeak'] < min_cc * medianfast(piv_data['ccPeak'])).reshape(piv_data['ccPeak'].shape[0], piv_data['ccPeak'].shape[1], 1)
        else:
            aux_low_cc = piv_data['ccPeak'] < min_cc * medianfast(piv_data['ccPeak'])

        for kt in range(status.shape[2]):
            for kx in range(status.shape[1]):
                for ky in range(status.shape[0]):
                    if aux_low_cc[ky, kx, 0] if aux_low_cc.shape[2] == 1 else aux_low_cc[ky, kx, kt]:
                        status[ky, kx, kt] = np.bitwise_or(status[ky, kx, kt], statusbit)

    for kpass in range(passes):  # proceed in two passes
        # replace anything invalid by NaN. Invalid vectors are those with any flag in status, except flags
        # "smoothed" or "smoothed in a sequence".
        aux_status = np.copy(status)
        # Convert to int32 to avoid overflow
        aux_status = aux_status.astype(np.int32)
        aux_status = np.bitwise_and(aux_status, ~32)  # clear "smoothed" flag
        aux_status = np.bitwise_and(aux_status, ~256)  # clear "smoothed in sequence" flag
        U[aux_status != 0] = np.nan  # replace anything masked, wrong, interpolated or spurious by NaN
        V[aux_status != 0] = np.nan
        # pad U and V with NaNs to allow validation at borders
        aux_u = np.pad(U, ((dist_xy, dist_xy), (dist_xy, dist_xy), (dist_t, dist_t)), constant_values=np.nan)
        aux_v = np.pad(V, ((dist_xy, dist_xy), (dist_xy, dist_xy), (dist_t, dist_t)), constant_values=np.nan)
        tpass = time.time()
        # validate inner cells
        for kt in range(U.shape[2]):
            if (kt - 1) / 5 == round((kt - 1) / 5) and U.shape[2] > 1:
                if kt > 1:
                    print(f' Average time {time.time() - tpass:.2f} s per time slice.')
                print(f'Validation of vectors in a sequence ({exp_name}): pass {kpass + 1} of {passes}, time slice {kt + 1} of {U.shape[2]}...')
                if an_lock_file and len(an_lock_file) > 0:
                    with open(an_lock_file, 'w') as flock:
                        flock.write(f'{time.ctime()}\nValidating sequence...')
                tpass = time.time()
            for kx in range(U.shape[1]):
                for ky in range(U.shape[0]):
                    # compute the medians and deviations from median
                    aux_neigh_u = aux_u[ky:ky + 2 * dist_xy + 1, kx:kx + 2 * dist_xy + 1, kt:kt + 2 * dist_t + 1]
                    aux_neigh_v = aux_v[ky:ky + 2 * dist_xy + 1, kx:kx + 2 * dist_xy + 1, kt:kt + 2 * dist_t + 1]
                    vl_med_u[ky, kx, kt] = medianfast(aux_neigh_u)
                    vl_med_v[ky, kx, kt] = medianfast(aux_neigh_v)
                    aux_neigh_u[dist_xy, dist_xy, dist_t] = np.nan  # remove examined vector from the rms calculation
                    aux_neigh_v[dist_xy, dist_xy, dist_t] = np.nan
                    vl_rms_u[ky, kx, kt] = medianfast(np.abs(aux_neigh_u - vl_med_u[ky, kx, kt]))  # rms of values from the median
                    vl_rms_v[ky, kx, kt] = medianfast(np.abs(aux_neigh_v - vl_med_v[ky, kx, kt]))
                    if status[ky, kx, kt] == 0 and np.abs(U[ky, kx, kt] - vl_med_u[ky, kx, kt]) > (tresh * vl_rms_u[ky, kx, kt] + epsi):
                        status[ky, kx, kt] = np.bitwise_or(status[ky, kx, kt], statusbit)
                    if status[ky, kx, kt] == 0 and np.abs(V[ky, kx, kt] - vl_med_v[ky, kx, kt]) > (tresh * vl_rms_v[ky, kx, kt] + epsi):
                        status[ky, kx, kt] = np.bitwise_or(status[ky, kx, kt], statusbit)
            if kt > 1:
                print(f' Validation pass finished in {time.time() - tpass:.2f} s.')

    # replace spurious vectors with NaN's
    spurious = np.bitwise_or(np.bitwise_and(status, 4) != 0, np.bitwise_and(status, 128) != 0)
    U[spurious] = np.nan
    V[spurious] = np.nan

    # output detailed piv_data
    if U.shape[2] == 1:
        # If U is a 3D array with a single time slice, we need to extract the 2D slice
        aux_spur = np.bitwise_or(np.bitwise_and(status[:, :, 0], 4) != 0, np.bitwise_and(status[:, :, 0], 128) != 0)
        vl_n_spur = np.sum(aux_spur)
        spurious_x = X[aux_spur]
        spurious_y = Y[aux_spur]
        # Check if piv_data['U'] is a 3D array
        if len(piv_data['U'].shape) < 3:
            spurious_u = piv_data['U'][aux_spur]
            spurious_v = piv_data['V'][aux_spur]
        else:
            spurious_u = piv_data['U'][:, :, 0][aux_spur]
            spurious_v = piv_data['V'][:, :, 0][aux_spur]
    else:
        vl_n_spur = np.full(U.shape[2], np.nan)
        for kt in range(U.shape[2]):
            aux_spur = np.bitwise_or(np.bitwise_and(status[:, :, kt], 4) != 0, np.bitwise_and(status[:, :, kt], 128) != 0)
            vl_n_spur[kt] = np.sum(aux_spur)

    # output variables
    piv_data['U'] = U
    piv_data['V'] = V
    piv_data['Status'] = status.astype(np.uint16)
    piv_data['spuriousN'] = vl_n_spur
    if U.shape[2] == 1:
        piv_data['spuriousX'] = spurious_x
        piv_data['spuriousY'] = spurious_y
        piv_data['spuriousU'] = spurious_u
        piv_data['spuriousV'] = spurious_v

    return piv_data


def stdfast(in_array):
    in_array = in_array.flatten()
    notnan = ~np.isnan(in_array)
    n = np.sum(notnan)
    in_array[~notnan] = 0
    avg = np.sum(in_array) / n
    out = np.sqrt(np.sum(((in_array - avg) * notnan) ** 2) / n)  # there should be -1 in the denominator for true std
    return out


import numpy as np

def medianfast(in_array):
    in_array = in_array.flatten()
    in_array = in_array[~np.isnan(in_array)]
    if in_array.size == 0:
        return np.nan
    in_array = np.sort(in_array)
    N = in_array.size
    if N % 2 == 0:
        return (in_array[N//2 - 1] + in_array[N//2]) / 2
    else:
        return in_array[N//2]
