# Cross-correlation module for PIV analysis
# Python implementation of the MATLAB PIVsuite by Jiri Vejrazka

import numpy as np
from .piv_parameters import PIVParameters

def piv_cross_corr(exIm1, exIm2, pivData, pivPar):
    """
    Cross-correlates two images to find object displacement between them.

    Parameters:
        exIm1, exIm2: Pair of expanded images (use piv_interrogate for their generation)
        pivData: Dictionary containing detailed results
        pivPar: Parameters defining the evaluation

    Returns:
        pivData: Updated dictionary with cross-correlation results
        ccPeakIm: Expanded image containing cross-correlation functions for all IAs (normalized by ccPeak)
    """
    U = pivData['U']
    V = pivData['V']
    status = pivData['Status']
    ccPeak = np.copy(U)                  # will contain peak levels
    ccPeakSecondary = np.copy(U)         # will contain level of secondary peaks

    # Convert pivPar to PIVParameters if it's not already
    pivPar = PIVParameters.from_tuple_or_dict(pivPar)

    # Extract parameters from pivPar
    iaSizeX = pivPar.get_parameter('iaSizeX')
    iaSizeY = pivPar.get_parameter('iaSizeY')
    ccRemoveIAMean = pivPar.ccRemoveIAMean
    ccMaxDisplacement = pivPar.ccMaxDisplacement
    ccWindow = pivPar.ccWindow
    ccCorrectWindowBias = pivPar.ccCorrectWindowBias
    ccMethod = pivPar.get_parameter('ccMethod')
    ccMaxDCNdist = pivPar.ccMaxDCNdist

    iaNX = pivData['X'].shape[1]
    iaNY = pivData['X'].shape[0]
    ccStd1 = np.full_like(pivData['U'], np.nan)
    ccStd2 = np.full_like(pivData['U'], np.nan)
    ccMean1 = np.full_like(pivData['U'], np.nan)
    ccMean2 = np.full_like(pivData['U'], np.nan)

    # initialize "expanded image" for storing cross-correlations
    ccPeakIm = np.full_like(exIm1, np.nan)   # same size as expanded images

    # peak position is shifted by 1 or 0.5 px, depending on IA size
    ccPxShiftX = 1 if iaSizeX % 2 == 0 else 0.5
    ccPxShiftY = 1 if iaSizeY % 2 == 0 else 0.5

    # Create windowing function W and loss-of-correlation function F
    auxX = np.ones((iaSizeY, 1)) * np.arange(-(iaSizeX-1)/2, (iaSizeX-1)/2 + 1)
    auxY = np.arange(-(iaSizeY-1)/2, (iaSizeY-1)/2 + 1).reshape(-1, 1) * np.ones((1, iaSizeX))
    EtaX = auxX / iaSizeX
    EtaY = auxY / iaSizeY
    KsiX = 2 * EtaX
    KsiY = 2 * EtaY

    # Select window function
    window_type = ccWindow.lower()
    if window_type == 'uniform':
        W = np.ones((iaSizeY, iaSizeX))
        F = (1 - np.abs(KsiX)) * (1 - np.abs(KsiY))
    elif window_type == 'parzen':
        W = (1 - 2 * np.abs(EtaX)) * (1 - 2 * np.abs(EtaY))
        auxFx = np.full_like(auxX, np.nan)
        auxFy = np.full_like(auxX, np.nan)
        auxOK = np.abs(KsiX) <= 1/2
        auxFx[auxOK] = 1 - 6 * KsiX[auxOK]**2 + 6 * np.abs(KsiX[auxOK])**3
        auxFx[~auxOK] = 2 - 6 * np.abs(KsiX[~auxOK]) + 6 * KsiX[~auxOK]**2 - 2 * np.abs(KsiX[~auxOK])**3
        auxOK = np.abs(KsiY) <= 1/2
        auxFy[auxOK] = 1 - 6 * KsiY[auxOK]**2 + 6 * np.abs(KsiY[auxOK])**3
        auxFy[~auxOK] = 2 - 6 * np.abs(KsiY[~auxOK]) + 6 * KsiY[~auxOK]**2 - 2 * np.abs(KsiY[~auxOK])**3
        F = auxFx * auxFy
    elif window_type == 'hanning':
        W = (0.5 + 0.5 * np.cos(2 * np.pi * EtaX)) * (0.5 + 0.5 * np.cos(2 * np.pi * EtaY))
        F = (2/3 * (1 - np.abs(KsiX)) * (1 + 0.5 * np.cos(2 * np.pi * KsiX)) + 0.5 / np.pi * np.sin(2 * np.pi * np.abs(KsiX))) * \
            (2/3 * (1 - np.abs(KsiY)) * (1 + 0.5 * np.cos(2 * np.pi * KsiY)) + 0.5 / np.pi * np.sin(2 * np.pi * np.abs(KsiY)))
    elif window_type == 'welch':
        W = (1 - (2 * EtaX)**2) * (1 - (2 * EtaY)**2)
        F = (1 - 5 * KsiX**2 + 5 * np.abs(KsiX)**3 - np.abs(KsiX)**5) * \
            (1 - 5 * KsiY**2 + 5 * np.abs(KsiY)**3 - np.abs(KsiY)**5)
    elif window_type == 'gauss':
        W = np.exp(-8 * EtaX**2) * np.exp(-8 * EtaY**2)
        F = np.exp(-4 * KsiX**2) * np.exp(-4 * KsiY**2)
    elif window_type == 'gauss1':
        W = np.exp(-8 * (EtaX**2 + EtaY**2)) - np.exp(-2)
        W[W < 0] = 0
        W /= np.max(W)
        F = np.nan
    elif window_type == 'gauss2':
        W = np.exp(-16 * (EtaX**2 + EtaY**2)) - np.exp(-4)
        W[W < 0] = 0
        W /= np.max(W)
        F = np.nan
    elif window_type == 'gauss0.5':
        W = np.exp(-4 * (EtaX**2 + EtaY**2)) - np.exp(-1)
        W[W < 0] = 0
        W /= np.max(W)
        F = np.nan
    elif window_type == 'nogueira':
        W = 9 * (1 - 4 * np.abs(EtaX) + 4 * EtaX**2) * (1 - 4 * np.abs(EtaY) + 4 * EtaY**2)
        F = np.nan
    elif window_type == 'hanning2':
        W = (0.5 + 0.5 * np.cos(2 * np.pi * EtaX))**2 * (0.5 + 0.5 * np.cos(2 * np.pi * EtaY))**2
        F = np.nan
    elif window_type == 'hanning4':
        W = (0.5 + 0.5 * np.cos(2 * np.pi * EtaX))**4 * (0.5 + 0.5 * np.cos(2 * np.pi * EtaY))**4
        F = np.nan

    # Limit F to not be too small
    if isinstance(F, np.ndarray):
        F[F < 0.5] = 0.5

    # Cross-correlate expanded images and do subpixel interpolation
    for kx in range(iaNX):
        for ky in range(iaNY):
            failFlag = status[ky, kx]
            if failFlag == 0:
                imIA1 = exIm1[(ky * iaSizeY):((ky + 1) * iaSizeY), (kx * iaSizeX):((kx + 1) * iaSizeX)]
                imIA2 = exIm2[(ky * iaSizeY):((ky + 1) * iaSizeY), (kx * iaSizeX):((kx + 1) * iaSizeX)]
                auxMean1 = np.mean(imIA1)
                auxMean2 = np.mean(imIA2)
                imIA1 -= ccRemoveIAMean * auxMean1
                imIA2 -= ccRemoveIAMean * auxMean2
                imIA1 *= W
                imIA2 *= W
                auxStd1 = stdfast(imIA1)
                auxStd2 = stdfast(imIA2)
                # Get the cross-correlation method
                cc_method = ccMethod.lower() if isinstance(ccMethod, str) else 'fft'
                if cc_method == 'fft':
                    cc = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(imIA1)) * np.fft.fft2(imIA2)))) / (auxStd1 * auxStd2) / (iaSizeX * iaSizeY)
                    auxPeak = np.max(cc)
                    Upx = np.argmax(np.max(cc, axis=0))
                    Vpx = np.argmax(cc[:, Upx])
                elif cc_method == 'dcn':
                    cc = dcn(imIA1, imIA2, ccMaxDCNdist) / (auxStd1 * auxStd2) / (iaSizeX * iaSizeY)
                    auxPeak = np.max(cc)
                    Upx = np.argmax(np.max(cc, axis=0))
                    Vpx = np.argmax(cc[:, Upx])
                    if (Upx != iaSizeX / 2 + ccPxShiftX) or (Vpx != iaSizeY / 2 + ccPxShiftY):
                        cc = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(imIA1)) * np.fft.fft2(imIA2)))) / (auxStd1 * auxStd2) / (iaSizeX * iaSizeY)
                        auxPeak = np.max(cc)
                        Upx = np.argmax(np.max(cc, axis=0))
                        Vpx = np.argmax(cc[:, Upx])
                if (abs(Upx - iaSizeX / 2 - ccPxShiftX) > ccMaxDisplacement * iaSizeX) or \
                (abs(Vpx - iaSizeY / 2 - ccPxShiftY) > ccMaxDisplacement * iaSizeY):
                    failFlag = np.bitwise_or(failFlag, 2)
                if ccCorrectWindowBias and not np.isnan(F).any():
                    ccCor = cc / F
                else:
                    ccCor = cc
                try:
                    dU = (np.log(ccCor[Vpx, Upx - 1]) - np.log(ccCor[Vpx, Upx + 1])) / \
                        (np.log(ccCor[Vpx, Upx - 1]) + np.log(ccCor[Vpx, Upx + 1]) - 2 * np.log(ccCor[Vpx, Upx])) / 2
                    dV = (np.log(ccCor[Vpx - 1, Upx]) - np.log(ccCor[Vpx + 1, Upx])) / \
                        (np.log(ccCor[Vpx - 1, Upx]) + np.log(ccCor[Vpx + 1, Upx]) - 2 * np.log(ccCor[Vpx, Upx])) / 2
                except:
                    failFlag = np.bitwise_or(failFlag, 3)
                    dU = np.nan
                    dV = np.nan
                if not np.isreal(dU) or not np.isreal(dV):
                    failFlag = np.bitwise_or(failFlag, 3)
            else:
                cc = np.full((iaSizeY, iaSizeX), np.nan)
                auxPeak = np.nan
                auxStd1 = np.nan
                auxStd2 = np.nan
                auxMean1 = np.nan
                auxMean2 = np.nan
                Upx = iaSizeX / 2
                Vpx = iaSizeY / 2
            if failFlag == 0:
                U[ky, kx] = pivData['iaU0'][ky, kx] + Upx + dU - iaSizeX / 2 - ccPxShiftX
                V[ky, kx] = pivData['iaV0'][ky, kx] + Vpx + dV - iaSizeY / 2 - ccPxShiftY
            else:
                U[ky, kx] = np.nan
                V[ky, kx] = np.nan
            status[ky, kx] = failFlag
            ccPeakIm[(ky * iaSizeY):((ky + 1) * iaSizeY), (kx * iaSizeX):((kx + 1) * iaSizeX)] = cc
            ccPeak[ky, kx] = auxPeak
            ccStd1[ky, kx] = auxStd1
            ccStd2[ky, kx] = auxStd2
            ccMean1[ky, kx] = auxMean1
            ccMean2[ky, kx] = auxMean2
            try:
                cc[Vpx - 2:Vpx + 3, Upx - 2:Upx + 3] = 0
                ccPeakSecondary[ky, kx] = np.max(cc)
            except:
                try:
                    cc[Vpx - 1:Vpx + 2, Upx - 1:Upx + 2] = 0
                    ccPeakSecondary[ky, kx] = np.max(cc)
                except:
                    ccPeakSecondary[ky, kx] = np.nan

    ccFailedI = np.bitwise_and(status, 2).astype(bool)
    ccSubpxFailedI = np.bitwise_and(status, 3).astype(bool)

    pivData['Status'] = status.astype(np.uint16)
    pivData['U'] = U
    pivData['V'] = V
    pivData['ccPeak'] = ccPeak
    pivData['ccPeakSecondary'] = ccPeakSecondary
    pivData['ccStd1'] = ccStd1
    pivData['ccStd2'] = ccStd2
    pivData['ccMean1'] = ccMean1
    pivData['ccMean2'] = ccMean2
    pivData['ccFailedN'] = np.sum(ccFailedI)
    pivData['ccSubpxFailedN'] = np.sum(ccSubpxFailedI)
    pivData['ccW'] = W
    del pivData['iaU0']
    del pivData['iaV0']

    return pivData, ccPeakIm


def stdfast(in_array):
    """
    Computes root-mean-square (reprogrammed, because std in Matlab is somewhat slow due to some additional tests).
    """
    in_array = in_array.flatten()
    notnan = ~np.isnan(in_array)
    n = np.sum(notnan)

    # Handle the case when all values are NaN
    if n == 0:
        return 0.0

    in_array_copy = in_array.copy()  # Create a copy to avoid modifying the original
    in_array_copy[~notnan] = 0
    avg = np.sum(in_array_copy) / n
    out = np.sqrt(np.sum(((in_array_copy - avg) * notnan) ** 2) / n)  # there should be -1 in the denominator for true std
    return out

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
