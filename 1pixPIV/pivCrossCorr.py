# function [pivData,ccPeakIm] = pivCrossCorr(exIm1,exIm2,pivData,pivPar)
# % pivCrossCorr - cross-correlates two images to find object displacement between them
# %
# % Usage:
# % [pivData,ccPeakIm] = pivCrossCorr(exIm1,exIm2,pivData,pivPar)
# %
# % Inputs:
# %    Exim1,Exim2 ... pair of expanded images (use pivInterrogate for their generation)
# %    pivData ... (struct) structure containing more detailed results. Following fields are required (use
# %             pivInterrogate for generating them):
# %        Status ... matrix describing status of velocity vectors (for values, see Outputs section)
# %        iaX, iaY ... matrices with centers of interrogation areas
# %        iaU0, iaV0 ... mean shift of IA's
# %    pivPar ... (struct) parameters defining the evaluation. Following fields are considered:
# %       ccRemoveIAMean ... if =0, do not remove IA's mean before cross-correlation; if =1, remove the mean; if
# %                          in between, remove the mean partially
# %       ccMaxDisplacement ... maximum allowed displacement to accept cross-correlation peak. This parameter is 
# %                             a multiplier; e.g. setting ccMaxDisplacement = 0.6 means that the cross-
# %                             correlation peak must be located within [-0.6*iaSizeX...0.6*iaSizeX,
# %                             -0.6*iaSizeY...0.6*iaSizeY] from the zero displacement. Note: IA offset is not 
# %                             included in the displacement, hence real displacement can be larger if 
# %                             ccIAmethod is any other than 'basic'.
# %       ccWindow ... windowing function p. 389 in ref [1]). Possible values are 'uniform', 'gauss',
# %             'parzen', 'hanning', 'welch'
# %       ccCorrectWindowBias ... Set whether cc is corrected for bias due to IA windowing.
# %            - default value: true ( will correct peak )
# %       ccMethod ... set methods for finding cross-correlation of interrogation areas. Possible values are
# %              'fft' (use Fast Fourrier Transform' and 'dcn' (use discrete convolution). Option 'fft' is more
# %              suitable for initial guess of velocity. Option 'dcn' is suitable for final iterations of the
# %              velocity field, if the displacement corrections are already small.
# %       ccMaxDCNdist ... Defines maximum displacement, for which the correlation is computed by DCN method 
# %              (apllies only for ccMethod = 'dcn'). 
# % Outputs:
# %    pivData  ... (struct) structure containing more detailed results. Following fields are added or updated:
# %        X, Y, U, V ... contains velocity field
# %        Status ... matrix with statuis of velocity vectors (uint8). Bits have this coding:
# %            1 ... masked (set by pivInterrogate)
# %            2 ... cross-correlation failed (set by pivCrossCorr)
# %            4 ... peak detection failed (set by pivCrossCorr)
# %            8 ... indicated as spurious by median test (set by pivValidate)
# %           16 ... interpolated (set by pivReplaced)
# %           32 ... smoothed (set by pivSmooth)
# %        ccPeak ... table with values of cross-correlation peak
# %        ccPeakSecondary ... table with values of secondary cross-correlation peak (maximum of
# %            crosscorrelation, if 5x5 neighborhood of primary peak is removed)
# %        ccStd1, ccStd2 ... tables with standard deviation of pixel values in interrogation area, for the
# %            first and second image in the image pair
# %        ccMean1, ccMean2 ... tables with mean of pixel values in interrogation area, for the
# %            first and second image in the image pair
# %        ccFailedN ... number of vectors for which cross-correlation failed
# %            at distance larger than ccMaxDisplacement*(iaSizeX,iaSizeY) )
# %        ccSubpxFailedN ... number of vectors for which subpixel interpolation failed
# %      - Note: fields iaU0 and iaV0 are removed from pivData
# %    ccPeakIm ... expanded image containing cross-correlation functions for all IAs (normalized by .ccPeak)
# % 
# % Important local variables:
# %    failFlag ... contains value of status elements of the vector being processed
# %    Upx, Vpx ... rough position of cross-correlation peak (before subpixel interpolation, in integer 
# %                     number of pixels)
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
# %%


# %% 0. Initialization

import numpy as np

U = pivData['U']
V = pivData['V']
status = pivData['Status']
ccPeak = np.copy(U)                  # will contain peak levels
ccPeakSecondary = np.copy(U)         # will contain level of secondary peaks
iaSizeX = pivPar['iaSizeX']
iaSizeY = pivPar['iaSizeY']
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

if 'ccWindow' not in pivPar or pivPar['ccWindow'].lower() == 'uniform':
    W = np.ones((iaSizeY, iaSizeX))
    F = (1 - np.abs(KsiX)) * (1 - np.abs(KsiY))
elif pivPar['ccWindow'].lower() == 'parzen':
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
elif pivPar['ccWindow'].lower() == 'hanning':
    W = (0.5 + 0.5 * np.cos(2 * np.pi * EtaX)) * (0.5 + 0.5 * np.cos(2 * np.pi * EtaY))
    F = (2/3 * (1 - np.abs(KsiX)) * (1 + 0.5 * np.cos(2 * np.pi * KsiX)) + 0.5 / np.pi * np.sin(2 * np.pi * np.abs(KsiX))) * \
        (2/3 * (1 - np.abs(KsiY)) * (1 + 0.5 * np.cos(2 * np.pi * KsiY)) + 0.5 / np.pi * np.sin(2 * np.pi * np.abs(KsiY)))
elif pivPar['ccWindow'].lower() == 'welch':
    W = (1 - (2 * EtaX)**2) * (1 - (2 * EtaY)**2)
    F = (1 - 5 * KsiX**2 + 5 * np.abs(KsiX)**3 - np.abs(KsiX)**5) * \
        (1 - 5 * KsiY**2 + 5 * np.abs(KsiY)**3 - np.abs(KsiY)**5)
elif pivPar['ccWindow'].lower() == 'gauss':
    W = np.exp(-8 * EtaX**2) * np.exp(-8 * EtaY**2)
    F = np.exp(-4 * KsiX**2) * np.exp(-4 * KsiY**2)
elif pivPar['ccWindow'].lower() == 'gauss1':
    W = np.exp(-8 * (EtaX**2 + EtaY**2)) - np.exp(-2)
    W[W < 0] = 0
    W /= np.max(W)
    F = np.nan
elif pivPar['ccWindow'].lower() == 'gauss2':
    W = np.exp(-16 * (EtaX**2 + EtaY**2)) - np.exp(-4)
    W[W < 0] = 0
    W /= np.max(W)
    F = np.nan
elif pivPar['ccWindow'].lower() == 'gauss0.5':
    W = np.exp(-4 * (EtaX**2 + EtaY**2)) - np.exp(-1)
    W[W < 0] = 0
    W /= np.max(W)
    F = np.nan
elif pivPar['ccWindow'].lower() == 'nogueira':
    W = 9 * (1 - 4 * np.abs(EtaX) + 4 * EtaX**2) * (1 - 4 * np.abs(EtaY) + 4 * EtaY**2)
    F = np.nan
elif pivPar['ccWindow'].lower() == 'hanning2':
    W = (0.5 + 0.5 * np.cos(2 * np.pi * EtaX))**2 * (0.5 + 0.5 * np.cos(2 * np.pi * EtaY))**2
    F = np.nan
elif pivPar['ccWindow'].lower() == 'hanning4':
    W = (0.5 + 0.5 * np.cos(2 * np.pi * EtaX))**4 * (0.5 + 0.5 * np.cos(2 * np.pi * EtaY))**4
    F = np.nan

# Limit F to not be too small
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
            imIA1 -= pivPar['ccRemoveIAMean'] * auxMean1
            imIA2 -= pivPar['ccRemoveIAMean'] * auxMean2
            imIA1 *= W
            imIA2 *= W
            auxStd1 = stdfast(imIA1)
            auxStd2 = stdfast(imIA2)
            if pivPar['ccMethod'].lower() == 'fft':
                cc = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(imIA1)) * np.fft.fft2(imIA2)))) / (auxStd1 * auxStd2) / (iaSizeX * iaSizeY)
                auxPeak = np.max(cc)
                Upx = np.argmax(np.max(cc, axis=0))
                Vpx = np.argmax(cc[:, Upx])
            elif pivPar['ccMethod'].lower() == 'dcn':
                cc = dcn(imIA1, imIA2, pivPar['ccMaxDCNdist']) / (auxStd1 * auxStd2) / (iaSizeX * iaSizeY)
                auxPeak = np.max(cc)
                Upx = np.argmax(np.max(cc, axis=0))
                Vpx = np.argmax(cc[:, Upx])
                if (Upx != iaSizeX / 2 + ccPxShiftX) or (Vpx != iaSizeY / 2 + ccPxShiftY):
                    cc = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(np.fft.fft2(imIA1)) * np.fft.fft2(imIA2)))) / (auxStd1 * auxStd2) / (iaSizeX * iaSizeY)
                    auxPeak = np.max(cc)
                    Upx = np.argmax(np.max(cc, axis=0))
                    Vpx = np.argmax(cc[:, Upx])
            if (abs(Upx - iaSizeX / 2 - ccPxShiftX) > pivPar['ccMaxDisplacement'] * iaSizeX) or \
               (abs(Vpx - iaSizeY / 2 - ccPxShiftY) > pivPar['ccMaxDisplacement'] * iaSizeY):
                failFlag = np.bitwise_or(failFlag, 2)
            if pivPar['ccCorrectWindowBias'] and not np.isnan(F).any():
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


def stdfast(in_array):
    """
    Computes root-mean-square (reprogrammed, because std in Matlab is somewhat slow due to some additional tests).
    """
    in_array = in_array.flatten()
    notnan = ~np.isnan(in_array)
    n = np.sum(notnan)
    in_array[~notnan] = 0
    avg = np.sum(in_array) / n
    out = np.sqrt(np.sum(((in_array - avg) * notnan) ** 2) / n)  # there should be -1 in the denominator for true std
    return out


import numpy as np

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
