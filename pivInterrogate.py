# function [exIm1,exIm2,pivData] = pivInterrogate(im1,im2,pivData,pivPar)
# % pivInterrogate - splits images into interrogation areas and create expanded images suitable for pivCrossCorr
# %
# % Usage:
# % [exIm1,exIm2,pivData] = pivInterrogate(im1,im2,pivData,pivPar,Xest,Yest,Uest,Vest)
# %
# % Inputs:
# %    im1,im2 ... image pair (either images, or strings containing paths to image files)
# %    pivData ... (struct) structure containing detailed results. No fields are required as the input, but
# %             the existing fields will be copies to pivData at the output. If exist, folowing fields will be
# %             used:
# %        X, Y ... position, at which U, V velocity/displacement is provided 
# %        U, V ... displacements in x and y direction (will be used for image deformation). If these fields do
# %            not exist, zero velocity is assumed
# %    pivPar ... (struct) parameters defining the evaluation. Following fields are considered:
# %       iaSizeX, iaSizeY ... size of interrogation area [px]
# %       iaStepX, iaStepY ... step between interrogation areas [px]
# %       imMask1, imMask2 ... Masking images for im1 and im2. It should be either empty (no mask), or of the
# %                            same size as im1 and im2. Masked pixels should be 0 in .imMaskX, non-masked 
# %                            pixels should be 1.
# %       iaMethod ... way, how interrogation area are created. Possible values are
# %           'basic' ... interrogatio areas are regularly distribute rectangles 
# %           'offset' ... (not coded) interrogation areas are shifted by the estimated displacement
# %           'deflinear' ... (not coded) deformable interrogation areas with linear deformation
# %           'defspline' ... (not coded) deformable interrogation areas with spline deformation
# %           Note: if Uest and Vest contains only zeros or if they are empty/unspecified, 'basic' method is 
# %                 always invoked regardless .iaMethod setting
# %       iaImageToDeform ... defines, which image should deform (if .iaMethod is 'deflinear' or 'defspline'), or
# %                           in which images IA should be shifted (.iaMethod == 'offset'). Possible values are
# %           'image1', 'image2' ... either im1 or im2 is deformed correspondingly to Uest and Vest
# %           'both' ... deformation both images are deformed by Uest/2 and Vest/2. More CPU time is required.
# %       iaImageInterpolationMethod ... way, how the images are interpolated when deformable IAs are used (for 
# %                                      .iaMethod == 'deflinear' or 'defspline'. Possible values are:
# %           'linear', 'spline' ... interpolation is carried out using interp2 function with option either
# %                                  '*linear' or '*spline'
# %       iaPreprocMethod ... defines image preprocessing method. Possible values are
# %            'none' ... no image preprocessing
# %            'MinMax' ... MinMax filter is applied (see p. 248 in Ref. [1])
# %       iaMinMaxSize ... (applies only if iaPreprocMethod is 'MinMax'). Size of MinMax filter kernel.
# %       iaMinMaxLevel ... (applies only if iaPreprocMethod is 'MinMax'). Contrast level, below which
# %            contrast in not more enhanced.
# % Outputs:
# %    exIm1, exIm2 ... expanded image 1 and 2. Expanded image is an image, in which IAs are side-by-side 
# %          (expanded image has size [iaNX*iaSizeX,iaNY*iaSizeY]); if a pixel appears in n IAs, it will be 
# %          present n times in the expanded image. If iaStepX == iaSizeX and iaStepY == iaSizeY and method is 
# %          'basic' (or Uest == Vest ==0), expanded image is the same as original image (except cropping).
# %    pivData  ... (struct) structure containing more detailed results. If some fiels were present in pivData at the
# %              input, they are repeated. Followinf fields are added:
# %        imFilename1, imFilename2 ... path and filename of image files (stored only if im1 and im2 are 
# %              filenames)
# %        imMaskFilename1, imMaskFilename2 ... path and filename of masking files (stored only if imMask1 and 
# %              imMask2 are filenames)
# %        imArray1, imArray2 ... arrays with read and preprocessed images (these fields are removed at
# %              pivAnalyzeImagePair.m)
# %        imMaskArray1, imMaskArray2 ... arrays containing Boolean variable (they are removed by
# %              pivAnalyzeImagePair.m)
# %        imNo1, imNo2, imPairNo ... image number (completed only if im1 and im2 are filenames with images).
# %              For example, if im1 and im2 are 'Img000005.bmp' and 'Img000006.bmp', value will be imNo1 = 5, 
# %              imNo2 = 6, and imPairNo = 5.5.
# %        N ... number of interrogation area (= of velocity vectors)
# %        X, Y ... matrices with centers of interrogation areas
# %        Status ... matrix with statuis of velocity vectors (uint8). Bits have this coding:
# %            1 ... masked (set by pivInterrogate)
# %            2 ... cross-correlation failed (set by pivCrossCorr)
# %            4 ... peak detection failed (set by pivCrossCorr)
# %            8 ... indicated as spurious by median test (set by pivValidate)
# %           16 ... interpolated (set by pivReplaced)
# %           32 ... smoothed (set by pivSmooth)
# %        iaU0, iaV0 ... mean shift of IAs in the deformed image
# %        iaSizeX, iaSizeY, iaStepX, iaStepY ... copy of dorresponding fields in pivPar input
# %        imSizeX, imSizeY ... image size in pixels
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
# %% Acronyms and meaning of variables used in this subroutine:
# %    IA ... concerns "Interrogation Area"
# %    im ... image
# %    dx ... some index
# %    ex ... expanded (image)
# %    est ... estimate (velocity from previous pass) - will be used to deform image
# %    aux ... auxiliary variable (which is of no use just a few lines below)
# %    cc ... cross-correlation
# %    vl ... validation
# %    sm ... smoothing
# %    Word "velocity" should be understood as "displacement"



# %% 0. Read images, if required. Read mask images. Preprocess images, if required
# % Extract from pivPar some frequently used fields (for shortenning the code);
iaSizeX = pivPar.iaSizeX;
iaSizeY = pivPar.iaSizeY;
iaStepX = pivPar.iaStepX;
iaStepY = pivPar.iaStepY;

# read images if im1 and im2 are filepaths
if 'imArray1' not in pivData:   # read files only if not read in the previous pass
    if isinstance(im1, str):
        imgNo, filename, folder = treat_img_path(im1)
        pivData['imFilename1'] = im1
        pivData['imNo1'] = imgNo
        im1 = imread(im1)
    if isinstance(im2, str):
        imgNo, filename, folder = treat_img_path(im2)
        pivData['imFilename2'] = im2
        pivData['imNo2'] = imgNo
        im2 = imread(im2)
    try:
        pivData['imPairNo'] = (pivData['imNo1'] + pivData['imNo2']) / 2
    except:
        pass
    
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    
    # read image masks if pivPar.imMask1 and pivPar.imMask2 are filepaths
    if isinstance(pivPar['imMask1'], str):
        pivData['imMaskFilename1'] = pivPar['imMask1']
        if pivPar['imMask1']:
            imMask1 = imread(pivPar['imMask1'])
        else:
            imMask1 = None
    else:
        imMask1 = pivPar['imMask1']
    if isinstance(pivPar['imMask2'], str):
        pivData['imMaskFilename2'] = pivPar['imMask2']
        if pivPar['imMask2']:
            imMask2 = imread(pivPar['imMask2'])
        else:
            imMask2 = None
    else:
        imMask2 = pivPar['imMask2']
    pivData['imMaskArray1'] = imMask1
    pivData['imMaskArray2'] = imMask2
    
    # check the consistence of images
    auxDiff = np.abs(np.array(im1.shape) - np.array(im2.shape)) + np.abs(np.array(imMask1.shape) - np.array(imMask2.shape))
    if imMask1 is not None:
        auxDiff += np.abs(np.array(im1.shape) - np.array(imMask1.shape))
    if np.sum(auxDiff) > 0:
        raise ValueError('PIVsuite:InconsistImgs', 'Image 1, Image 2 (and possible mask image) are inconsistent in size.')
    
    # Preprocess images.
    if pivPar['iaPreprocMethod'].lower() == 'minmax':
        im1 = min_max_filter(im1, pivPar, imMask1)
        im2 = min_max_filter(im2, pivPar, imMask2)
    pivData['imArray1'] = im1
    pivData['imArray2'] = im2
    
else:
    # if images read and preprocessed in previous pass, read them from pivData variable
    im1 = pivData['imArray1']
    im2 = pivData['imArray2']
    imMask1 = pivData['imMaskArray1']
    imMask2 = pivData['imMaskArray2']

# 1. Compute position of IA's
# get size of the image
imSizeX = im1.shape[1]
imSizeY = im1.shape[0]

# get the number of IA's
iaNX = (imSizeX - iaSizeX) // iaStepX + 1
iaNY = (imSizeY - iaSizeY) // iaStepY + 1

# distribute IA's (undeformed image, no offset):
auxLengthX = iaStepX * (iaNX - 1) + iaSizeX
auxLengthY = iaStepY * (iaNY - 1) + iaSizeY
auxFirstIAX = (imSizeX - auxLengthX) // 2 + 1
auxFirstIAY = (imSizeY - auxLengthY) // 2 + 1
iaStartX = np.arange(auxFirstIAX, auxFirstIAX + iaNX * iaStepX, iaStepX)
iaStartY = np.arange(auxFirstIAY, auxFirstIAY + iaNY * iaStepY, iaStepY)
iaStopX = iaStartX + iaSizeX - 1
iaStopY = iaStartY + iaSizeY - 1
iaCenterX = (iaStartX + iaStopX) / 2
iaCenterY = (iaStartY + iaStopY) / 2
X, Y = np.meshgrid(iaCenterX, iaCenterY)

# initialize status variable
status = np.zeros((iaNY, iaNX))

# if velocity estimate is not specified, initialize it. If velocity field is specified, use it as velocity estimation:
if 'X' not in pivData or 'Y' not in pivData or 'U' not in pivData or 'V' not in pivData:
    Xest = X
    Yest = Y
    Uest = np.zeros((iaNY, iaNX))
    Vest = np.zeros((iaNY, iaNX))
else:
    Xest = pivData['X']
    Yest = pivData['Y']
    Uest = pivData['U']
    Vest = pivData['V']

# if velocity estimation is zero (happens also if not specified), set method as 'basic'
if np.max(np.abs(Uest)) + np.max(np.abs(Vest)) < 20 * np.finfo(float).eps:
    pivPar['iaMethod'] = 'basic'
if pivPar['iaMethod'].lower() == 'basic':
    Uest = np.zeros_like(Uest)
    Vest = np.zeros_like(Vest)

# 2. Mask images
# mark masked pixels as NaNs. Later, NaNs are replaced by mean of non-masked pixels within each IA.
if imMask1 is not None:
    imMask1 = ~imMask1.astype(bool)
    im1[imMask1] = np.nan
if imMask2 is not None:
    imMask2 = ~imMask2.astype(bool)
    im2[imMask2] = np.nan

# 3. Create expanded images
# Expanded image is an image, in which IAs are side-by-side (expanded image has size 
# [iaNX*iaSizeX,iaNY*iaSizeY]); if a pixel appears in n IAs, it will be present n times in the expanded image.
# If iaStepX == iaSizeX and iaStepY == iaSizeY and method is 'basic' (or Uest == Vest ==0), expanded image is
# the same as original image (except cropping)

# initialize expanded images
exIm1 = np.full((iaNY * iaSizeY, iaNX * iaSizeX), np.nan, dtype=np.float32)
exIm2 = np.full_like(exIm1, np.nan)

# do everything with first image, then with the second
if pivPar['iaMethod'].lower() == 'basic':
    # standard interrogation - no IA offset or deformation
    for kx in range(iaNX):
        for ky in range(iaNY):
            # get the interrogation areas
            imIA1 = im1[iaStartY[ky]:iaStopY[ky]+1, iaStartX[kx]:iaStopX[kx]+1]
            imIA2 = im2[iaStartY[ky]:iaStopY[ky]+1, iaStartX[kx]:iaStopX[kx]+1]
            # set the masked pixel to mean value of remaining pixels in the IA
            masked1 = np.isnan(imIA1)
            masked2 = np.isnan(imIA2)
            auxMean1 = np.sum(imIA1 * (~masked1)) / np.sum(~masked1)
            auxMean2 = np.sum(imIA2 * (~masked2)) / np.sum(~masked2)
            imIA1[masked1] = auxMean1
            imIA2[masked2] = auxMean2
            # copy it to expanded images
            exIm1[ky*iaSizeY:(ky+1)*iaSizeY, kx*iaSizeX:(kx+1)*iaSizeX] = imIA1
            exIm2[ky*iaSizeY:(ky+1)*iaSizeY, kx*iaSizeX:(kx+1)*iaSizeX] = imIA2
            # check the number of masked pixels, and if larger than 1/2*iaSizeX*iaSizeY, consider IA as masked or outside
            if np.sum(masked1 + masked2) > 0.5 * iaSizeX * iaSizeY:
                status[ky, kx] = 1
    # set the interpolated velocity to zeros
    U0 = np.zeros((iaNY, iaNX))
    V0 = np.zeros((iaNY, iaNX))
elif pivPar['iaMethod'].lower() == 'offset':
    # interrogation with offset of IA, no deformation of IA        
    # interpolate the velocity estimates to the new grid and round it
    if np.sum(np.isnan(Uest + Vest)) > 0:
        Uest = inpaint_nans(Uest)
        Vest = inpaint_nans(Vest)
    U0 = interp2(Xest, Yest, Uest, X, Y, 'linear')
    V0 = interp2(Xest, Yest, Vest, X, Y, 'linear')
    if np.sum(np.isnan(U0 + V0)) > 0:
        U0 = inpaint_nans(U0)
        V0 = inpaint_nans(V0)
    if pivPar['iaImageToDeform'].lower() != 'both':
        U0 = np.round(U0)
        V0 = np.round(V0)
    else:
        U0 = 2 * np.round(U0 / 2)
        V0 = 2 * np.round(V0 / 2)
    # create index matrices: in which position of corresponding pixels in image pair is stored
    auxX, auxY = np.meshgrid(np.arange(iaSizeX), np.arange(iaSizeY))
    for kx in range(iaNX):
        for ky in range(iaNY):
            # calculate the shift of IAs
            if pivPar['iaImageToDeform'].lower() == 'image1':
                dxX1 = iaStartX[kx] + auxX - U0[ky, kx]
                dxX2 = iaStartX[kx] + auxX
                dxY1 = iaStartY[ky] + auxY - V0[ky, kx]
                dxY2 = iaStartY[ky] + auxY
            elif pivPar['iaImageToDeform'].lower() == 'image2':
                dxX1 = iaStartX[kx] + auxX
                dxX2 = iaStartX[kx] + auxX + U0[ky, kx] / 2
                dxY1 = iaStartY[ky] + auxY
                dxY2 = iaStartY[ky] + auxY + V0[ky, kx] / 2
            elif pivPar['iaImageToDeform'].lower() == 'both':
                dxX1 = iaStartX[kx] + auxX - U0[ky, kx] / 2
                dxX2 = iaStartX[kx] + auxX + U0[ky, kx] / 2
                dxY1 = iaStartY[ky] + auxY - V0[ky, kx] / 2
                dxY2 = iaStartY[ky] + auxY + V0[ky, kx] / 2
            # check, where the shifted pixel is out of image (shifted IA goes outside image), and set the
            # corresponding index to 1 (will be corrected later). Pixels outside will be treated as masked
            masked1 = (dxX1 < 1) | (dxX1 > im1.shape[1]) | (dxY1 < 1) | (dxY1 > im1.shape[0])
            masked2 = (dxX2 < 1) | (dxX2 > im2.shape[1]) | (dxY2 < 1) | (dxY2 > im2.shape[0])
            dxX1[masked1] = 1
            dxY1[masked1] = 1
            dxX2[masked2] = 1
            dxY2[masked2] = 1
            # convert double-indexing of matrix (M(k,j)) to single index (M(l), l = (j-1)*rows + k
            dxS1 = (dxX1 - 1) * im1.shape[0] + dxY1
            dxS2 = (dxX2 - 1) * im2.shape[0] + dxY2
            # copy the IA from the original image
            imIA1 = im1[dxS1.astype(int)]
            imIA2 = im2[dxS2.astype(int)]
            # add masked pixel (NaNs) to outside pixels (masked1 and masked2)
            masked1 = masked1 | np.isnan(imIA1)
            masked2 = masked2 | np.isnan(imIA2)
            # correct masked pixels - replace them by mean
            auxMean1 = np.sum(imIA1 * (~masked1)) / np.sum(~masked1)
            auxMean2 = np.sum(imIA2 * (~masked2))
            imIA1[masked1] = auxMean1
            imIA2[masked2] = auxMean2
            # copy IA to the expanded image
            exIm1[ky*iaSizeY:(ky+1)*iaSizeY, kx*iaSizeX:(kx+1)*iaSizeX] = imIA1
            exIm2[ky*iaSizeY:(ky+1)*iaSizeY, kx*iaSizeX:(kx+1)*iaSizeX] = imIA2
            # check the number of masked or outside pixels, and if larger than 1/2*iaSizeX*iaSizeY,
            # consider IA as masked or outside
            if np.sum(masked1 + masked2) > 0.5 * iaSizeX * iaSizeY:
                status[ky, kx] = 1
else:
    raise ValueError('Unknown iaMethod.')

# 4. Output results via pivData variable
pivData['X'] = X
pivData['Y'] = Y
pivData['U'] = np.full_like(X, np.nan)
pivData['V'] = np.full_like(X, np.nan)
pivData['N'] = X.size
pivData['Status'] = status.astype(np.uint16)
pivData['maskedN'] = np.sum(status & 1)
pivData['imSizeX'] = imSizeX
pivData['imSizeY'] = imSizeY
pivData['iaSizeX'] = iaSizeX
pivData['iaSizeY'] = iaSizeY
pivData['iaStepX'] = iaStepX
pivData['iaStepY'] = iaStepY
pivData['iaU0'] = U0
pivData['iaV0'] = V0

# %% LOCAL FUNCTIONS

import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, uniform_filter
from scipy.ndimage.morphology import binary_dilation

def min_max_filter(im, piv_par, mask=None):
    """
    MinMax filter - corrects uneven background and normalizes image contrast
    adapted following algorithm described on p. 250, Ref. [1]
    """
    S = piv_par['iaMinMaxSize']
    L = piv_par['iaMinMaxLevel']
    
    # create masking matrix (ones in a circular matrix)
    domain = np.ones((S, S))
    auxX = np.ones((S, 1)) * np.arange(-(S-1)/2, (S-1)/2 + 1)
    auxY = np.arange(-(S-1)/2, (S-1)/2 + 1).reshape(-1, 1) * np.ones((1, S))
    auxD = np.sqrt(auxX**2 + auxY**2)
    domain[auxD + 1/4 >= (S-1)/2] = 0
    N = np.sum(domain)
    domain = domain.astype(np.float32)
    
    # if no Mask is specified, skip it
    masking = False
    if mask is not None and mask.size > 0:
        mask = (mask == 0)  # Masked pixels are 1 now
        masking = np.sum(mask) > 0
    
    # Compute local low value and filter it
    im = im.astype(np.float64)
    if masking:
        im1 = im.copy()
        im1[mask] = np.max(im1)
        lo = minimum_filter(im1, footprint=domain, mode='mirror')
        lo = uniform_filter(lo, size=S)
    else:
        lo = minimum_filter(im, footprint=domain, mode='mirror')
        lo = uniform_filter(lo, size=S)
    
    # Compute local high value and filter it
    if masking:
        im1 = im.copy()
        im1[mask] = np.min(im1)
        hi = maximum_filter(im1, footprint=domain, mode='mirror')
        hi = uniform_filter(hi, size=S)
    else:
        hi = maximum_filter(im, footprint=domain, mode='mirror')
        hi = uniform_filter(hi, size=S)
    
    # enlarge mask (pixels in enlarged mask will not be considered during normalization)
    if masking:
        mask_f = binary_dilation(mask, structure=domain)
    
    # compute contrast and put lower limit on it
    contrast = hi - lo
    contrast = np.where(contrast > L, contrast - L, L)
    corrected = (im - lo) / contrast
    
    # normalize image
    corr_max = corrected.copy()
    if masking:
        corr_max[mask_f] = 0
    corr_max = np.max(corr_max[S:-S, S:-S])
    corrected = 255 * corrected / corr_max
    corrected[corrected > 255] = 255
    
    return corrected.astype(np.float32)

import os
import re

def treat_img_path(path):
    """
    Separate the path to get the folder, filename, and number if contained in the name.
    """
    filename = ''
    img_no = None
    folder = ''
    
    if len(path) > 0:
        path_reversed = path[::-1]
        I = path_reversed.find('/') if '/' in path_reversed else path_reversed.find('\\')
        Idot = path_reversed.find('.')
        
        try:
            folder = path_reversed[I+1:][::-1]
        except:
            folder = ''
        
        try:
            filename = path_reversed[Idot+1:I][::-1]
        except:
            filename = ''
        
        try:
            aux = re.findall(r'\d', filename)
            img_no = int(''.join(aux))
        except:
            img_no = None
    
    return img_no, filename, folder

function [out] = indexi(array,I1,I2)
% index array for getting values between items (if In is not integer, give value on midway betweem pixels);
if numel(array) == 0
    out = 0; 
    return;
end
A1 = array(floor(I1),floor(I2));
A2 = array(ceil(I1),floor(I2));
A3 = array(floor(I1),ceil(I2));
A4 = array(ceil(I1),ceil(I2));
out = (A1+A2+A3+A4)/4;
end
