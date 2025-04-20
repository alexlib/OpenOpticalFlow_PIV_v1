import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, uniform_filter
from scipy.ndimage.morphology import binary_dilation
import os
import re
from imageio.v2 import imread
from scipy.interpolate import griddata

def inpaint_nans(array):
    """Simple function to replace NaNs with interpolated values"""
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Find NaN values
    mask = np.isnan(array)
    # Get only valid values
    x1 = xx[~mask]
    y1 = yy[~mask]
    values = array[~mask]

    # Interpolate using nearest neighbor
    return griddata((x1, y1), values, (xx, yy), method='nearest')

def interp2(x, y, z, xi, yi, method='linear'):
    """Simple 2D interpolation function similar to MATLAB's interp2"""
    return griddata((x.flatten(), y.flatten()), z.flatten(), (xi, yi), method=method)

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

def piv_interrogate(im1, im2, pivData, pivPar):
    """
    Splits images into interrogation areas and create expanded images suitable for pivCrossCorr.

    Parameters:
        im1, im2: Image pair (either images, or strings containing paths to image files)
        pivData: Dictionary containing detailed results. No fields are required as the input, but
                the existing fields will be copied to pivData at the output. If exist, following fields will be used:
            X, Y: Position at which U, V velocity/displacement is provided
            U, V: Displacements in x and y direction (will be used for image deformation)
        pivPar: Parameters defining the evaluation

    Returns:
        exIm1, exIm2: Expanded images 1 and 2
        pivData: Updated dictionary with additional fields
    """
    # This is a simplified implementation of the piv_interrogate function
    # It creates basic interrogation areas without deformation

    # Extract parameters from pivPar
    iaSizeX = pivPar['iaSizeX']
    iaSizeY = pivPar['iaSizeY']
    iaStepX = pivPar['iaStepX']
    iaStepY = pivPar['iaStepY']

    # Process input images if they are file paths
    if isinstance(im1, str):
        img_no, filename, folder = treat_img_path(im1)
        pivData['imFilename1'] = im1
        pivData['imNo1'] = img_no
        im1 = imread(im1)

    if isinstance(im2, str):
        img_no, filename, folder = treat_img_path(im2)
        pivData['imFilename2'] = im2
        pivData['imNo2'] = img_no
        im2 = imread(im2)

    # Convert images to float32
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # Get image dimensions
    imSizeY, imSizeX = im1.shape

    # Calculate number of interrogation areas
    iaNX = (imSizeX - iaSizeX) // iaStepX + 1
    iaNY = (imSizeY - iaSizeY) // iaStepY + 1

    # Calculate positions of interrogation areas
    iaStartX = np.arange(0, iaNX * iaStepX, iaStepX)
    iaStartY = np.arange(0, iaNY * iaStepY, iaStepY)
    iaStopX = iaStartX + iaSizeX - 1
    iaStopY = iaStartY + iaSizeY - 1
    iaCenterX = (iaStartX + iaStopX) / 2
    iaCenterY = (iaStartY + iaStopY) / 2
    X, Y = np.meshgrid(iaCenterX, iaCenterY)

    # Initialize expanded images
    exIm1 = np.zeros((iaNY * iaSizeY, iaNX * iaSizeX), dtype=np.float32)
    exIm2 = np.zeros_like(exIm1)

    # Initialize status array
    status = np.zeros((iaNY, iaNX), dtype=np.uint16)

    # Extract interrogation areas
    for kx in range(iaNX):
        for ky in range(iaNY):
            # Get interrogation areas
            y_start = int(iaStartY[ky])
            y_end = int(min(iaStopY[ky] + 1, imSizeY))
            x_start = int(iaStartX[kx])
            x_end = int(min(iaStopX[kx] + 1, imSizeX))

            imIA1 = im1[y_start:y_end, x_start:x_end]
            imIA2 = im2[y_start:y_end, x_start:x_end]

            # Copy to expanded images
            exIm1[ky*iaSizeY:ky*iaSizeY+(y_end-y_start), kx*iaSizeX:kx*iaSizeX+(x_end-x_start)] = imIA1
            exIm2[ky*iaSizeY:ky*iaSizeY+(y_end-y_start), kx*iaSizeX:kx*iaSizeX+(x_end-x_start)] = imIA2

    # Set zero velocity
    U0 = np.zeros((iaNY, iaNX))
    V0 = np.zeros((iaNY, iaNX))

    # Update pivData with results
    pivData['X'] = X
    pivData['Y'] = Y
    pivData['U'] = np.full_like(X, np.nan)
    pivData['V'] = np.full_like(X, np.nan)
    pivData['N'] = X.size
    pivData['Status'] = status
    pivData['maskedN'] = np.sum(status & 1)
    pivData['imSizeX'] = imSizeX
    pivData['imSizeY'] = imSizeY
    pivData['iaSizeX'] = iaSizeX
    pivData['iaSizeY'] = iaSizeY
    pivData['iaStepX'] = iaStepX
    pivData['iaStepY'] = iaStepY
    pivData['iaU0'] = U0
    pivData['iaV0'] = V0

    return exIm1, exIm2, pivData


def indexi(array, I1, I2):
    """Index array for getting values between items (if In is not integer, give value on midway between pixels)"""
    if array.size == 0:
        return 0

    A1 = array[int(np.floor(I1)), int(np.floor(I2))]
    A2 = array[int(np.ceil(I1)), int(np.floor(I2))]
    A3 = array[int(np.floor(I1)), int(np.ceil(I2))]
    A4 = array[int(np.ceil(I1)), int(np.ceil(I2))]
    return (A1 + A2 + A3 + A4) / 4
