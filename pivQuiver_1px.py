import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Union

def piv_quiver(piv_data: Dict[str, Any], *args: Any) -> None:
    """
    Displays a quiver plot with the velocity field from PIV analysis with a background image.

    Usage:
        1. piv_quiver(piv_data, whatToPlot1, option1, option1Value, option2, option2value, ...)
            Show data for an image pair.
        2. piv_quiver(piv_data, 'timeSlice', timeSliceNo, whatToPlot1, option1, option1Value, ...)
            Choose a time slice of an image sequence and show data for it.
        3. piv_quiver(piv_data, 'XYrange', [xmin, xmax, ymin, ymax], whatToPlot1, option1, option1Value, ...)
            Choose a part of the spatial extent and show data for it.
        4. piv_quiver(piv_data, 'timeSlice', timeSliceNo, 'XYrange', [xmin, xmax, ymin, ymax], whatToPlot1, option1, option1Value, ...)
        5. piv_quiver(piv_data, cellWithOptions);

    Inputs:
        piv_data ... structure containing results of PIV analysis.
        whatToPlot ... string containing information on what should be shown. Possible values are:
            'Umag' ... plot background with velocity magnitude.
            'UmagMean' ... plot background with magnitude of a time-average velocity vector.
            'U', 'V' ... plot background with velocity components.
            'Umean', 'Vmean' ... plot background with mean values of velocity components.
            'ccPeak' ... plot background with the amplitude of the cross-correlation peak.
            'ccPeakMean' ... plot background with the time-average of amplitude of the cross-correlation peak.
            'ccPeak2nd' ... plot background with the amplitude of the secondary cross-correlation peak.
            'ccPeak2ndMean' ... plot background with the time-average of amplitude of the secondary cross-correlation peak.
            'ccDetect' ... plot background with the detectability (ccPeak/ccPeakSecondary).
            'ccDetectMean' ... plot background with the mean detectability (ccPeak/ccPeakSecondary).
            'ccStd', 'ccStd1', 'ccStd2' ... plot background with RMS values in the IA (either average of both images, image1, or image2).
            'ccMean', 'ccMean1', 'ccMean2' ... plot background with mean pixel values in the IA (either average of both images, image1, or image2).
            'k' ... plot background with turbulence energy k (must be previously computed).
            'RSuu', 'RSvv', 'RSuv' ... plot background with Reynolds stress (horizontal normal, vertical normal, or shear).
            'vort' ... plot background with vorticity (must be computed by pivManipulateData).
            'epsLEPIV', 'epsMeanLEPIV' ... instantaneous and time-averaged energy dissipation rate, determined by LEPIV (must be computed by pivManipulateData).
            'image1', 'image2' ... show image in the background (works only if images or paths to them are stored in pivData).
            'imageSup' ... show superposed images in the background (works only if images or paths to them are stored in pivData).
            'quiver' ... show quiver plot with the velocity field.
            'invLoc' ... mark spurious vectors with a cross.
            'spQuiver' ... show quiver plot with the mean velocity field obtained by single-pixel correlation.
            'spU', 'spV' ... plot background with velocity components obtained by single-pixel cross-correlation; automatically selects between 'spUfit', 'spUint', or 'spU0' (or equivalents for V).
            'spUmag' ... plot background with mean velocity magnitude obtained by single-pixel cross-correlation; automatically selects between 'spUfitmag', 'spUintmag', or 'spU0mag'.
            'spU0', 'spUint', 'spUfit' ... plot background with velocity components obtained by single-pixel cross-correlation, with velocity determined by sub-pixel peak interpolation, integration of CC, or by optimization, respectively; horizontal velocity component.
            'spV0', 'spVint', 'spVfit' ... as option above, but vertical velocity component.
            'spUfilt', 'spVfilt', 'spUfiltMag' ... ############# Doplnit #########################
            'spU0mag', 'spUintmag', 'spUfitmag' ... as options above, but plots velocity magnitude.
            'spC1int', 'spC2int', 'spPhiint', 'spC0int' ... results of characterization of CC peak using its integrals.
            'spC1fit', 'spC2fit', 'spPhifit', 'spC0fit' ... results of characterization of CC peak using its fitting by 2D Gaussian.
            'spCCmax' ... maximum value of CC.
            'spACC1int', 'spACC2int', 'spACPhiint', 'spACC0int' ... results of characterization of AC peak using its integrals.
            'spACC1fit', 'spACC2fit', 'spACPhifit', 'spACC0fit' ... results of characterization of AC peak using its fitting by 2D Gaussian.
            'spACmax' ... maximum value of AC.
            'spCC' ... plot background with expanded single-pixel cross-correlation function. It is recommended to use this only on a part of image using 'crop' options.
            'spAC' ... shows expanded single-pixel auto-correlation function. It is recommended to use it only on a part of image using 'crop' options.
            'spdUdX', 'spdUdY', 'spdVdX', 'spdVdY' ... derivatives of the velocity field ################ doplnit
        optionX ... string defining which option will be set. Possible values are:
            'colorMap' ... colormap for background plot. See help for colormap in Matlab's documentation.
            'clipLo', 'clipHi' ... set minimum and maximum value of plotted quantity. If applied to quiver plot, applies to velocity magnitude.
            'subtractU', 'subtractV' ... velocity which is subtracted from the velocity field. Typically, mean velocity can be subtracted to visualize unsteady structures superposed on the mean flow.
            'lineSpec' ... (only for quiver) specify appearance of vectors in the quiver plot. Use standard Matlab specifications as in plot command (e.g., '-k' for black solid line, ':y' for dotted yellow vectors, etc.).
            'qScale' ... (only for quiver) define length of vectors. If positive, length of vectors is qScale times displacement. If negative, vector length is (-qScale * autoScale), where autoScale is that one for which 20% of vectors are longer than their distance, and 80% are shorter.
            'crop' ... limits the range of coordinates for which the plots are shown. Should be followed by array [xmin, xmax, ymin, ymax]. If X or Y should not be limited, adjust it to -Inf or +Inf. This option should be used prior to "WhatToDraw" arguments, as it applies to drawing commands which follow this option.
            'selectLo', 'selectHi' ... (only for quiver) only vectors whose length is longer than selectLo, and/or shorter than 'selectHi', will be shown.
            'selectStat' ... (only for quiver) only vectors with a specific status will be shown. Possible values for status are:
                'valid' ... only non-replaced vectors will be shown.
                'replaced' ... only replaced vectors will be shown.
                'ccFailed' ... only replaced vectors at positions where cross-correlation failed will be shown.
                'invalid' ... only replaced vectors at positions in which validation indicated spurious vectors will be shown.
            'selectMult' ... (only for quiver) only vectors having a specific multiplicator factor will be shown.
            'vecPos' ... (only for quiver) defines where arrows should be located. If 0, arrows originate in measurement location (center of interrogation area). If 1, arrows finish in the measurement location. If 0.5 (default), arrows midpoint are in measurement positions.
            'timeSlice' ... (applies only to data containing velocity sequences) select which time slice will be considered thereafter.
            'title' ... writes a title to the plot (should be followed by a string with the title).
        optionXValue ... value of the optional parameter.

    Examples:
        piv_quiver(piv_data, 'Umag', 'quiver') ... show background with velocity magnitude and plot velocity vectors.
        piv_quiver(piv_data, 'Umag', 'clipHi', 5, 'quiver', 'selectStat', 'valid', 'lineSpec', '-k', 'quiver', 'selectStat', 'replaced', 'lineSpec', '-w') ... show velocity magnitude (clipping longer displacement than 5 pixels), and the velocity field (valid vectors by black, replaced by white color).
        piv_quiver(piv_data, 'UmagMean', 'timeSlice', 1, 'quiver', 'lineSpec', '-k', 'timeSlice', 10, 'quiver', 'lineSpec', '-y') ... show mean velocity magnitude and two instantaneous velocity fields at different times, by black and yellow vector.
    """
    # Initializations
    piv_slice = piv_data.copy()
    piv_all_slices = piv_data.copy()
    slice_no = np.nan
    xmin, xmax, ymin, ymax = -np.inf, np.inf, -np.inf, np.inf

    # Parse input arguments
    inputs = args if isinstance(args[0], (list, tuple)) else list(args)
    ki = 0

    while ki < len(inputs):
        if isinstance(inputs[ki], str):
            command = inputs[ki].lower()
            if command == 'timeslice':
                try:
                    slice_no = inputs[ki + 1]
                    ki += 1
                    piv_slice = piv_manipulate_data('readTimeSlice', piv_data, slice_no)
                    if np.isinf(xmin) and np.isinf(xmax) and np.isinf(ymin) and np.isinf(ymax):
                        piv_slice = piv_manipulate_data('limitX', piv_slice, [xmin, xmax])
                        piv_slice = piv_manipulate_data('limitY', piv_slice, [ymin, ymax])
                except Exception as e:
                    print(f"Error (pivQuiver): Error when selecting time slice. {e}")
                    return

            elif command == 'crop':
                try:
                    aux_range = inputs[ki + 1]
                    xmin, xmax, ymin, ymax = aux_range
                    ki += 1
                    if not np.isnan(slice_no):
                        piv_slice = piv_manipulate_data('readTimeSlice', piv_data, slice_no)
                    else:
                        piv_slice = piv_data
                    piv_slice = piv_manipulate_data('limitX', piv_slice, [xmin, xmax])
                    piv_slice = piv_manipulate_data('limitY', piv_slice, [ymin, ymax])
                    piv_all_slices = piv_data
                    piv_all_slices = piv_manipulate_data('limitX', piv_all_slices, [xmin, xmax])
                    piv_all_slices = piv_manipulate_data('limitY', piv_all_slices, [ymin, ymax])
                except Exception as e:
                    print(f"Error (pivQuiver): Error when selecting XY range. {e}")
                    return

            elif command in ['umag', 'umagmean', 'spumag', 'spu0mag', 'spuintmag', 'spufitmag', 'spufiltmag',
                             'u', 'umean', 'spu', 'spu0', 'spuint', 'spufit', 'spufilt',
                             'v', 'vmean', 'spv', 'spv0', 'spvint', 'spvfit', 'spvfilt']:
                options = {
                    'colormap': plt.cm.jet,
                    'clipLo': -np.inf,
                    'clipHi': np.inf,
                    'subtractU': 0,
                    'subtractV': 0,
                    'spMinCC': -np.inf
                }
                options, ki = parse_options(inputs, ki + 1, options)
                data = piv_slice

                if command in ['umag', 'u', 'v']:
                    if options['subtractU'] == 'mean':
                        options['subtractU'] = np.nanmean(data['U'])
                    if options['subtractV'] == 'mean':
                        options['subtractV'] = np.nanmean(data['V'])
                    qu = data['U'] - options['subtractU']
                    qv = data['V'] - options['subtractV']
                    xmin, xmax = np.min(data['X']), np.max(data['X'])
                    ymin, ymax = np.min(data['Y']), np.max(data['Y'])

                elif command in ['umagmean', 'umean', 'vmean']:
                    if options['subtractU'] == 'mean':
                        options['subtractU'] = np.nanmean(data['U'])
                    if options['subtractV'] == 'mean':
                        options['subtractV'] = np.nanmean(data['V'])
                    if 'Umean' in data:
                        qu = data['Umean'] - options['subtractU']
                        qv = data['Vmean'] - options['subtractV']
                    else:
                        qu = data['U'] - options['subtractU']
                        qv = data['V'] - options['subtractV']
                    xmin, xmax = np.min(data['X']), np.max(data['X'])
                    ymin, ymax = np.min(data['Y']), np.max(data['Y'])

                elif command in ['spumag', 'spu', 'spv']:
                    if options['subtractU'] == 'mean':
                        options['subtractU'] = np.nanmean(data['U'])
                    if options['subtractV'] == 'mean':
                        options['subtractV'] = np.nanmean(data['V'])
                    if np.sum(~np.isnan(data['spUfit'])) > 1:
                        qu = data['spUfit'] - options['subtractU']
                        qv = data['spVfit'] - options['subtractV']
                    elif np.sum(~np.isnan(data['spUint'])) > 1:
                        qu = data['spUint'] - options['subtractU']
                        qv = data['spVint'] - options['subtractV']
                    else:
                        qu = data['spU0'] - options['subtractU']
                        qv = data['spV0'] - options['subtractV']
                    Ccmax = data['spCCmax']
                    xmin, xmax = np.min(data['spX']), np.max(data['spX'])
                    ymin, ymax = np.min(data['spY']), np.max(data['spY'])
                    aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                    qu[aux_NOK] = np.nan
                    qv[aux_NOK] = np.nan

                elif command in ['spu0mag', 'spu0', 'spv0']:
                    if options['subtractU'] == 'mean':
                        options['subtractU'] = np.nanmean(data['U'])
                    if options['subtractV'] == 'mean':
                        options['subtractV'] = np.nanmean(data['V'])
                    qu = data['spU0'] - options['subtractU']
                    qv = data['spV0'] - options['subtractV']
                    Ccmax = data['spCCmax']
                    xmin, xmax = np.min(data['spX']), np.max(data['spX'])
                    ymin, ymax = np.min(data['spY']), np.max(data['spY'])
                    aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                    qu[aux_NOK] = np.nan
                    qv[aux_NOK] = np.nan

                elif command in ['spu0int', 'spuint', 'spvint']:
                    if options['subtractU'] == 'mean':
                        options['subtractU'] = np.nanmean(data['U'])
                    if options['subtractV'] == 'mean':
                        options['subtractV'] = np.nanmean(data['V'])
                    qu = data['spUint'] - options['subtractU']
                    qv = data['spVint'] - options['subtractV']
                    Ccmax = data['spCCmax']
                    xmin, xmax = np.min(data['spX']), np.max(data['spX'])
                    ymin, ymax = np.min(data['spY']), np.max(data['spY'])
                    aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                    qu[aux_NOK] = np.nan
                    qv[aux_NOK] = np.nan

                elif command in ['spumagfit', 'spufit', 'spvfit']:
                    if options['subtractU'] == 'mean':
                        options['subtractU'] = np.nanmean(data['U'])
                    if options['subtractV'] == 'mean':
                        options['subtractV'] = np.nanmean(data['V'])
                    qu = data['spUfit'] - options['subtractU']
                    qv = data['spVfit'] - options['subtractV']
                    Ccmax = data['spCCmax']
                    xmin, xmax = np.min(data['spX']), np.max(data['spX'])
                    ymin, ymax = np.min(data['spY']), np.max(data['spY'])
                    aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                    qu[aux_NOK] = np.nan
                    qv[aux_NOK] = np.nan

                elif command in ['spufiltmag', 'spufilt', 'spvfilt']:
                    if options['subtractU'] == 'mean':
                        options['subtractU'] = np.nanmean(data['U'])
                    if options['subtractV'] == 'mean':
                        options['subtractV'] = np.nanmean(data['V'])
                    qu = data['spUfiltered'] - options['subtractU']
                    qv = data['spVfiltered'] - options['subtractV']
                    Ccmax = data['spCCmax']
                    xmin, xmax = np.min(data['spX']), np.max(data['spX'])
                    ymin, ymax = np.min(data['spY']), np.max(data['spY'])
                    aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                    qu[aux_NOK] = np.nan
                    qv[aux_NOK] = np.nan

                # Calculate Umag and clip it
                if qu.ndim > 2 and (command in ['umag', 'u', 'v']):
                    print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                    qu = np.mean(qu, axis=2)
                    qv = np.mean(qv, axis=2)
                elif command in ['umagmean', 'umean', 'vmean']:
                    qu = np.mean(qu, axis=2)
                    qv = np.mean(qv, axis=2)

                if command in ['umag', 'umagmean', 'spumag', 'spu0mag', 'spuintmag', 'spufitmag', 'spufiltmag']:
                    q = np.sqrt(qu**2 + qv**2)
                elif command in ['u', 'spu', 'umean', 'spu0', 'spuint', 'spufit', 'spufilt']:
                    q = qu
                elif command in ['v', 'spv', 'vmean', 'spv0', 'spvint', 'spvfit', 'spvfilt']:
                    q = qv

                q[q < options['clipLo']] = options['clipLo']
                q[q > options['clipHi']] = options['clipHi']

                plt.figure()
                plt.imshow(q, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap=options['colormap'])
                plt.colorbar()
                plt.axis('equal')
                plt.show()

            elif command in ['ccpeak', 'ccpeakmean', 'ccpeak2nd', 'ccpeak2ndmean', 'ccdetect', 'ccdetectmean',
                             'ccstd', 'ccstd1', 'ccstd2', 'ccmean', 'ccmean1', 'ccmean2',
                             'k', 'vort', 'epslepiv', 'epsmeanlepiv',
                             'rsuu', 'rsvv', 'rsuv',
                             'spc1int', 'spc2int', 'spphiint', 'spc0int',
                             'spc1fit', 'spc2fit', 'spphifit', 'spc0fit',
                             'spccmax',
                             'spacc1int', 'spacc2int', 'spacphiint', 'spacc0int',
                             'spacc1fit', 'spacc2fit', 'spacphifit', 'spacc0fit',
                             'spacmax',
                             'spdudx', 'spdudy', 'spdvdx', 'spdvdy',
                             'sprsuu', 'sprsvv', 'sprsuv']:
                options = {
                    'colormap': plt.cm.jet,
                    'clipLo': -np.inf,
                    'clipHi': np.inf,
                    'spMinCC': -np.inf
                }
                options, ki = parse_options(inputs, ki + 1, options)

                try:
                    if command == 'ccpeak':
                        data = piv_slice
                        q = data['ccPeak']
                        if q.ndim > 2:
                            print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                            q = np.mean(q, axis=2)

                    elif command == 'ccpeak2nd':
                        data = piv_slice
                        q = data['ccPeakSecondary']
                        if q.ndim > 2:
                            print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                            q = np.mean(q, axis=2)

                    elif command == 'ccpeakmean':
                        data = piv_all_slices
                        q = data['ccPeak']
                        q = np.mean(q, axis=2)

                    elif command == 'ccpeak2ndmean':
                        data = piv_all_slices
                        q = data['ccPeakSecondary']
                        aux = np.isnan(q)
                        q[aux] = np.mean(q[~aux])
                        q = np.mean(q, axis=2)

                    elif command == 'ccdetect':
                        data = piv_slice
                        q1 = data['ccPeak']
                        q2 = data['ccPeakSecondary']
                        aux = np.isnan(q2)
                        q2[aux] = q1[aux]
                        q = q1 / q2
                        if q.ndim > 2:
                            print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                            q = np.mean(q, axis=2)

                    elif command == 'ccdetectmean':
                        data = piv_all_slices
                        q1 = data['ccPeak']
                        q2 = data['ccPeakSecondary']
                        aux = np.isnan(q2)
                        q2[aux] = np.mean(q2[~aux])
                        q = q1 / q2
                        q = np.mean(q, axis=2)

                    elif command == 'ccstd':
                        data = piv_all_slices
                        q = np.sqrt(data['ccStd1'] * data['ccStd2'])
                        q = np.mean(q, axis=2)

                    elif command == 'ccstd1':
                        data = piv_all_slices
                        q = data['ccStd1']
                        q = np.mean(q, axis=2)

                    elif command == 'ccstd2':
                        data = piv_all_slices
                        q = data['ccStd2']
                        q = np.mean(q, axis=2)

                    elif command == 'ccmean':
                        data = piv_all_slices
                        q = 0.5 * (data['ccMean1'] + data['ccMean2'])
                        q = np.mean(q, axis=2)

                    elif command == 'ccmean1':
                        data = piv_all_slices
                        q = data['ccMean1']
                        q = np.mean(q, axis=2)

                    elif command == 'ccmean2':
                        data = piv_all_slices
                        q = data['ccMean2']
                        q = np.mean(q, axis=2)

                    elif command == 'k':
                        data = piv_all_slices
                        q = data['k']

                    elif command == 'vort':
                        try:
                            data = piv_slice
                            q = data['vorticity']
                        except KeyError:
                            data = piv_postprocess('vorticity', piv_slice)
                            q = data['vorticity']

                    elif command == 'epslepiv':
                        try:
                            data = piv_slice
                            q = data['epsLEPIV']
                        except KeyError:
                            data = piv_postprocess('LEPIVdissip', piv_slice)
                            q = data['epsLEPIV']

                    elif command == 'epsmeanlepiv':
                        try:
                            data = piv_slice
                            q = data['epsMeanLEPIV']
                        except KeyError:
                            data = piv_postprocess('LEPIVdissip', piv_slice)
                            q = data['epsMeanLEPIV']

                    elif command == 'rsuu':
                        data = piv_all_slices
                        q = data['RSuu']

                    elif command == 'rsvv':
                        data = piv_all_slices
                        q = data['RSvv']

                    elif command == 'rsuv':
                        data = piv_all_slices
                        q = data['RSuv']

                    elif command == 'spc1int':
                        data = piv_all_slices
                        q = data['spC1int']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spc2int':
                        data = piv_all_slices
                        q = data['spC2int']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spphiint':
                        data = piv_all_slices
                        q = data['spPhiint']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spc0int':
                        data = piv_all_slices
                        q = data['spC0int']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spc1fit':
                        data = piv_all_slices
                        q = data['spC1fit']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spc2fit':
                        data = piv_all_slices
                        q = data['spC2fit']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spphifit':
                        data = piv_all_slices
                        q = data['spPhifit']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spc0fit':
                        data = piv_all_slices
                        q = data['spC0fit']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spccmax':
                        data = piv_all_slices
                        q = data['spCCmax']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacc1int':
                        data = piv_all_slices
                        q = data['spACC1int']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacc2int':
                        data = piv_all_slices
                        q = data['spACC2int']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacphiint':
                        data = piv_all_slices
                        q = data['spACPhiint']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacc0int':
                        data = piv_all_slices
                        q = data['spACC0int']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacc1fit':
                        data = piv_all_slices
                        q = data['spACC1fit']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacc2fit':
                        data = piv_all_slices
                        q = data['spACC2fit']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacphifit':
                        data = piv_all_slices
                        q = data['spACPhifit']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacc0fit':
                        data = piv_all_slices
                        q = data['spACC0fit']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spacmax':
                        data = piv_all_slices
                        q = data['spACmax']
                        Ccmax = data['spACmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spdudx':
                        data = piv_all_slices
                        q = data['spdUdX']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spdudy':
                        data = piv_all_slices
                        q = data['spdUdY']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spdvdx':
                        data = piv_all_slices
                        q = data['spdVdX']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spdvdy':
                        data = piv_all_slices
                        q = data['spdVdY']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spd':
                        data = piv_all_slices
                        q = data['spD']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spc1':
                        data = piv_all_slices
                        q = data['spC1']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spc2':
                        data = piv_all_slices
                        q = data['spC2']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spp1':
                        data = piv_all_slices
                        q = data['spP1']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'spp2':
                        data = piv_all_slices
                        q = data['spP2']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'sprsuu':
                        data = piv_all_slices
                        q = data['spRSuu']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'sprsvv':
                        data = piv_all_slices
                        q = data['spRSvv']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    elif command == 'sprsuv':
                        data = piv_all_slices
                        q = data['spRSuv']
                        Ccmax = data['spCCmax']
                        aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                        q[aux_NOK] = np.nan

                    if command in ['ccpeak', 'ccpeak2nd', 'ccpeakmean', 'ccpeak2ndmean', 'ccdetect', 'ccdetectmean',
                                    'ccstd', 'ccstd1', 'ccstd2', 'ccmean', 'ccmean1', 'ccmean2', 'k', 'vort',
                                    'epslepiv', 'epsmeanlepiv', 'rsuu', 'rsvv', 'rsuv']:
                        xmin, xmax = np.min(data['X']), np.max(data['X'])
                        ymin, ymax = np.min(data['Y']), np.max(data['Y'])

                    elif command in ['spc1int', 'spc2int', 'spphiint', 'spc0int', 'spc1fit', 'spc2fit', 'spphifit', 'spc0fit',
                                      'spacc1int', 'spacc2int', 'spacphiint', 'spacc0int', 'spacc1fit', 'spacc2fit',
                                      'spacphifit', 'spacc0fit', 'spccmax', 'spacmax',
                                      'spdudx', 'spdudy', 'spdvdx', 'spdvdy',
                                      'spd', 'spc1', 'spc2', 'spp1', 'spp2',
                                      'sprsuu', 'sprsvv', 'sprsuv']:
                        xmin, xmax = np.min(data['spX']), np.max(data['spX'])
                        ymin, ymax = np.min(data['spY']), np.max(data['spY'])

                except Exception as e:
                    print(f"Error (pivQuiver): Failed to plot desired data ({command}). {e}")
                    return

                q[q < options['clipLo']] = options['clipLo']
                q[q > options['clipHi']] = options['clipHi']

                plt.figure()
                plt.imshow(q, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap=options['colormap'])
                plt.colorbar()
                plt.axis('equal')
                plt.show()

            elif command in ['spcc', 'spac', 'spccnorm', 'spccnorm2']:
                options = {
                    'colormap': plt.cm.jet,
                    'clipLo': -np.inf,
                    'clipHi': np.inf
                }
                options['colormap'].set_bad(color='w')
                options, ki = parse_options(inputs, ki + 1, options)
                data = piv_all_slices

                if command in ['spcc', 'spccnorm', 'spccnorm2']:
                    q = data['spCC']
                    dXNeg = data['spDeltaXNeg']
                    dXPos = data['spDeltaXPos']
                    dYNeg = data['spDeltaYNeg']
                    dYPos = data['spDeltaYPos']
                elif command == 'spac':
                    q = data['spAC']
                    dXNeg = (q.shape[2] - 1) // 2
                    dXPos = dXNeg
                    dYNeg = dXNeg
                    dYPos = dXNeg

                Qflat = np.full((q.shape[0] * (q.shape[2] + 1) + 1, q.shape[1] * (q.shape[3] + 1) + 1), np.nan)
                for ky in range(q.shape[0]):
                    for kx in range(q.shape[1]):
                        if command == 'spcc':
                            Qflat[2 + ky * (q.shape[2] + 1):(ky + 1) * (q.shape[2] + 1),
                                2 + kx * (q.shape[3] + 1):(kx + 1) * (q.shape[3] + 1)] = q[ky, kx, :, :]
                        elif command == 'spccnorm':
                            norm0 = np.min(q[ky, kx, :, :])
                            norm1 = np.max(q[ky, kx, :, :]) - np.min(q[ky, kx, :, :])
                            if norm1 < 0.01:
                                norm1 = 0.01
                            Qflat[2 + ky * (q.shape[2] + 1):(ky + 1) * (q.shape[2] + 1),
                                2 + kx * (q.shape[3] + 1):(kx + 1) * (q.shape[3] + 1)] = (q[ky, kx, :, :] - norm0) / norm1
                        elif command == 'spccnorm2':
                            norm1 = np.max(q[ky, kx, :, :]) - np.min(q[ky, kx, :, :])
                            if norm1 < 0.01:
                                norm1 = 0.01
                            Qflat[2 + ky * (q.shape[2] + 1):(ky + 1) * (q.shape[2] + 1),
                                2 + kx * (q.shape[3] + 1):(kx + 1) * (q.shape[3] + 1)] = q[ky, kx, :, :] / norm1

                Qflat[Qflat < options['clipLo']] = options['clipLo']
                Qflat[Qflat > options['clipHi']] = options['clipHi']

                plt.figure()
                dx = data['spX'][0, 1] - data['spX'][0, 0]
                dy = data['spY'][1, 0] - data['spY'][0, 0]
                plt.imshow(Qflat, extent=[np.min(data['spX']) - dx * (dXNeg + 1) / (dXNeg + dXPos + 2),
                                           np.max(data['spX']) + dx * (dXPos + 1) / (dXNeg + dXPos + 2),
                                           np.min(data['spY']) - dy * (dYNeg + 1) / (dYNeg + dYPos + 2),
                                           np.max(data['spY']) + dy * (dYPos + 1) / (dYNeg + dYPos + 2)],
                           origin='lower', cmap=options['colormap'])
                plt.colorbar()
                plt.axis('equal')
                plt.show()

            elif command in ['sppeak0', 'sppeakint', 'sppeakfit', 'spzero']:
                options = {
                    'lineSpec': '-w',
                    'spMinCC': -np.inf
                }
                options, ki = parse_options(inputs, ki + 1, options)
                data = piv_all_slices
                N = data['spX'].size
                X = data['spX'].flatten()
                Y = data['spY'].flatten()
                ccMax = data['spCCmax'].flatten()
                dXNeg = data['spDeltaXNeg']
                dXPos = data['spDeltaXPos']
                dYNeg = data['spDeltaYNeg']
                dYPos = data['spDeltaYPos']
                dx = data['spX'][0, 1] - data['spX'][0, 0]
                dy = data['spY'][1, 0] - data['spY'][0, 0]

                if command == 'spzero':
                    options['pointSpec'] = 'xy'
                    options, ki = parse_options(inputs, ki + 1, options)
                    U = np.zeros(N)
                    V = np.zeros(N)

                elif command == 'sppeak0':
                    options['pointSpec'] = 'm.'
                    options, ki = parse_options(inputs, ki + 1, options)
                    U = data['spU0'].flatten()
                    V = data['spV0'].flatten()

                elif command == 'sppeakint':
                    options['pointSpec'] = 'om'
                    options['lineParts'] = 60
                    options, ki = parse_options(inputs, ki + 1, options)
                    U = data['spUint'].flatten()
                    V = data['spVint'].flatten()
                    Uel = data['spU0'].flatten()
                    Vel = data['spV0'].flatten()
                    c1 = data['spC1int'].flatten()
                    c2 = data['spC2int'].flatten()
                    phi = data['spPhiint'].flatten()

                elif command == 'sppeakfit':
                    options['pointSpec'] = 'xm'
                    options['lineParts'] = 60
                    options, ki = parse_options(inputs, ki + 1, options)
                    U = data['spUfit'].flatten()
                    V = data['spVfit'].flatten()
                    Uel = data['spUfit'].flatten()
                    Vel = data['spVfit'].flatten()
                    c1 = data['spC1fit'].flatten()
                    c2 = data['spC2fit'].flatten()
                    phi = data['spPhifit'].flatten()

                aux_NOK = np.logical_or(ccMax < options['spMinCC'])
                CCcenterX = X + U * dx / (dXNeg + dXPos + 2)
                CCcenterY = Y + V * dy / (dYNeg + dYPos + 2)
                CCcenterX[aux_NOK] = np.nan
                CCcenterY[aux_NOK] = np.nan

                plt.figure()
                plt.plot(CCcenterX, CCcenterY, options['pointSpec'])

                if command in ['sppeakint', 'sppeakfit']:
                    phiel = np.linspace(0, 2 * np.pi, options['lineParts'])
                    Xel = np.zeros(N * (len(phiel) + 1))
                    Yel = np.zeros(N * (len(phiel) + 1))
                    for kk in range(N):
                        ksiel = 0.294 * c1[kk] * np.cos(phiel)
                        etael = 0.294 * c2[kk] * np.sin(phiel)
                        xel = Uel[kk] + ksiel * np.cos(phi[kk]) + etael * np.sin(phi[kk])
                        yel = Vel[kk] - ksiel * np.sin(phi[kk]) + etael * np.cos(phi[kk])
                        aux_NOK = np.logical_or(xel < -dXNeg - 0.5, xel > dXPos + 0.5) | np.logical_or(yel < -dYNeg - 0.5, yel > dYPos + 0.5)
                        xel[aux_NOK] = np.nan
                        yel[aux_NOK] = np.nan
                        xel = X[kk] + xel * dx / (dXNeg + dXPos + 2)
                        yel = Y[kk] + yel * dy / (dYNeg + dYPos + 2)
                        Xel[kk * (len(phiel) + 1):(kk + 1) * (len(phiel) + 1) - 1] = xel
                        Yel[kk * (len(phiel) + 1):(kk + 1) * (len(phiel) + 1) - 1] = yel

                    plt.plot(Xel, Yel, options['lineSpec'])

                plt.axis('equal')
                plt.show()

            elif command in ['image1', 'image2', 'imagesup']:
                options = {
                    'colormap': plt.cm.gray,
                    'expScale': 1
                }
                options, ki = parse_options(inputs, ki + 1, options)

                try:
                    if command == 'image1':
                        if isinstance(piv_slice['imFilename1'], str):
                            img = plt.imread(piv_slice['imFilename1'])
                        elif isinstance(piv_slice['imFilename1'], list):
                            img = plt.imread(piv_slice['imFilename1'][0])

                    elif command == 'image2':
                        if isinstance(piv_slice['imFilename2'], str):
                            img = plt.imread(piv_slice['imFilename2'])
                        elif isinstance(piv_slice['imFilename2'], list):
                            img = plt.imread(piv_slice['imFilename2'][0])

                    elif command == 'imagesup':
                        if isinstance(piv_slice['imFilename1'], str):
                            img1 = plt.imread(piv_slice['imFilename1'])
                            img2 = plt.imread(piv_slice['imFilename2'])
                        elif isinstance(piv_slice['imFilename1'], list):
                            img1 = plt.imread(piv_slice['imFilename1'][0])
                            img2 = plt.imread(piv_slice['imFilename2'][0])
                        img = np.maximum(img1, img2)

                    if np.isinf(xmin):
                        xmin = 0
                    if np.isinf(xmax):
                        xmax = img.shape[1] - 1
                    if np.isinf(ymin):
                        ymin = 0
                    if np.isinf(ymax):
                        ymax = img.shape[0] - 1

                    img = img[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1]

                    plt.figure()
                    plt.imshow(img, extent=[xmin * options['expScale'], xmax * options['expScale'],
                                             ymin * options['expScale'], ymax * options['expScale']], cmap=options['colormap'])
                    plt.axis('equal')
                    plt.colorbar()
                    plt.show()

                except Exception as e:
                    print(f"Warning (pivQuiver): Unable to read or display image(s). {e}")

            elif command == 'invloc':
                options = {
                    'lineSpec': 'xk',
                    'markerSize': 4
                }
                options, ki = parse_options(inputs, ki + 1, options)
                data = piv_slice
                X = []
                Y = []

                if 'spuriousX' in data:
                    X = data['spuriousX']
                    Y = data['spuriousY']

                fail = np.logical_or(np.bitwise_and(data['Status'], 2))
                X = np.append(X, data['X'][fail])
                Y = np.append(Y, data['Y'][fail])

                plt.figure()
                plt.plot(X, Y, options['lineSpec'], markersize=options['markerSize'])
                plt.axis('equal')
                plt.show()

            elif command in ['quiver', 'quivermean', 'spquiver']:
                options = {
                    'lineSpec': '-k',
                    'qScale': -1.5,
                    'qScalePercentile': 0.8,
                    'clipLo': -np.inf,
                    'clipHi': np.inf,
                    'subtractU': 0,
                    'subtractV': 0,
                    'selectLo': -np.inf,
                    'selectHi': np.inf,
                    'selectStat': 'all',
                    'selectMult': 0,
                    'vecpos': 0.5,
                    'spMinCC': -np.inf
                }
                options, ki = parse_options(inputs, ki + 1, options)
                if options['subtractU'] == 'mean':
                    options['subtractU'] = np.nanmean(data['U'])
                if options['subtractV'] == 'mean':
                    options['subtractV'] = np.nanmean(data['V'])

                if command == 'quiver':
                    data = piv_slice
                    if 'N' not in data:
                        data['N'] = len(data['X'])
                    X = data['X'].flatten()
                    Y = data['Y'].flatten()
                    U = (data['U'] - options['subtractU']).flatten()
                    V = (data['V'] - options['subtractV']).flatten()
                    S = data['Status'].flatten()
                    dx = data['iaStepX']
                    dy = data['iaStepY']
                    xmin = np.min(data['X']) - dx / 2
                    xmax = np.max(data['X']) + dx / 2
                    ymin = np.min(data['Y']) - dy / 2
                    ymax = np.max(data['Y']) + dy / 2

                elif command == 'quivermean':
                    data = piv_all_slices
                    if 'N' not in data:
                        data['N'] = len(data['X'])
                    X = data['X'].flatten()
                    Y = data['Y'].flatten()
                    if 'Umean' in data:
                        U = (data['Umean'] - options['subtractU']).flatten()
                        V = (data['Vmean'] - options['subtractV']).flatten()
                    else:
                        U = (np.mean(data['U'], axis=2) - options['subtractU']).flatten()
                        V = (np.mean(data['V'], axis=2) - options['subtractV']).flatten()
                    S = np.zeros(data['N'])
                    dx = data['iaStepX']
                    dy = data['iaStepY']
                    xmin = np.min(data['X']) - dx / 2
                    xmax = np.max(data['X']) + dx / 2
                    ymin = np.min(data['Y']) - dy / 2
                    ymax = np.max(data['Y']) + dy / 2

                elif command == 'spquiver':
                    data = piv_all_slices
                    data['N'] = len(data['spX'])
                    X = data['spX'].flatten()
                    Y = data['spY'].flatten()
                    Ccmax = data['spCCmax'].flatten()
                    U = (np.mean(data['spUfit'], axis=2) - options['subtractU']).flatten()
                    V = (np.mean(data['spVfit'], axis=2) - options['subtractV']).flatten()
                    S = np.zeros(data['N'])
                    dx = data['spX'][0, 1] - data['spX'][0, 0]
                    dy = data['spY'][1, 0] - data['spY'][0, 0]
                    xmin = np.min(data['spX']) - dx / 2
                    xmax = np.max(data['spX']) + dx / 2
                    ymin = np.min(data['spY']) - dy / 2
                    ymax = np.max(data['spY']) + dy / 2
                    aux_NOK = np.logical_or(Ccmax < options['spMinCC'])
                    U[aux_NOK] = np.nan
                    V[aux_NOK] = np.nan

                if 'multiplicator' in data:
                    M = data['multiplicator'][:, :, 0].flatten()
                else:
                    M = np.zeros(data['N'])

                Umag = np.sqrt(U**2 + V**2)
                localScale = np.ones(data['N'])
                aux = np.logical_or(Umag > options['clipHi'])
                localScale[aux] = options['clipHi'] / Umag[aux]
                aux = np.logical_or(Umag < options['clipLo'])
                localScale[aux] = options['clipLo'] / Umag[aux]

                auxUmag = np.sort(Umag[~np.isnan(Umag)])
                auxUmag = auxUmag[int(options['qScalePercentile'] * len(auxUmag))]
                if np.isnan(options['qScale']) or options['qScale'] == 0:
                    options['qScale'] = 2 * np.sqrt(dx**2 + dy**2) / auxUmag
                elif options['qScale'] < 0:
                    options['qScale'] = -options['qScale'] * np.sqrt(dx**2 + dy**2) / auxUmag

                U = U * localScale * options['qScale']
                V = V * localScale * options['qScale']

                ok = np.logical_and(~np.isnan(U), ~np.isnan(V)) & np.logical_and(Umag > options['selectLo'], Umag < options['selectHi'])

                if options['selectStat'] == 'all':
                    pass
                elif options['selectStat'] == 'valid':
                    ok = np.logical_and(ok, ~np.logical_or(np.bitwise_and(S, 5), np.bitwise_and(S, 8)))
                elif options['selectStat'] == 'replaced':
                    ok = np.logical_and(ok, np.logical_or(np.bitwise_and(S, 5), np.bitwise_and(S, 5)))
                elif options['selectStat'] == 'ccfailed':
                    ok = np.logical_and(ok, ~np.logical_or(np.bitwise_and(S, 3), np.bitwise_and(S, 2)))
                elif options['selectStat'] == 'invalid':
                    ok = np.logical_and(ok, np.logical_or(np.bitwise_and(S, 4), np.bitwise_and(S, 7)))

                if options['selectMult'] != 0:
                    ok = np.logical_and(ok, M == options['selectMult'])

                X = X - options['vecpos'] * U
                Y = Y - options['vecpos'] * V

                X = X[ok]
                Y = Y[ok]
                U = U[ok]
                V = V[ok]

                if 'imSizeX' in piv_data:
                    inFrame = np.logical_and(np.logical_and(X >= xmin, Y >= ymin), np.logical_and(X <= xmax, Y <= ymax)) & np.logical_and(np.logical_and(X + U >= xmin, Y + V >= ymin), np.logical_and(X + U <= xmax, Y + V <= ymax))
                else:
                    inFrame = np.ones_like(X, dtype=bool)

                plt.figure()
                plt.quiver(X[inFrame], Y[inFrame], U[inFrame], V[inFrame], angles='xy', scale_units='xy', scale=1, color=options['lineSpec'])
                plt.axis('equal')
                plt.gca().invert_yaxis()
                plt.show()

            elif command == 'title':
                try:
                    title_text = inputs[ki + 1]
                    ki += 1
                    plt.title(title_text)
                except Exception as e:
                    print(f"Warning (pivQuiver): Title badly specified; ignoring 'title' options. {e}")

            else:
                print(f"Warning (pivQuiver): Unable to parse input '{inputs[ki]}'. Ignoring it.")

        else:
            print(f"Warning (pivQuiver): Unable to parse input {ki}th input. Ignoring it.")

        ki += 1

    plt.show()

def parse_options(inputs: List[Any], ki: int, defaults: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Extract commands from input cell.

    Args:
        inputs (List[Any]): List of input arguments.
        ki (int): Current index in the input list.
        defaults (Dict[str, Any]): Default options.

    Returns:
        Tuple[Dict[str, Any], int]: Parsed options and the new index.
    """
    options = defaults.copy()
    success = len(inputs) - ki + 1 >= 2
    while success:
        try:
            command = inputs[ki]
            value = inputs[ki + 1]
            success = False
            for name in defaults.keys():
                if command.lower() == name.lower():
                    success = True
                    options[name] = value
                    ki += 2
                    break
        except Exception:
            success = False
    return options, ki - 1

def vorticity_colormap() -> List[List[float]]:
    """
    Color map used for vorticity display.

    Returns:
        List[List[float]]: Custom colormap for vorticity.
    """
    cm = np.ones((256, 3))
    cm[129:, 1] = np.linspace(1, 0, 127)
    cm[129:, 2] = np.linspace(1, 0, 127)
    cm[:128, 0] = np.linspace(0, 1, 128)
    cm[:128, 1] = np.linspace(0, 1, 128)
    return cm
