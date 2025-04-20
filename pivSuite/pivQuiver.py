function pivQuiver(pivData,varargin)
% pivQuiver - displays quiver plot with the velocity field from PIV analysis with a background image
%
% Usage:
%    1. pivQuiver(pivData,whatToPlot1,option1,option1Value,option2,option2value,...,whatToPlot2,optionN,...)
%          show data for an image pair
%    2. pivQuiver(pivData,'timeSlice',timeSliceNo,whatToPlot1,option1,option1Value,...)
%          choose a timeslice of an image sequence and show data for it
%    3. pivQuiver(pivData,'XYrange',[xmin,xmax,ymin,ymax],whatToPlot1,option1,option1Value,...)
%          choose a part of spacial extent and show data for it
%    4. pivQuiver(pivData,'timeSlice',timeSliceNo,'XYrange',[xmin,xmax,ymin,ymax],whatToPlot1,option1,option1Value,...)
%    5. pivQuiver(pivData,CellWithOptions);
%
% Inputs:
%    pivData ... structure containing results of PIV analysis
%    whatToPlot ... string containing information, what should be shown. Possible values are:
%        'Umag' ... plot background with velocity magnitude
%        'UmagMean' ... plot background with magnitude of an time-average velocity vector
%        'U','V' ... plot background with velocity components
%        'Umean','Vmean' ... plot background with mean values of velocity components
%        'ccPeak' ... plot background with the amplitude of the cross-correlation peak
%        'ccPeakMean' ... plot background with the time-average of amplitude of the cross-correlation peak
%        'ccPeak2nd' ... plot background with the amplitude of the secondary cross-correlation peak
%        'ccPeak2ndMean' ... plot background with the time-average of amplitude of the secondary 
%             cross-correlation peak
%        'ccDetect' ... plot background with the detectability (ccPeak/ccPeakSecondary).
%        'ccDetectMean' ... plot background with the mean detectability (ccPeak/ccPeakSecondary).
%        'ccStd','ccStd1','ccStd2' ... plots background with RMS values in the IA (either average of both
%             images, image1, or image2)
%        'ccMean','ccMean1','ccMean2' ... plots background with mean pixel values in the IA (either average of both
%             images, image1, or image2)
%        'k' ... plot background with turbulence energy k (those should be previously computed using)
%        'RSuu','RSvv','RSuv' ... plot background with Reynolds stress (horizontal normal, vertical normal, or
%             shear)
%        'vort' ... plot background with vorticity (must be computed by pivManipulateData)
%        'epsLEPIV', 'epsMeanLEPIV' ... instantaneous and time-averaged
%             energy dissipation rate, determined by LEPIV (must be computed by pivManipulateData)
%        'image1','image2' ... show image in the background (works only, if images or paths to them are stored
%             in pivData)
%        'imageSup' ... show superposed images in the background (works only, if images or paths to them are stored
%             in pivData)
%        'quiver' ... show quiver plot with the velocity field
%        'invLoc' ... mark spurious vectors with a cross
%    optionX ... string defining, which option will be set. Possible values are:
%        'colorMap' ... colormap for background plot. See help for colormap in Matlab's documentation.
%        'clipLo','clipHi' ... set minimum and maximum value of plotted quantity. If applied to quiver plot,
%            applies to velocity magnitude.
%        'subtractU','subtractV' ... velocity, which is subtracted from the velocity field. Typically, mean
%            velocity can be subtracted in order to visualize unsteady structures superposed on the mean flow.
%        'lineSpec' ... (only for quiver) specify appearance of vectors in the quiver plot. Use standard
%            Matlab specifications as in plot command (e.b. '-k' for black solid line, ':y' for dotted yellow
%            vectors etc).
%        'qScale' ... (only for quiver) define length of vectors. If positive, length of vectors is qScale
%            times displacement. If negative, vector length is (-qScale * autoScale), where autoScale is that
%            one for which 20% of vectors are longer than their distance, and 80% are shorter.
%        'crop' ... limits the range of coordinates, for which the plots are shown. Should be followed by
%            array [xmin,xmax,ymin,ymax]. If X or Y should not be limited, adjust it to -Inf or +Inf. This
%            option should be used prior to "WhatToDraw" arguments, as it applies to drawing commands which
%            follows this option.
%        'selectLo','selectHi' ... (only for quiver) ... only vectors whose length is longer than selectLo,
%            and/or shorter than 'selectHi', will be shown.
%        'selectStat' ... (only for quiver) Only vectors with a specific status will be shown. Possible values
%            for status are
%                'valid' ... only non-replaced vectors will be shown
%                'replaced' ... only replaced vectors will be shown
%                'ccFailed' ... only replaced vectors at positions, where cross-correlation failed, will be
%                    shown
%                'invalid' ... only replaced vectors at positions, in which validation indicated spurious
%                    vectors, will be shown
%        'selectMult' ... (only for quiver) only vectors having a specific multiplicator factors will be shown
%        'vecPos' ... (only for quiver) defines, where arrows should be located. If 0, arrows originate in
%            measurement location (center of interrogation area). If 1, arrows finishin the measurement
%            location. If 0.5 (default), arrows midpoint are in measurement positions.
%        'timeSlice' ... (applies only to data containing velocity sequences) Select, which time slice will be
%            considered thereinafter
%        'title' ... writes a title to the plot (should be followed by a string with the title)
%    optionXValue ... value of the optional parameter
%
% Examples:
%    pivQuiver(pivData,'Umag','quiver') ... show background with velocity magnitude and plot velocity vectors
%    pivQuiver(pivData,'Umag','clipHi',5,...
%        'quiver','selectStat','valid','lineSpec','-k',...
%        'quiver','selectStat','replaced','lineSpec','-w') ... show velocity magnitude (clipping longer
%            displacement than 5 pixels), and the velocity field (valid vectors by black, replaced by white
%            color)
%    pivQuiver(pivData,'UmagMean',...
%        'timeSlice',1,'quiver','lineSpec','-k',...
%        'timeSlice',10,'quiver','lineSpec','-y',) ... show mean velocity magnitude and two instantaneous
%            velocity fields at different times, by black and yellow vector

%#ok<*CTCH>

% if no desktop, skip
if 'matplotlib' not in sys.modules:
    return
# get input; can be variable, but also a cell structure
inputs = varargin

ki = 0

# initializations
pivSlice = pivData
pivAllSlices = pivData
sliceNo = np.nan
xmin = -np.inf
xmax = np.inf
ymin = -np.inf
ymax = np.inf

while len(inputs) > ki:
    if isinstance(inputs[ki], str):
        command = inputs[ki].lower()
        if command == 'timeslice':
            try:
                sliceNo = inputs[ki + 1]
                ki += 1
                pivSlice = pivManipulateData('readTimeSlice', pivData, sliceNo)
                if np.isinf(xmin) and np.isinf(xmax) and np.isinf(ymin) and np.isinf(ymax):
                    pivSlice = pivManipulateData('limitX', pivSlice, [xmin, xmax])
                    pivSlice = pivManipulateData('limitY', pivSlice, [ymin, ymax])
            except:
                print('Error (pivQuiver.m): Error when selecting time slice.')
                return

        elif command == 'crop':
            try:
                auxRange = inputs[ki + 1]
                xmin = auxRange[0]
                xmax = auxRange[1]
                ymin = auxRange[2]
                ymax = auxRange[3]
                ki += 1
                if not np.isnan(sliceNo):
                    pivSlice = pivManipulateData('readTimeSlice', pivData, sliceNo)
                else:
                    pivSlice = pivData
                pivSlice = pivManipulateData('limitX', pivSlice, [xmin, xmax])
                pivSlice = pivManipulateData('limitY', pivSlice, [ymin, ymax])
                pivAllSlices = pivData
                pivAllSlices = pivManipulateData('limitX', pivAllSlices, [xmin, xmax])
                pivAllSlices = pivManipulateData('limitY', pivAllSlices, [ymin, ymax])
            except:
                print('Error (pivQuiver.m): Error when selecting XY range.')
                return

        elif command in ('umag', 'umagmean', 'u', 'umean', 'v', 'vmean'):
            # default options
            options = {
                'colormap': plt.cm.jet(np.linspace(0, 1, 256)),
                'cliplo': -np.inf,
                'cliphi': +np.inf,
                'subtractu': 0,
                'subtractv': 0
            }
            data = pivSlice
            if command in ('umag', 'u', 'v'):
                options, ki = parseOptions(inputs, ki + 1, options)
                if options['subtractu'] == 'mean':
                    options['subtractu'] = meannan(data['U'])
                if options['subtractv'] == 'mean':
                    options['subtractv'] = meannan(data['V'])
                qu = (data['U'] - options['subtractu'])
                qv = (data['V'] - options['subtractv'])
                xmin = np.min(data['X'])
                xmax = np.max(data['X'])
                ymin = np.min(data['Y'])
                ymax = np.max(data['Y'])
            elif command in ('umagmean', 'umean', 'vmean'):
                options, ki = parseOptions(inputs, ki + 1, options)
                if options['subtractu'] == 'mean':
                    options['subtractu'] = meannan(data['U'])
                if options['subtractv'] == 'mean':
                    options['subtractv'] = meannan(data['V'])
                if 'Umean' in data:
                    qu = (data['Umean'] - options['subtractu'])
                    qv = (data['Vmean'] - options['subtractv'])
                else:
                    qu = (data['U'] - options['subtractu'])
                    qv = (data['V'] - options['subtractv'])
                xmin = np.min(data['X'])
                xmax = np.max(data['X'])
                ymin = np.min(data['Y'])
                ymax = np.max(data['Y'])

            # calculate Umag and clip it
            if len(np.shape(qu)) > 2 and (command == 'umag' or command == 'u' or command == 'v'):
                print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                qu = np.mean(qu, axis=2)
                qv = np.mean(qv, axis=2)
            elif command == 'umagmean' or command == 'umean' or command == 'vmean':
                qu = np.mean(qu, axis=2)
                qv = np.mean(qv, axis=2)

            if command in ('umag', 'umagmean'):
                q = np.sqrt(qu**2 + qv**2)
            elif command in ('u', 'umean'):
                q = qu
            elif command in ('v', 'vmean'):
                q = qv

            q[q < options['cliplo']] = options['cliplo']
            q[q > options['cliphi']] = options['cliphi']
            plt.ioff()
            if options['cliplo'] == -np.inf:
                qMin = np.min(q)
            else:
                qMin = options['cliplo']
            if options['cliphi'] == +np.inf:
                qMax = np.max(q)
            else:
                qMax = options['cliphi']
            xmin = np.min(xmin)
            xmax = np.max(xmax)
            try:
                plt.imshow(q.T, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap=options['colormap'], vmin=qMin, vmax=qMax)
            except:
                print('pivQuiver: Failed when plotting {}'.format(command))
            plt.axis('equal')
            plt.colorbar()
            plt.ion()

        elif command in ('ccpeak', 'ccpeakmean', 'ccpeak2nd', 'ccpeak2ndmean', 'ccdetect', 'ccdetectmean',
                         'ccstd', 'ccstd1', 'ccstd2', 'ccmean', 'ccmean1', 'ccmean2',
                         'k', 'vort', 'epslepiv', 'epsmeanlepiv',
                         'rsuu', 'rsvv', 'rsuv'):
            # default options
            if command == 'vort':
                options = {'colormap': vorticityColorMap()}
            else:
                options = {'colormap': plt.cm.jet(np.linspace(0, 1, 256))}
            options.update({'cliplo': -np.inf, 'cliphi': +np.inf, 'spmincc': -np.inf})
            options, ki = parseOptions(inputs, ki + 1, options)
            # set quantity q, which will be drawn
            try:
                if command == 'ccpeak':
                    data = pivSlice
                    q = data['ccPeak']
                    if len(np.shape(q)) > 2:
                        print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                        q = np.mean(q, axis=2)
                elif command == 'ccpeak2nd':
                    data = pivSlice
                    q = data['ccPeakSecondary']
                    if len(np.shape(q)) > 2:
                        print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                        q = np.mean(q, axis=2)
                elif command == 'ccpeakmean':
                    data = pivAllSlices
                    q = data['ccPeak']
                    q = np.mean(q, axis=2)
                elif command == 'ccpeak2ndmean':
                    data = pivAllSlices
                    q = data['ccPeakSecondary']
                    aux = np.isnan(q)
                    q[aux] = np.nanmean(q)
                    q = np.mean(q, axis=2)
                elif command == 'ccdetect':
                    data = pivSlice
                    q1 = data['ccPeak']
                    q2 = data['ccPeakSecondary']
                    aux = np.isnan(q2)
                    q2[aux] = q1[aux]
                    q = q1 / q2
                    if len(np.shape(q)) > 2:
                        print('pivQuiver: Warning: data contains multiple time slices. Plotting mean value.')
                        q = np.mean(q, axis=2)
                elif command == 'ccdetectmean':
                    data = pivAllSlices
                    q1 = data['ccPeak']
                    q2 = data['ccPeakSecondary']
                    aux = np.isnan(q2)
                    q2[aux] = np.nanmean(q2)
                    q = q1 / q2
                    q = np.mean(q, axis=2)
                elif command == 'ccstd':
                    data = pivAllSlices
                    q = np.sqrt(data['ccStd1'] * data['ccStd2'])
                    q = np.mean(q, axis=2)
                elif command == 'ccstd1':
                    data = pivAllSlices
                    q = data['ccStd1']
                    q = np.mean(q, axis=2)
                elif command == 'ccstd2':
                    data = pivAllSlices
                    q = data['ccStd2']
                    q = np.mean(q, axis=2)
                elif command == 'ccmean':
                    data = pivAllSlices
                    q = 1/2 * (data['ccMean1'] + data['ccMean2'])
                    q = np.mean(q, axis=2)
                elif command == 'ccmean1':
                    data = pivAllSlices
                    q = data['ccMean1']
                    q = np.mean(q, axis=2)
                elif command == 'ccmean2':
                    data = pivAllSlices
                    q = data['ccMean2']
                    q = np.mean(q, axis=2)
                elif command == 'k':
                    data = pivAllSlices
                    q = data['k']
                elif command == 'vort':
                    try:
                        data = pivSlice
                        q = data['vorticity']
                    except:
                        data = pivPostprocess('vorticity', pivSlice)
                        q = data['vorticity']
                elif command == 'epslepiv':
                    try:
                        data = pivSlice
                        q = data['epsLEPIV']
                    except:
                        data = pivPostprocess('LEPIVdissip', pivSlice)
                        q = data['epsLEPIV']
                elif command == 'epsmeanlepiv':
                    try:
                        data = pivSlice
                        q = data['epsMeanLEPIV']
                    except:
                        data = pivPostprocess('LEPIVdissip', pivSlice)
                        q = data['epsMeanLEPIV']
                elif command == 'rsuu':
                    data = pivAllSlices
                    q = data['RSuu']
                elif command == 'rsvv':
                    data = pivAllSlices
                    q = data['RSvv']
                elif command == 'rsuv':
                    data = pivAllSlices
                    q = data['RSuv']

                # get min and max of X and Y
                if command in ('ccpeak', 'ccpeak2nd', 'ccpeakmean', 'ccpeak2ndmean', 'ccdetect', 'ccdetectmean',
                                 'ccstd', 'ccstd1', 'ccstd2', 'ccmean', 'ccmean1', 'ccmean2', 'k', 'vort',
                                 'epslepiv', 'epsmeanlepiv', 'rsuu', 'rsvv', 'rsuv'):
                    xmin = np.min(data['X'])
                    xmax = np.max(data['X'])
                    ymin = np.min(data['Y'])
                    ymax = np.max(data['Y'])
            except Exception as e:
                print('Error (pivQuiver.m): Failed to plot desired data ({}).'.format(command))
                print(e)
                return

            # remove complex numbers
            auxNOK = ~np.isreal(q)
            if np.sum(auxNOK) > 0:
                print('WARNING (pivQuiver.m): Complex quantity encountered when plotting {}.'.format(command))
                q = np.abs(q)

            # clip data
            q[q < options['cliplo']] = options['cliplo']
            q[q > options['cliphi']] = options['cliphi']
            plt.ioff()
            if options['cliplo'] == -np.inf:
                qMin = np.min(q)
            else:
                qMin = options['cliplo']
            if options['cliphi'] == +np.inf:
                qMax = np.max(q)
            else:
                qMax = options['cliphi']

            # plot quantity q
            plt.imshow(q.T, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap=options['colormap'], vmin=qMin, vmax=qMax)
            plt.axis('equal')
            plt.colorbar()
            plt.ion()

        elif command in ('image1', 'image2', 'imagesup'):
            # default options
            options = {'colormap': plt.cm.gray(np.linspace(0, 1, 256)), 'expscale': 1}
            options, ki = parseOptions(inputs, ki + 1, options)
            try:
                if command == 'image1':
                    if isinstance(pivSlice['imFilename1'], str):
                        img = plt.imread(pivSlice['imFilename1'])
                    elif isinstance(pivSlice['imFilename1'], dict):
                        img = plt.imread(pivSlice['imFilename1'][0])
                elif command == 'image2':
                    if isinstance(pivSlice['imFilename2'], str):
                        img = plt.imread(pivSlice['imFilename2'])
                    elif isinstance(pivSlice['imFilename2'], dict):
                        img = plt.imread(pivSlice['imFilename2'][0])
                elif command == 'imagesup':
                    if isinstance(pivSlice['imFilename1'], str):
                        img1 = plt.imread(pivSlice['imFilename1'])
                        img2 = plt.imread(pivSlice['imFilename2'])
                    elif isinstance(pivSlice['imFilename1'], dict):
                        img1 = plt.imread(pivSlice['imFilename1'][0])
                        img2 = plt.imread(pivSlice['imFilename2'][0])
                    img = np.maximum(img1, img2)

                if np.isinf(xmin):
                    xmin = 0
                if np.isinf(xmax):
                    xmax = np.shape(img)[1] - 1
                if np.isinf(ymin):
                    ymin = 0
                if np.isinf(ymax):
                    ymax = np.shape(img)[0] - 1

                img = img[int(ymin):int(ymax) + 1, int(xmin):int(xmax) + 1]
                plt.ioff()
                plt.imshow(img, extent=[ymin * options['expscale'], ymax * options['expscale'],
                                        xmin * options['expscale'], xmax * options['expscale']],
                            origin='upper', cmap=options['colormap'])  # Note the change here
                plt.axis('equal')
                plt.colorbar(ticks=[])
                plt.ion()

            except Exception as e:
                print('Warning (pivQuiver.m): Unable to read or display image(s).')
                print(e)

        elif command == 'invloc':
            # show by a marker all locations, where i) crosscorelation failed to find a CC peak, ii) where
            # a velocity was marked as spurious
            options = {'linespec': 'xk', 'markersize': 4}
            options, ki = parseOptions(inputs, ki + 1, options)
            data = pivSlice
            X = []
            Y = []
            # get list of spurious locations
            if 'spuriousX' in data:
                X = data['spuriousX']
                Y = data['spuriousY']
            # get list of CC failure locations
            fail = (data['Status'] & 2) != 0
            X = np.concatenate((X, data['X'][fail])) if len(X) > 0 else data['X'][fail]
            Y = np.concatenate((Y, data['Y'][fail])) if len(Y) > 0 else data['Y'][fail]
            # show results
            plt.plot(X, Y, options['linespec'], markersize=options['markersize'])

        elif command in ('quiver', 'quivermean'):
            # set default option and parse options
            options = {
                'linespec': '-k',
                'qscale': -1.5,
                'qscalepercentile': 0.8,
                'cliplo': -np.inf,
                'cliphi': +np.inf,
                'subtractu': 0,
                'subtractv': 0,
                'selectlo': -np.inf,
                'selecthi': +np.inf,
                'selectstat': 'all',
                'selectmult': 0,
                'vecpos': 0.5,
                'spmincc': -np.inf
            }
            options, ki = parseOptions(inputs, ki + 1, options)
            if options['subtractu'] == 'mean':
                options['subtractu'] = meannan(data['U'])
            if options['subtractv'] == 'mean':
                options['subtractv'] = meannan(data['V'])
            # get data
            if command == 'quiver':
                data = pivSlice
                if 'N' not in data:
                    data['N'] = np.size(data['X'])
                X = np.reshape(data['X'], (data['N'], 1))
                Y = np.reshape(data['Y'], (data['N'], 1))
                U = np.reshape(data['U'], (data['N'], 1)) - options['subtractu']
                V = np.reshape(data['V'], (data['N'], 1)) - options['subtractv']
                S = np.reshape(data['Status'], (data['N'], 1))
                dx = data['iaStepX']
                dy = data['iaStepY']
                xmin = np.min(data['X']) - dx/2
                xmax = np.max(data['X']) + dx/2
                ymin = np.min(data['Y']) - dy/2
                ymax = np.max(data['Y']) + dy/2

            elif command == 'quivermean':
                data = pivAllSlices
                if 'N' not in data:
                    data['N'] = np.size(data['X'])
                X = np.reshape(data['X'], (data['N'], 1))
                Y = np.reshape(data['Y'], (data['N'], 1))
                if 'Umean' in data:
                    U = np.reshape(data['Umean'], (data['N'], 1)) - options['subtractu']
                    V = np.reshape(data['Vmean'], (data['N'], 1)) - options['subtractv']
                else:
                    U = np.reshape(np.mean(data['U'], axis=2), (data['N'], 1)) - options['subtractu']
                    V = np.reshape(np.mean(data['V'], axis=2), (data['N'], 1)) - options['subtractv']
                S = np.zeros((data['N'], 1))
                dx = data['iaStepX']
                dy = data['iaStepY']
                xmin = np.min(data['X']) - dx/2
                xmax = np.max(data['X']) + dx/2
                ymin = np.min(data['Y']) - dy/2
                ymax = np.max(data['Y']) + dy/2
            if 'multiplicator' in data:
                M = np.reshape(data['multiplicator'][:, :, 0], (data['N'], 1))
            else:
                M = np.zeros((data['N'], 1))

            # clip velocity magnitude
            Umag = np.sqrt(U**2 + V**2)
            localScale = np.ones((data['N'], 1))
            aux = Umag > options['cliphi']
            localScale[aux] = options['cliphi'] / Umag[aux]
            aux = Umag < options['cliplo']
            localScale[aux] = options['cliplo'] / Umag[aux]

            # compute the scale
            auxUmag = np.sort(Umag[~np.isnan(Umag)])
            auxUmag_value = auxUmag[int(options['qscalepercentile'] * np.size(auxUmag))]
            if np.isnan(options['qscale']) or options['qscale'] == 0:
                options['qscale'] = 2 * np.sqrt(dx**2 + dy**2) / auxUmag_value
            elif options['qscale'] < 0:
                options['qscale'] = -options['qscale'] * np.sqrt(dx**2 + dy**2) / auxUmag_value

            # select data - Umag
            ok = ((~np.isnan(U)) | (~np.isnan(V))) & \
                 (Umag > options['selectlo']) & (Umag < options['selecthi'])

            # select data - status
            selectstat = options['selectstat'].lower()
            if selectstat == 'all':
                pass
            elif selectstat == 'valid':
                ok = ok & ~((S & (1 << 4)) != 0 | (S & (1 << 7)) != 0)
            elif selectstat == 'replaced':
                ok = ok & ((S & (1 << 4)) != 0 | (S & (1 << 4)) != 0)
            elif selectstat == 'ccfailed':
                ok = ok & ~((S & (1 << 2)) != 0 | (S & (1 << 1)) != 0)
            elif selectstat == 'invalid':
                ok = ok & ((S & (1 << 3)) != 0) | ((S & (1 << 6)) != 0)

            # select data - multiplicator
            if options['selectmult'] != 0:
                ok = ok & (M == options['selectmult'])

            # apply scale
            U = U * localScale * options['qscale']
            V = V * localScale * options['qscale']

            # change vector position
            X = X - options['vecpos'] * U
            Y = Y - options['vecpos'] * V

            # apply selection
            X = X[ok.flatten()]
            Y = Y[ok.flatten()]
            U = U[ok.flatten()]
            V = V[ok.flatten()]

            # select only vectors in frame
            if 'imSizeX' in pivData:
                inFrame = ((X >= xmin) & (Y >= ymin) & (X <= xmax) & (Y <= ymax) &
                           (X + U >= xmin) & (Y + V >= ymin) & (X + U <= xmax) & (Y + V <= ymax))
            else:
                inFrame = np.ones(np.shape(X), dtype=bool)

            # plot quiver
            plt.quiver(X[inFrame], Y[inFrame], U[inFrame], V[inFrame], color=options['linespec'][1:],
                       scale=1, angles='xy', width=0.003, headwidth=2, headlength=3)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gca().invert_yaxis()

        elif command == 'title':
            try:
                titletext = inputs[ki + 1]
                ki += 1
                plt.title(titletext)
            except:
                print('Warning (pivQuiver.m): Title badly specified; ignoring "title" options.')

        else:
            print('Warning (pivQuiver.m): Unable to parse input "{}". Ignoring it.'.format(inputs[ki]))
    else:
        print('Warning (pivQuiver.m): Unable to parse input {}th input. Ignoring it.'.format(ki))
    ki += 1

plt.draw()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def parseOptions(input_list, kin, defaults):
    """
    Extract commands from input list.

    Args:
        input_list (list): List of input arguments.
        kin (int): Starting index.
        defaults (dict): Dictionary of default options.

    Returns:
        tuple: A tuple containing the updated options dictionary and the updated index.
    """
    options = defaults.copy()
    names = list(defaults.keys())
    success = len(input_list) - kin + 1 >= 2
    while success:
        try:
            command = input_list[kin]
            value = input_list[kin + 1]
            success = False
            for jj, name in enumerate(names):
                if command.lower() == name.lower():
                    success = True
                    options[name] = value
                    kin += 2
                    break
        except IndexError:
            success = False
    return options, kin - 1


def vorticityColorMap():
    """
    Create a colormap for vorticity display.

    Returns:
        numpy.ndarray: A colormap array.
    """
    cm = np.ones((256, 3))
    cm[128:, 1] = np.linspace(1, 0, 128)
    cm[128:, 2] = np.linspace(1, 0, 128)
    cm[:128, 0] = np.linspace(0, 1, 128)
    cm[:128, 1] = np.linspace(0, 1, 128)
    return cm


def meannan(arr):
    """
    Compute the mean of an array, ignoring NaN values.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        float: Mean of the array, excluding NaN values.
    """
    arr = np.reshape(arr, -1)
    arr = arr[~np.isnan(arr)]
    return np.mean(arr)
