import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import json
import scipy.io as sio
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pivSuite.pivCrossCorr import piv_cross_corr
from src.pivSuite.piv_parameters import PIVParameters

def generate_test_data(case_name, size_x=32, size_y=32, pattern='random', noise_level=0.1):
    """
    Generate test data for PIV cross-correlation.

    Parameters:
        case_name: Name of the test case
        size_x, size_y: Size of the interrogation area
        pattern: Type of pattern ('random', 'particles', 'gradient')
        noise_level: Level of noise to add

    Returns:
        Dictionary with test data
    """
    np.random.seed(42)  # For reproducibility

    # Create base images
    if pattern == 'random':
        im1 = np.random.rand(size_y, size_x)
        # Create im2 with a known displacement
        dx, dy = 2, 1  # Displacement in pixels
        im2 = np.roll(np.roll(im1, dx, axis=1), dy, axis=0)
        # Add some noise
        im2 += np.random.normal(0, noise_level, im2.shape)
        im2 = np.clip(im2, 0, 1)
    elif pattern == 'particles':
        # Create particle image with Gaussian particles
        im1 = np.zeros((size_y, size_x))
        num_particles = int(size_x * size_y * 0.1)  # 10% fill
        for _ in range(num_particles):
            x = np.random.randint(0, size_x)
            y = np.random.randint(0, size_y)
            sigma = 1.0
            # Create a Gaussian particle
            y_grid, x_grid = np.mgrid[0:size_y, 0:size_x]
            particle = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            im1 += particle

        # Normalize
        im1 /= np.max(im1)

        # Create im2 with a known displacement
        dx, dy = 2, 1  # Displacement in pixels
        im2 = np.roll(np.roll(im1, dx, axis=1), dy, axis=0)
        # Add some noise
        im2 += np.random.normal(0, noise_level, im2.shape)
        im2 = np.clip(im2, 0, 1)
    elif pattern == 'gradient':
        # Create gradient image
        x = np.linspace(0, 1, size_x)
        y = np.linspace(0, 1, size_y)
        xx, yy = np.meshgrid(x, y)
        im1 = xx * yy

        # Create im2 with a known displacement
        dx, dy = 2, 1  # Displacement in pixels
        im2 = np.roll(np.roll(im1, dx, axis=1), dy, axis=0)
        # Add some noise
        im2 += np.random.normal(0, noise_level, im2.shape)
        im2 = np.clip(im2, 0, 1)

    # Create expanded images (just use the original images for simplicity)
    exIm1 = im1.copy()
    exIm2 = im2.copy()

    # Create pivData
    pivData = {
        'U': np.zeros((1, 1)),
        'V': np.zeros((1, 1)),
        'Status': np.zeros((1, 1), dtype=np.uint16),
        'X': np.array([[size_x / 2]]),
        'Y': np.array([[size_y / 2]]),
        'iaU0': np.zeros((1, 1)),
        'iaV0': np.zeros((1, 1))
    }

    # Create pivPar
    pivPar = PIVParameters()
    pivPar.iaSizeX = size_x
    pivPar.iaSizeY = size_y
    pivPar.ccMethod = 'fft'  # or 'dcn'

    return {
        'case_name': case_name,
        'exIm1': exIm1,
        'exIm2': exIm2,
        'pivData': pivData,
        'pivPar': pivPar,
        'expected_dx': dx,
        'expected_dy': dy
    }

def run_matlab_piv_cross_corr(exIm1, exIm2, pivData, pivPar):
    """
    Run the MATLAB implementation of pivCrossCorr.

    Parameters:
        exIm1, exIm2: Expanded images
        pivData: Dictionary with PIV data
        pivPar: PIV parameters

    Returns:
        Dictionary with results from MATLAB
    """
    # Create temporary directory for MATLAB files
    temp_dir = tempfile.mkdtemp()

    # Save inputs to MAT files
    input_mat_path = os.path.join(temp_dir, 'input.mat')

    # Convert pivPar to a dictionary for MATLAB
    pivPar_dict = {
        'iaSizeX': pivPar.iaSizeX,
        'iaSizeY': pivPar.iaSizeY,
        'ccMethod': pivPar.ccMethod,
        'ccRemoveIAMean': pivPar.ccRemoveIAMean,
        'ccMaxDisplacement': pivPar.ccMaxDisplacement,
        'ccWindow': pivPar.ccWindow,
        'ccCorrectWindowBias': pivPar.ccCorrectWindowBias,
        'ccMaxDCNdist': pivPar.ccMaxDCNdist
    }

    # Save inputs to MAT file
    sio.savemat(input_mat_path, {
        'exIm1': exIm1,
        'exIm2': exIm2,
        'pivData': pivData,
        'pivPar': pivPar_dict
    })

    # Create MATLAB script with the implementation of pivCrossCorr
    matlab_script_path = os.path.join(temp_dir, 'run_piv_cross_corr.m')
    matlab_script = f"""
    % Load input data
    load('{input_mat_path}');

    % Implementation of pivCrossCorr function
    function [pivData,ccPeakIm] = pivCrossCorr(exIm1,exIm2,pivData,pivPar)
        U = pivData.U;
        V = pivData.V;
        status = pivData.Status;
        ccPeak = U;                  % will contain peak levels
        ccPeakSecondary = U;         % will contain level of secondary peaks
        iaSizeX = pivPar.iaSizeX;
        iaSizeY = pivPar.iaSizeY;
        iaNX = size(pivData.X,2);
        iaNY = size(pivData.X,1);
        ccStd1 = pivData.U+NaN;
        ccStd2 = pivData.U+NaN;
        ccMean1 = pivData.U+NaN;
        ccMean2 = pivData.U+NaN;

        % initialize "expanded image" for storing cross-correlations
        ccPeakIm = exIm1 + NaN;   % same size as expanded images

        % peak position is shifted by 1 or 0.5 px, depending on IA size
        if rem(iaSizeX,2) == 0
            ccPxShiftX = 1;
        else
            ccPxShiftX = 0.5;
        end
        if rem(iaSizeY,2) == 0
            ccPxShiftY = 1;
        else
            ccPxShiftY = 0.5;
        end

        %% 1. Create windowing function W and loss-of-correlation function F
        % (ref. [1], Table 8.1, p. 390)
        auxX = ones(iaSizeY,1)*(-(iaSizeX-1)/2:(iaSizeX-1)/2);
        auxY = (-(iaSizeY-1)/2:(iaSizeY-1)/2)'*ones(1,iaSizeX);
        EtaX = auxX/iaSizeX;
        EtaY = auxY/iaSizeY;
        KsiX = 2*EtaX;
        KsiY = 2*EtaY;
        if ~isfield(pivPar,'ccWindow') || strcmpi(pivPar.ccWindow,'uniform')
            W = ones(iaSizeY,iaSizeX);
            F = (1-abs(KsiX)).*(1-abs(KsiY));
        elseif strcmpi(pivPar.ccWindow,'parzen')
            % window function W
            W = (1-2*abs(EtaX)).*(1-2*abs(EtaY));
            % loss of correlation function F
            auxFx = auxX + NaN; % initialization
            auxFy = auxX + NaN;
            auxOK = logical(abs(KsiX)<=1/2);
            auxFx(auxOK) = 1-6*KsiX(auxOK).^2+6*abs(KsiX(auxOK)).^3;
            auxFx(~auxOK) = 2-6*abs(KsiX(~auxOK))+6*KsiX(~auxOK).^2-2*abs(KsiX(~auxOK)).^3;
            auxOK = logical(abs(KsiY)<=1/2);
            auxFy(auxOK) = 1-6*KsiY(auxOK).^2+6*abs(KsiY(auxOK)).^3;
            auxFy(~auxOK) = 2-6*abs(KsiY(~auxOK))+6*KsiY(~auxOK).^2-2*abs(KsiY(~auxOK)).^3;
            F = auxFx.*auxFy;
        elseif strcmpi(pivPar.ccWindow,'Hanning')
            W = (1/2+1/2*cos(2*pi*EtaX)).*(1/2+1/2*cos(2*pi*EtaY));
            F = (2/3*(1-abs(KsiX)).*(1+1/2*cos(2*pi*KsiX))+1/2/pi*sin(2*pi*abs(KsiX))) .*...
                (2/3*(1-abs(KsiY)).*(1+1/2*cos(2*pi*KsiY))+1/2/pi*sin(2*pi*abs(KsiY)));
        elseif strcmpi(pivPar.ccWindow,'Welch')
            W = (1-(2*EtaX).^2).*(1-(2*EtaY).^2);
            F = (1-5*KsiX.^2+5*abs(KsiX).^3-abs(KsiX).^5) .* ...
                (1-5*KsiY.^2+5*abs(KsiY).^3-abs(KsiY).^5);
        elseif strcmpi(pivPar.ccWindow,'Gauss')
            W = exp(-8*EtaX.^2).*exp(-8*EtaY.^2);
            F = exp(-4*KsiX.^2).*exp(-4*KsiY.^2);
        elseif strcmpi(pivPar.ccWindow,'Gauss1')
            W = exp(-8*(EtaX.^2+EtaY.^2)) - exp(-2);
            W(W<0) = 0;
            W = W/max(max(W));
            F = NaN;
        elseif strcmpi(pivPar.ccWindow,'Gauss2')
            W = exp(-16*(EtaX.^2+EtaY.^2)) - exp(-4);
            W(W<0) = 0;
            W = W/max(max(W));
            F = NaN;
        elseif strcmpi(pivPar.ccWindow,'Gauss0.5')
            W = exp(-4*(EtaX.^2+EtaY.^2)) - exp(-1);
            W(W<0) = 0;
            W = W/max(max(W));
            F = NaN;
        elseif strcmpi(pivPar.ccWindow,'Nogueira')
            W = 9*(1-4*abs(EtaX)+4*EtaX.^2).*(1-4*abs(EtaY)+4*EtaY.^2);
            F = NaN;
        elseif strcmpi(pivPar.ccWindow,'Hanning2')
            W = (1/2+1/2*cos(2*pi*EtaX)).^2.*(1/2+1/2*cos(2*pi*EtaY)).^2;
            F = NaN;
        elseif strcmpi(pivPar.ccWindow,'Hanning4')
            W = (1/2+1/2*cos(2*pi*EtaX)).^4.*(1/2+1/2*cos(2*pi*EtaY)).^4;
            F = NaN;
        end
        % Limit F to not be too small
        F(F<0.5) = 0.5;

        %% 2. Cross-correlate expanded images and do subpixel interpolation
        % loop over interrogation areas
        for kx = 1:iaNX
            for ky = 1:iaNY
                failFlag = status(ky,kx);
                % if not masked, get individual interrogation areas from the expanded images
                if failFlag == 0
                    imIA1 = exIm1(1+(ky-1)*iaSizeY:ky*iaSizeY,1+(kx-1)*iaSizeX:kx*iaSizeX);
                    imIA2 = exIm2(1+(ky-1)*iaSizeY:ky*iaSizeY,1+(kx-1)*iaSizeX:kx*iaSizeX);
                    % remove IA mean
                    auxMean1 = mean2(imIA1);
                    auxMean2 = mean2(imIA2);
                    imIA1 = imIA1 - pivPar.ccRemoveIAMean*auxMean1;
                    imIA2 = imIA2 - pivPar.ccRemoveIAMean*auxMean2;
                    % apply windowing function
                    imIA1 = imIA1.*W;
                    imIA2 = imIA2.*W;
                    % compute rms for normalization of cross-correlation
                    auxStd1 = stdfast(imIA1);
                    auxStd2 = stdfast(imIA2);
                    % do the cross-correlation and normalize it
                    switch lower(pivPar.ccMethod)
                        case 'fft'
                            cc = fftshift(real(ifft2(conj(fft2(imIA1)).*fft2(imIA2))))/(auxStd1*auxStd2)/(iaSizeX*iaSizeY);
                            % find the cross-correlation peak
                            [auxPeak,Upx] = max(max(cc));
                            [aux,Vpx] = max(cc(:,Upx));     %#ok<ASGLU>
                        case 'dcn'
                            cc = dcn(imIA1,imIA2,pivPar.ccMaxDCNdist)/(auxStd1*auxStd2)/(iaSizeX*iaSizeY);
                            % find the cross-correlation peak
                            [auxPeak,Upx] = max(max(cc));
                            [aux,Vpx] = max(cc(:,Upx));     %#ok<ASGLU>
                            if (Upx~=iaSizeX/2+ccPxShiftX) || (Vpx~=iaSizeY/2+ccPxShiftY)
                                cc = fftshift(real(ifft2(conj(fft2(imIA1)).*fft2(imIA2))))/(auxStd1*auxStd2)/(iaSizeX*iaSizeY);
                                % find the cross-correlation peak
                                [auxPeak,Upx] = max(max(cc));
                                [aux,Vpx] = max(cc(:,Upx));     %#ok<ASGLU>
                            end
                    end

                    % if the displacement is too large (too close to border), set fail flag
                    if (abs(Upx-iaSizeX/2-ccPxShiftX) > pivPar.ccMaxDisplacement*iaSizeX) || ...
                            (abs(Vpx-iaSizeY/2-ccPxShiftY) > pivPar.ccMaxDisplacement*iaSizeY)
                        failFlag =  bitset(failFlag,2);
                    end
                    % corect cc peak for bias caused by interrogation window (see ref. [1], p. 356, eq. (8.104))
                    if pivPar.ccCorrectWindowBias && ~isnan(F)
                        ccCor = cc./F;
                    else
                        ccCor = cc;
                    end
                       % note: this correction is applied only before finding peak position, otherwise spurious peaks are found at
                       % borders of IA
                    % sub-pixel interpolation (2x3point Gaussian fit, eq. 8.163, p. 375 in [1])
                    try
                        dU = (log(ccCor(Vpx,Upx-1)) - log(ccCor(Vpx,Upx+1)))/...
                            (log(ccCor(Vpx,Upx-1))+log(ccCor(Vpx,Upx+1))-2*log(ccCor(Vpx,Upx)))/2;
                        dV = (log(ccCor(Vpx-1,Upx)) - log(ccCor(Vpx+1,Upx)))/...
                            (log(ccCor(Vpx-1,Upx))+log(ccCor(Vpx+1,Upx))-2*log(ccCor(Vpx,Upx)))/2;
                    catch     %#ok<*CTCH>
                        failFlag = bitset(failFlag,3);
                        dU = NaN; dV = NaN;
                    end
                    % if imaginary, set fail flag
                    if (~isreal(dU)) || (~isreal(dV))
                        failFlag = bitset(failFlag,3);
                    end
                else
                    cc = zeros(iaSizeY,iaSizeX) + NaN;
                    auxPeak = NaN;
                    auxStd1 = NaN;
                    auxStd2 = NaN;
                    auxMean1 = NaN;
                    auxMean2 = NaN;
                    Upx = iaSizeX/2;
                    Vpx = iaSizeY/2;
                end
                % save the pivData information about cross-correlation, rough peak position and peak level
                if failFlag == 0
                    U(ky,kx) = pivData.iaU0(ky,kx) + Upx + dU - iaSizeX/2 - ccPxShiftX;               % this is subroutine's output
                    V(ky,kx) = pivData.iaV0(ky,kx) + Vpx + dV - iaSizeY/2 - ccPxShiftY;               % this is subroutine's output
                else
                    U(ky,kx) = NaN;
                    V(ky,kx) = NaN;
                end
                status(ky,kx) = failFlag;
                ccPeakIm(1+(ky-1)*iaSizeY:ky*iaSizeY,1+(kx-1)*iaSizeX:kx*iaSizeX) = cc;
                ccPeak(ky,kx) = auxPeak;
                ccStd1(ky,kx) = auxStd1;
                ccStd2(ky,kx) = auxStd2;
                ccMean1(ky,kx) = auxMean1;
                ccMean2(ky,kx) = auxMean2;
                % find secondary peak
                try
                    cc(Vpx-2:Vpx+2,Upx-2:Upx+2) = 0;
                    ccPeakSecondary(ky,kx) = max(max(cc));
                catch
                    try
                        cc(Vpx-1:Vpx+1,Upx-1:Upx+1) = 0;
                        ccPeakSecondary(ky,kx) = max(max(cc));
                    catch
                        ccPeakSecondary(ky,kx) = NaN;
                    end
                end % end of secondary peak search
            end % end of loop for ky
        end % end of loop for kx

        % get IAs where CC failed, and coordinates of corresponding IAs
        ccFailedI = logical(bitget(status,2));
        ccSubpxFailedI = logical(bitget(status,3));


        %% 3. Output results
        pivData.Status = uint16(status);
        pivData.U = U;
        pivData.V = V;
        pivData.ccPeak = ccPeak;
        pivData.ccPeakSecondary = ccPeakSecondary;
        pivData.ccStd1 = ccStd1;
        pivData.ccStd2 = ccStd2;
        pivData.ccMean1 = ccMean1;
        pivData.ccMean2 = ccMean2;
        pivData.ccFailedN = sum(sum(ccFailedI));
        pivData.ccSubpxFailedN = sum(sum(ccSubpxFailedI));
        pivData.ccW = W;
        pivData = rmfield(pivData,'iaU0');
        pivData = rmfield(pivData,'iaV0');
    end

    % Implementation of helper functions
    function m = mean2(x)
        m = mean(x(:));
    end

    function [out] = stdfast(in)
        % computes root-mean-square (reprogramed, because std in Matlab is somewhat slow due to some additional tests)
        in = reshape(in,1,numel(in));
        notnan = ~isnan(in);
        n = sum(notnan);
        in(~notnan) = 0;
        avg = sum(in)/n;
        out = sqrt(sum(((in - avg).*notnan).^2)/(n-0)); % there should be -1 in the denominator for true std
    end

    function [cc] = dcn(X1,X2,MaxD)
        % computes cross-correlation using discrete convolution
        Nx = size(X1,2);
        Ny = size(X1,1);
        cc = zeros(Ny,Nx);
        % create variables defining where is cc(0,0)
        dx0 = Nx/2;
        dy0 = Ny/2;
        if rem(Nx,2) == 0
            dx0 = dx0+1;
        else
            dx0 = dx0+0.5;
        end
        if rem(Ny,2) == 0
            dy0 = dy0+1;
        else
            dy0 = dy0+0.5;
        end
        % pad IAs
        X1p = zeros(Ny+2*MaxD,Nx+2*MaxD);
        X2p = zeros(Ny+2*MaxD,Nx+2*MaxD);
        X1p(MaxD+1:MaxD+Ny,MaxD+1:MaxD+Nx) = X1;
        X2p(MaxD+1:MaxD+Ny,MaxD+1:MaxD+Nx) = X2;
        % convolve
        for kx = -MaxD:MaxD
            for ky = -MaxD:MaxD
                if abs(kx)+abs(ky)>MaxD, continue; end
                cc(dy0+ky,dx0+kx) = sum(sum(...
                    X2p(ky+MaxD+1 : ky+MaxD+Ny,  kx+MaxD+1 : kx+MaxD+Nx) .* ...
                    X1p(   MaxD+1 : MaxD+Ny,        MaxD+1 : MaxD+Nx)));
            end
        end
    end

    % Run pivCrossCorr
    [pivData_out, ccPeakIm] = pivCrossCorr(exIm1, exIm2, pivData, pivPar);

    % Save results
    save('{os.path.join(temp_dir, 'output.mat')}', 'pivData_out', 'ccPeakIm');

    % Exit MATLAB
    exit;
    """

    with open(matlab_script_path, 'w') as f:
        f.write(matlab_script)

    # Run MATLAB script
    try:
        subprocess.run(['matlab', '-nodisplay', '-nosplash', '-nodesktop', '-r', f"run('{matlab_script_path}')"],
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"MATLAB execution failed: {e}")
        print(f"STDOUT: {e.stdout.decode('utf-8')}")
        print(f"STDERR: {e.stderr.decode('utf-8')}")
        return None

    # Load results
    output_mat_path = os.path.join(temp_dir, 'output.mat')
    if not os.path.exists(output_mat_path):
        print(f"MATLAB output file not found: {output_mat_path}")
        return None

    matlab_results = sio.loadmat(output_mat_path)

    # Clean up temporary files
    os.remove(input_mat_path)
    os.remove(matlab_script_path)
    os.remove(output_mat_path)
    os.rmdir(temp_dir)

    return {
        'pivData': matlab_results['pivData_out'],
        'ccPeakIm': matlab_results['ccPeakIm']
    }

def compare_results(matlab_results, python_results, test_case=None):
    """
    Compare results from MATLAB and Python implementations.

    Parameters:
        matlab_results: Results from MATLAB
        python_results: Results from Python
        test_case: Test case information (optional)

    Returns:
        Dictionary with comparison results
    """
    # Extract key values for comparison
    matlab_U = matlab_results['pivData']['U'][0, 0]
    matlab_V = matlab_results['pivData']['V'][0, 0]
    python_U = python_results[0]['U'][0, 0]
    python_V = python_results[0]['V'][0, 0]

    # Calculate differences
    U_diff = matlab_U - python_U
    V_diff = matlab_V - python_V

    # Calculate relative errors
    U_rel_error = abs(U_diff / matlab_U) if matlab_U != 0 else abs(U_diff)
    V_rel_error = abs(V_diff / matlab_V) if matlab_V != 0 else abs(V_diff)

    # Check if results are close
    U_is_close = abs(U_diff) < 0.1  # Tolerance of 0.1 pixels
    V_is_close = abs(V_diff) < 0.1  # Tolerance of 0.1 pixels

    return {
        'matlab_U': matlab_U,
        'matlab_V': matlab_V,
        'python_U': python_U,
        'python_V': python_V,
        'U_diff': U_diff,
        'V_diff': V_diff,
        'U_rel_error': U_rel_error,
        'V_rel_error': V_rel_error,
        'U_is_close': U_is_close,
        'V_is_close': V_is_close,
        'is_close': U_is_close and V_is_close
    }

def run_test_case(test_case):
    """
    Run a test case and compare MATLAB and Python results.

    Parameters:
        test_case: Dictionary with test case data

    Returns:
        Dictionary with test results
    """
    print(f"Running test case: {test_case['case_name']}")

    # Run Python implementation
    python_results = piv_cross_corr(
        test_case['exIm1'],
        test_case['exIm2'],
        test_case['pivData'],
        test_case['pivPar']
    )

    # Run MATLAB implementation
    matlab_results = run_matlab_piv_cross_corr(
        test_case['exIm1'],
        test_case['exIm2'],
        test_case['pivData'],
        test_case['pivPar']
    )

    if matlab_results is None:
        print(f"❌ {test_case['case_name']}: MATLAB execution failed")
        return None

    # Compare results
    comparison = compare_results(matlab_results, python_results, test_case)

    # Print results
    if comparison['is_close']:
        print(f"✅ {test_case['case_name']}: MATLAB and Python results are close")
    else:
        print(f"❌ {test_case['case_name']}: MATLAB and Python results differ")
        print(f"  MATLAB U: {comparison['matlab_U']}, V: {comparison['matlab_V']}")
        print(f"  Python U: {comparison['python_U']}, V: {comparison['python_V']}")
        print(f"  Difference U: {comparison['U_diff']}, V: {comparison['V_diff']}")
        print(f"  Relative Error U: {comparison['U_rel_error']}, V: {comparison['V_rel_error']}")

    return {
        'test_case': test_case,
        'python_results': python_results,
        'matlab_results': matlab_results,
        'comparison': comparison
    }

def test_piv_cross_corr():
    """
    Test the piv_cross_corr function against the MATLAB implementation.
    """
    # Generate test cases
    test_cases = [
        generate_test_data("Random pattern, FFT method", pattern='random', size_x=32, size_y=32),
        generate_test_data("Particle pattern, FFT method", pattern='particles', size_x=32, size_y=32),
        generate_test_data("Gradient pattern, FFT method", pattern='gradient', size_x=32, size_y=32),
    ]

    # Add DCN method test cases
    for case in test_cases.copy():
        dcn_case = case.copy()
        dcn_case['case_name'] = case['case_name'].replace('FFT', 'DCN')
        # Create a new PIVParameters object with the same values
        dcn_case['pivPar'] = PIVParameters()
        dcn_case['pivPar'].iaSizeX = case['pivPar'].iaSizeX
        dcn_case['pivPar'].iaSizeY = case['pivPar'].iaSizeY
        dcn_case['pivPar'].ccMethod = 'dcn'
        test_cases.append(dcn_case)

    # Add window function test cases
    window_functions = ['uniform', 'parzen', 'hanning', 'welch', 'gauss']
    for window in window_functions:
        case = generate_test_data(f"Random pattern, {window.capitalize()} window", pattern='random')
        # Set the window function
        case['pivPar'].ccWindow = window
        test_cases.append(case)

    # Run test cases
    results = []
    for case in test_cases:
        result = run_test_case(case)
        if result is not None:
            results.append(result)

    # Summarize results
    success_count = sum(1 for r in results if r['comparison']['is_close'])
    total_count = len(results)

    print(f"\nSummary: {success_count}/{total_count} test cases passed")

    if success_count < total_count:
        print("\nFailed test cases:")
        for result in results:
            if not result['comparison']['is_close']:
                print(f"  - {result['test_case']['case_name']}")
                print(f"    MATLAB U: {result['comparison']['matlab_U']}, V: {result['comparison']['matlab_V']}")
                print(f"    Python U: {result['comparison']['python_U']}, V: {result['comparison']['python_V']}")
                print(f"    Difference U: {result['comparison']['U_diff']}, V: {result['comparison']['V_diff']}")

    return results

if __name__ == "__main__":
    test_piv_cross_corr()
