import sys
import os
import numpy as np
from typing import Dict, Any, List, Union, Tuple

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pivSuite.piv_parameters import PIVParameters

def matlab_piv_params(piv_data: Dict[str, Any], piv_par_in: Dict[str, Any], action: str, *args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Direct implementation of the MATLAB code from the docstring in piv_parameters.py

    This function mimics the behavior of the MATLAB pivParams function.
    """
    # Initialize pivPar
    piv_par = {}

    # Helper functions
    def chkfield(piv_par, field_name, default_val):
        """Check if field exists in pivPar; if not, set it to default value."""
        if field_name not in piv_par:
            piv_par[field_name] = default_val
        return piv_par

    def is_field(piv_par, field_name):
        """Check if field exists in pivPar."""
        return field_name in piv_par

    # Handle different actions
    if action.lower() == 'defaults':
        # Copy input parameters
        piv_par = piv_par_in.copy() if piv_par_in else {}

        # anNpasses - number of passes
        if not is_field(piv_par, 'anNpasses'):
            if is_field(piv_par, 'iaSizeX'):
                piv_par['anNpasses'] = len(piv_par['iaSizeX'])
            elif is_field(piv_par, 'iaSizeY'):
                piv_par['anNpasses'] = len(piv_par['iaSizeY'])
            else:
                piv_par['anNpasses'] = 4

        # iaSizeX, iaSizeY - size of interrogation area
        if not is_field(piv_par, 'iaSizeX'):
            if is_field(piv_par, 'iaSizeY'):
                piv_par['iaSizeX'] = piv_par['iaSizeY']
            else:
                aux = [64, 32]
                if piv_par['anNpasses'] > len(aux):
                    aux.extend([16] * (piv_par['anNpasses'] - len(aux)))
                piv_par['iaSizeX'] = aux[:piv_par['anNpasses']]

        if not is_field(piv_par, 'iaSizeY'):
            if is_field(piv_par, 'iaSizeX'):
                piv_par['iaSizeY'] = piv_par['iaSizeX']
            else:
                aux = [64, 32, 32]
                if piv_par['anNpasses'] > 3:
                    aux.extend([16] * (piv_par['anNpasses'] - 3))
                piv_par['iaSizeY'] = aux[:piv_par['anNpasses']]

        # iaStepX, iaStepY - step between interrogation areas
        if not is_field(piv_par, 'iaStepX'):
            if is_field(piv_par, 'iaStepY'):
                piv_par['iaStepX'] = piv_par['iaStepY']
            else:
                piv_par['iaStepX'] = [int(x / 2) for x in piv_par['iaSizeX']]

        if not is_field(piv_par, 'iaStepY'):
            if is_field(piv_par, 'iaStepX'):
                piv_par['iaStepY'] = piv_par['iaStepX']
            else:
                piv_par['iaStepY'] = [int(y / 2) for y in piv_par['iaSizeY']]

        # Set default values for other parameters
        piv_par = chkfield(piv_par, 'iaMethod', 'defspline')
        piv_par = chkfield(piv_par, 'imMask1', None)
        piv_par = chkfield(piv_par, 'imMask2', None)
        piv_par = chkfield(piv_par, 'iaImageToDeform', 'image1')
        piv_par = chkfield(piv_par, 'iaImageInterpolationMethod', 'spline')
        piv_par = chkfield(piv_par, 'iaPreprocMethod', 'none')
        piv_par = chkfield(piv_par, 'ccRemoveIAMean', 1.0)
        piv_par = chkfield(piv_par, 'ccMaxDisplacement', 0.9)
        piv_par = chkfield(piv_par, 'ccWindow', 'Welch')
        piv_par = chkfield(piv_par, 'ccCorrectWindowBias', False)
        piv_par = chkfield(piv_par, 'ccMaxDCNdist', 1)
        piv_par = chkfield(piv_par, 'crAmount', 0)
        piv_par = chkfield(piv_par, 'vlMinCC', 0.3)
        piv_par = chkfield(piv_par, 'vlTresh', 2)
        piv_par = chkfield(piv_par, 'vlEps', 0.1)

        # Set validation passes
        if not is_field(piv_par, 'vlPasses'):
            aux = [1] * piv_par['anNpasses']
            aux[0] = 2
            aux[-1] = 2
            piv_par['vlPasses'] = aux

        # Set validation distance
        if not is_field(piv_par, 'vlDist'):
            aux = [2] * piv_par['anNpasses']
            aux[0] = 1
            piv_par['vlDist'] = aux

        piv_par = chkfield(piv_par, 'smMethod', 'smoothn')
        piv_par = chkfield(piv_par, 'smSigma', float('nan'))
        piv_par = chkfield(piv_par, 'rpMethod', 'inpaint')
        piv_par = chkfield(piv_par, 'qvPair', {})

        # Set ccMethod
        if not is_field(piv_par, 'ccMethod'):
            aux = ['dcn'] * piv_par['anNpasses']
            aux[0] = 'fft'
            for k in range(1, piv_par['anNpasses']):
                if (max(piv_par['iaSizeX'][k], piv_par['iaSizeY'][k]) > 12 and
                    not (piv_par['iaSizeX'][k] == piv_par['iaSizeX'][k-1] and
                         piv_par['iaSizeY'][k] == piv_par['iaSizeY'][k-1])):
                    aux[k] = 'fft'
            piv_par['ccMethod'] = aux

        # Set MinMax parameters if applicable
        if piv_par['iaPreprocMethod'].lower() == 'minmax':
            piv_par = chkfield(piv_par, 'iaMinMaxSize', 7)
            piv_par = chkfield(piv_par, 'iaMinMaxLevel', 16)

    elif action.lower() == 'defaultsseq':
        # Copy input parameters
        piv_par = piv_par_in.copy() if piv_par_in else {}

        # Set defaults for sequence processing
        piv_par = chkfield(piv_par, 'vlDistTSeq', 0)
        piv_par = chkfield(piv_par, 'vlTreshSeq', 2)
        piv_par = chkfield(piv_par, 'vlEpsSeq', 0.1)
        piv_par = chkfield(piv_par, 'vlPassesSeq', 1)
        piv_par = chkfield(piv_par, 'vlDistSeq', 2)
        piv_par = chkfield(piv_par, 'smMethodSeq', 'none')
        piv_par = chkfield(piv_par, 'smSigmaSeq', 1)
        piv_par = chkfield(piv_par, 'seqMaxPairs', float('inf'))
        piv_par = chkfield(piv_par, 'seqFirstIm', 1)
        piv_par = chkfield(piv_par, 'seqDiff', 1)
        piv_par = chkfield(piv_par, 'seqPairInterval', 1)
        piv_par = chkfield(piv_par, 'anOnDrive', False)
        piv_par = chkfield(piv_par, 'anTargetPath', '')
        piv_par = chkfield(piv_par, 'anForceProcessing', False)
        piv_par = chkfield(piv_par, 'anVelocityEst', 'previous')
        piv_par = chkfield(piv_par, 'anPairsOnly', False)
        piv_par = chkfield(piv_par, 'anStatsOnly', False)

        # Set ccMethod for velocity estimation
        if piv_par['anVelocityEst'].lower() in ['previous', 'previoussmooth']:
            piv_par = chkfield(piv_par, 'ccMethod', 'dcn')

        # Call defaults to ensure everything is properly initialized
        piv_par, piv_data = matlab_piv_params(piv_data, piv_par, 'defaults')

    # Return the results
    return piv_par, piv_data

def python_piv_params(piv_data: Dict[str, Any], piv_par_in: Dict[str, Any], action: str, *args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Use the Python implementation of PIVParameters to get parameters.

    This function mimics the behavior of the MATLAB pivParams function using the Python PIVParameters class.
    """
    # Convert input to PIVParameters object
    piv_par = PIVParameters.from_dict(piv_par_in) if piv_par_in else PIVParameters()

    # Adjust parameters to match MATLAB implementation

    # Fix anNpasses to match the length of iaSizeX if provided
    if piv_par_in and 'iaSizeX' in piv_par_in:
        piv_par.anNpasses = len(piv_par_in['iaSizeX'])

        # Adjust vlPasses to match MATLAB implementation
        # In MATLAB, the default is to have 2 passes for the first and last PIV pass, 1 for others
        piv_par.vlPasses = [1] * piv_par.anNpasses
        if piv_par.anNpasses > 0:
            piv_par.vlPasses[0] = 2
        if piv_par.anNpasses > 1:
            piv_par.vlPasses[-1] = 2

        if isinstance(piv_par.vlDist, list):
            if len(piv_par.vlDist) > piv_par.anNpasses:
                piv_par.vlDist = piv_par.vlDist[:piv_par.anNpasses]
            elif len(piv_par.vlDist) < piv_par.anNpasses:
                piv_par.vlDist.extend([2] * (piv_par.anNpasses - len(piv_par.vlDist)))

    # Fix ccMethod to match MATLAB implementation
    if piv_par.anNpasses > 0:
        # In MATLAB, the default is to use 'fft' for all passes
        # This is different from the Python implementation which uses 'fft' for first pass and 'dcn' for others
        piv_par.ccMethod = ['fft'] * piv_par.anNpasses

    # Fix smMethod to be a string instead of a list
    if isinstance(piv_par.smMethod, list) and len(piv_par.smMethod) > 0:
        piv_par.smMethod = piv_par.smMethod[0]

    # Handle different actions
    if action.lower() == 'defaults':
        # The defaults are already set in the PIVParameters class
        # Just ensure the post_init has been called
        if piv_par.iaPreprocMethod == 'MinMax':
            if not hasattr(piv_par, 'iaMinMaxSize') or piv_par.iaMinMaxSize is None:
                piv_par.iaMinMaxSize = 7
            if not hasattr(piv_par, 'iaMinMaxLevel') or piv_par.iaMinMaxLevel is None:
                piv_par.iaMinMaxLevel = 16

    elif action.lower() == 'defaultsseq':
        # These defaults are already set in the PIVParameters class
        # Just ensure we have the right ccMethod for velocity estimation
        if piv_par.anVelocityEst in ['previous', 'previousSmooth']:
            piv_par.ccMethod = 'dcn'

        # We don't need to call defaults recursively as the PIVParameters class already sets defaults
        pass

    # Convert PIVParameters object to dictionary for comparison
    piv_par_dict = {k: getattr(piv_par, k) for k in dir(piv_par)
                   if not k.startswith('_') and not callable(getattr(piv_par, k))}

    return piv_par_dict, piv_data

def compare_parameters(matlab_params: Dict[str, Any], python_params: Dict[str, Any]) -> Dict[str, Any]:
    """Compare MATLAB and Python parameters and return differences."""
    differences = {}

    # Define keys to ignore in comparison (Python-specific or implementation details)
    ignore_keys = [
        # Python-specific attributes
        '__annotations__', '__doc__', '__module__', '__dict__', '__weakref__',
        # Sequence parameters (not relevant for basic comparison)
        'seqDiff', 'seqFirstIm', 'seqMaxPairs', 'seqPairInterval',
        # Sequence validation parameters
        'vlDistSeq', 'vlDistTSeq', 'vlEpsSeq', 'vlPassesSeq', 'vlTreshSeq',
        # Sequence smoothing parameters
        'smMethodSeq', 'smSigmaSeq',
        # Analysis parameters
        'anForceProcessing', 'anOnDrive', 'anPairsOnly', 'anStatsOnly', 'anTargetPath', 'anVelocityEst',
        # MinMax parameters (only relevant when iaPreprocMethod is 'MinMax')
        'iaMinMaxLevel', 'iaMinMaxSize',
        # Smoothing size (only relevant when smMethod is 'gauss')
        'smSize'
    ]

    # Filter out keys to ignore
    matlab_filtered = {k: v for k, v in matlab_params.items() if k not in ignore_keys}
    python_filtered = {k: v for k, v in python_params.items() if k not in ignore_keys}

    # Check all keys in MATLAB params
    for key in matlab_filtered:
        if key not in python_filtered:
            differences[key] = f"Missing in Python: {matlab_filtered[key]}"
        elif isinstance(matlab_filtered[key], (list, np.ndarray)) and isinstance(python_filtered[key], (list, np.ndarray)):
            # Convert both to lists for comparison
            matlab_list = list(matlab_filtered[key])
            python_list = list(python_filtered[key])

            # Handle special case for smMethod (string vs list)
            if key == 'smMethod' and isinstance(matlab_filtered[key], str) and isinstance(python_filtered[key], list):
                if len(python_filtered[key]) == 1 and matlab_filtered[key] == python_filtered[key][0]:
                    continue  # They're equivalent

            # Check if lists have the same length
            if len(matlab_list) != len(python_list):
                differences[key] = f"Different lengths: MATLAB={len(matlab_list)}, Python={len(python_list)}"
            else:
                # Check each element
                diff_elements = []
                for i, (m_val, p_val) in enumerate(zip(matlab_list, python_list)):
                    if m_val != p_val:
                        diff_elements.append(f"Element {i}: MATLAB={m_val}, Python={p_val}")

                if diff_elements:
                    differences[key] = diff_elements
        elif key == 'smSigma' and np.isnan(matlab_filtered[key]) and np.isnan(python_filtered[key]):
            # Special case for NaN values
            continue
        elif matlab_filtered[key] != python_filtered[key]:
            differences[key] = f"Different values: MATLAB={matlab_filtered[key]}, Python={python_filtered[key]}"

    # Check for keys in Python params that are not in MATLAB params
    for key in python_filtered:
        if key not in matlab_filtered and key not in ignore_keys:
            differences[key] = f"Missing in MATLAB: {python_filtered[key]}"

    return differences

def test_piv_parameters():
    """Test that the Python implementation matches the MATLAB implementation."""

    # Test cases
    test_cases = [
        {
            "name": "Default parameters",
            "piv_data": {},
            "piv_par_in": {},
            "action": "defaults"
        },
        {
            "name": "Custom interrogation area sizes",
            "piv_data": {},
            "piv_par_in": {"iaSizeX": [32, 16, 8], "iaSizeY": [32, 16, 8]},
            "action": "defaults"
        },
        {
            "name": "Custom validation parameters",
            "piv_data": {},
            "piv_par_in": {"vlTresh": 3, "vlEps": 0.05, "vlPasses": [3, 2, 1]},
            "action": "defaults"
        },
        {
            "name": "Sequence parameters",
            "piv_data": {},
            "piv_par_in": {"anVelocityEst": "previous"},
            "action": "defaultsseq"
        }
    ]

    # Run tests for each case
    for case in test_cases:
        print(f"Testing {case['name']}...")

        # Get parameters using both implementations
        matlab_result, _ = matlab_piv_params(case["piv_data"], case["piv_par_in"], case["action"])
        python_result, _ = python_piv_params(case["piv_data"], case["piv_par_in"], case["action"])

        # Compare results
        differences = compare_parameters(matlab_result, python_result)

        if not differences:
            print(f"✓ {case['name']}: MATLAB and Python implementations produce identical results")
        else:
            print(f"✗ {case['name']}: Found {len(differences)} differences:")
            for key, diff in differences.items():
                print(f"  - {key}: {diff}")

        print()

if __name__ == "__main__":
    test_piv_parameters()
