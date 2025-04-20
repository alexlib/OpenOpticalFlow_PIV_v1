import numpy as np
from typing import Dict, Any, Tuple, List, Union
from piv_parameters import PIVParameters

def piv_params(piv_data: Dict[str, Any], piv_par_in: Union[Dict[str, Any], PIVParameters, Tuple], action: str, *args: Any) -> Tuple[Union[Dict[str, Any], PIVParameters], Dict[str, Any]]:
    """
    Adjust content of pivPar variable, which controls settings of PIV analysis.

    Args:
        piv_data (Dict[str, Any]): Structure with PIV data. Can be empty.
        piv_par_in (Dict[str, Any]): Structure with parameters for PIV processing.
        action (str): Defines the action to perform.
        *args (Any): Additional arguments for specific actions.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Adjusted pivPar and pivData.
    """
    # Convert input to PIVParameters if it's not already
    if isinstance(piv_par_in, PIVParameters):
        piv_par = piv_par_in
    elif isinstance(piv_par_in, dict):
        piv_par = PIVParameters.from_dict(piv_par_in)
    elif isinstance(piv_par_in, tuple) and len(piv_par_in) > 0:
        if isinstance(piv_par_in[0], dict):
            piv_par = PIVParameters.from_dict(piv_par_in[0])
        elif isinstance(piv_par_in[0], PIVParameters):
            piv_par = piv_par_in[0]
        else:
            piv_par = PIVParameters()
    else:
        piv_par = PIVParameters()

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

        # Call defaults to ensure everything is properly initialized
        piv_par, piv_data = piv_params(piv_data, piv_par, 'Defaults')

    elif action.lower() == 'defaults1px':
        # First get the defaults
        piv_par, piv_data = piv_params(piv_data, piv_par, 'Defaults')

        # Set single-pixel specific parameters
        # These would ideally be in the PIVParameters class,
        # but for now we'll set them here for backward compatibility
        if not hasattr(piv_par, 'spDeltaXNeg') or piv_par.spDeltaXNeg is None:
            piv_par.spDeltaXNeg = 8
        if not hasattr(piv_par, 'spDeltaXPos') or piv_par.spDeltaXPos is None:
            piv_par.spDeltaXPos = 8
        if not hasattr(piv_par, 'spDeltaYNeg') or piv_par.spDeltaYNeg is None:
            piv_par.spDeltaYNeg = 8
        if not hasattr(piv_par, 'spDeltaYPos') or piv_par.spDeltaYPos is None:
            piv_par.spDeltaYPos = 8
        if not hasattr(piv_par, 'spDeltaAutoCorr') or piv_par.spDeltaAutoCorr is None:
            piv_par.spDeltaAutoCorr = 3
        if not hasattr(piv_par, 'spBindX') or piv_par.spBindX is None:
            piv_par.spBindX = 1
        if not hasattr(piv_par, 'spBindY') or piv_par.spBindY is None:
            piv_par.spBindY = 1
        if not hasattr(piv_par, 'spStepX') or piv_par.spStepX is None:
            piv_par.spStepX = min(piv_par.spBindX, piv_par.spBindY)
        if not hasattr(piv_par, 'spStepY') or piv_par.spStepY is None:
            piv_par.spStepY = min(piv_par.spBindX, piv_par.spBindY)
        if not hasattr(piv_par, 'spAvgSmooth') or piv_par.spAvgSmooth is None:
            piv_par.spAvgSmooth = 3
        if not hasattr(piv_par, 'spRmsSmooth') or piv_par.spRmsSmooth is None:
            piv_par.spRmsSmooth = 5
        if not hasattr(piv_par, 'spACsource') or piv_par.spACsource is None:
            piv_par.spACsource = 'both'
        if not hasattr(piv_par, 'spOnDrive') or piv_par.spOnDrive is None:
            piv_par.spOnDrive = True
        if not hasattr(piv_par, 'spForceProcessing') or piv_par.spForceProcessing is None:
            piv_par.spForceProcessing = False
        if not hasattr(piv_par, 'spSaveInterval') or piv_par.spSaveInterval is None:
            piv_par.spSaveInterval = 50
        if not hasattr(piv_par, 'spGFitNPasses') or piv_par.spGFitNPasses is None:
            piv_par.spGFitNPasses = 1
        if not hasattr(piv_par, 'spGFitMaxIter') or piv_par.spGFitMaxIter is None:
            piv_par.spGFitMaxIter = 10000

        # Arrays based on number of passes
        if not hasattr(piv_par, 'spGFitMinCc') or piv_par.spGFitMinCc is None:
            piv_par.spGFitMinCc = 0.05 * np.ones(piv_par.spGFitNPasses)
        if not hasattr(piv_par, 'spGFitMaxDist') or piv_par.spGFitMaxDist is None:
            piv_par.spGFitMaxDist = 5 * np.ones(piv_par.spGFitNPasses)
        if not hasattr(piv_par, 'spVlDist') or piv_par.spVlDist is None:
            piv_par.spVlDist = 2 * np.ones(piv_par.spGFitNPasses)
        if not hasattr(piv_par, 'spVlPasses') or piv_par.spVlPasses is None:
            piv_par.spVlPasses = 2 * np.ones(piv_par.spGFitNPasses)
        if not hasattr(piv_par, 'spVlTresh') or piv_par.spVlTresh is None:
            piv_par.spVlTresh = 1.8 * np.ones(piv_par.spGFitNPasses)
        if not hasattr(piv_par, 'spVlEps') or piv_par.spVlEps is None:
            piv_par.spVlEps = 0.05 * np.ones(piv_par.spGFitNPasses)
        if not hasattr(piv_par, 'spSmSigma') or piv_par.spSmSigma is None:
            piv_par.spSmSigma = np.append(2 * np.ones(piv_par.spGFitNPasses - 1), np.nan)

    elif action.lower() == 'singlepass':
        kpass = args[0]
        # Use the get_single_pass_params method to extract parameters for a specific pass
        piv_par = piv_par.get_single_pass_params(kpass)

        # Store parameters in pivData if needed
        if len(args) > 1:
            if 'pivPar' not in piv_data:
                piv_data['pivPar'] = [PIVParameters() for _ in range(piv_par.anNpasses)]
            piv_data['pivPar'][kpass] = piv_par

    elif action.lower() == 'defaultsjobmanagement':
        # Set job management parameters
        if not hasattr(piv_par, 'jmParallelJobs') or piv_par.jmParallelJobs is None:
            piv_par.jmParallelJobs = 4
        if not hasattr(piv_par, 'jmLockExpirationTime') or piv_par.jmLockExpirationTime is None:
            piv_par.jmLockExpirationTime = 600

    piv_par = order_fields(piv_par)
    return piv_par, piv_data

def chkfield(piv_par: Union[Dict[str, Any], PIVParameters], field_name: str, default_val: Any) -> Union[Dict[str, Any], PIVParameters]:
    """
    Check if a field exists in pivPar; if not, set it to the default value.

    Args:
        piv_par (Union[Dict[str, Any], PIVParameters]): The parameter object.
        field_name (str): The name of the field to check.
        default_val (Any): The default value to set if the field does not exist.

    Returns:
        Union[Dict[str, Any], PIVParameters]: The updated parameter object.
    """
    if isinstance(piv_par, dict):
        if field_name not in piv_par:
            piv_par[field_name] = default_val
    else:  # PIVParameters object
        if not hasattr(piv_par, field_name) or getattr(piv_par, field_name) is None:
            setattr(piv_par, field_name, default_val)
    return piv_par

def copy_value(piv_par: Union[Dict[str, Any], PIVParameters],
               piv_par_in: Union[Dict[str, Any], PIVParameters, Tuple],
               field_name: str, k: int, mode: int) -> Union[Dict[str, Any], PIVParameters]:
    """
    Copy a field value from pivParIn to pivPar based on the specified mode.

    Args:
        piv_par (Union[Dict[str, Any], PIVParameters]): The target parameter object.
        piv_par_in (Union[Dict[str, Any], PIVParameters, Tuple]): The source parameter object.
        field_name (str): The name of the field to copy.
        k (int): The index for array or list fields.
        mode (int): The copying mode (0: do nothing, 1: copy entire field, 2: copy k-th element).

    Returns:
        Union[Dict[str, Any], PIVParameters]: The updated parameter object.
    """
    # Handle different input types
    if isinstance(piv_par_in, tuple) and len(piv_par_in) > 0:
        if isinstance(piv_par_in[0], dict):
            piv_par_in = piv_par_in[0]
        elif isinstance(piv_par_in[0], PIVParameters):
            piv_par_in = piv_par_in[0]

    # Check if field exists in source
    field_exists = False
    if isinstance(piv_par_in, dict):
        field_exists = field_name in piv_par_in
        if field_exists:
            value = piv_par_in[field_name]
    else:  # PIVParameters object
        field_exists = hasattr(piv_par_in, field_name)
        if field_exists:
            value = getattr(piv_par_in, field_name)

    if not field_exists:
        return piv_par

    # Copy value based on mode
    if mode == 1:  # Copy entire field
        if isinstance(piv_par, dict):
            piv_par[field_name] = value
        else:  # PIVParameters object
            setattr(piv_par, field_name, value)
    elif mode == 2:  # Copy k-th element
        if isinstance(value, (list, np.ndarray)):
            if len(value) > k:
                result_value = value[k]
            elif len(value) > 0:
                result_value = value[-1]
            else:
                return piv_par
        else:
            result_value = value

        if isinstance(piv_par, dict):
            piv_par[field_name] = result_value
        else:  # PIVParameters object
            setattr(piv_par, field_name, result_value)

    return piv_par

def order_fields(piv_par: Union[Dict[str, Any], PIVParameters]) -> Union[Dict[str, Any], PIVParameters]:
    """
    Order the fields in a dictionary or ensure PIVParameters object is properly initialized.

    Args:
        piv_par (Union[Dict[str, Any], PIVParameters]): The parameter object.

    Returns:
        Union[Dict[str, Any], PIVParameters]: The parameter object with ordered fields.
    """
    if isinstance(piv_par, dict):
        return {key: piv_par[key] for key in sorted(piv_par)}
    return piv_par  # PIVParameters objects don't need ordering
