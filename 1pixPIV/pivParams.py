import numpy as np
from typing import Dict, Any, Tuple, List

def piv_params(piv_data: Dict[str, Any], piv_par_in: Dict[str, Any], action: str, *args: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
    piv_par = piv_par_in.copy()

    if action.lower() == 'defaults':
        # Set default values for unspecified fields in pivParIn
        if 'anNpasses' not in piv_par:
            piv_par['anNpasses'] = 4
        if 'iaSizeX' not in piv_par:
            piv_par['iaSizeX'] = [64, 32, 16, 16]
        if 'iaSizeY' not in piv_par:
            piv_par['iaSizeY'] = [64, 32, 16, 16]
        if 'iaStepX' not in piv_par:
            piv_par['iaStepX'] = [s // 2 for s in piv_par['iaSizeX']]
        if 'iaStepY' not in piv_par:
            piv_par['iaStepY'] = [s // 2 for s in piv_par['iaSizeY']]

        piv_par = chkfield(piv_par, 'iaMethod', 'defspline')
        piv_par = chkfield(piv_par, 'imMask1', [])
        piv_par = chkfield(piv_par, 'imMask2', [])
        piv_par = chkfield(piv_par, 'iaImageToDeform', 'image1')
        piv_par = chkfield(piv_par, 'iaImageInterpolationMethod', 'spline')
        piv_par = chkfield(piv_par, 'iaPreprocMethod', 'none')
        piv_par = chkfield(piv_par, 'ccRemoveIAMean', 1)
        piv_par = chkfield(piv_par, 'ccMaxDisplacement', 0.9)
        piv_par = chkfield(piv_par, 'ccWindow', 'Welch')
        piv_par = chkfield(piv_par, 'ccCorrectWindowBias', False)
        piv_par = chkfield(piv_par, 'ccMaxDCNdist', 1)
        piv_par = chkfield(piv_par, 'crAmount', 0)
        piv_par = chkfield(piv_par, 'vlMinCC', 0.3)
        piv_par = chkfield(piv_par, 'vlTresh', 2)
        piv_par = chkfield(piv_par, 'vlEps', 0.1)
        piv_par = chkfield(piv_par, 'vlPasses', [2, 1, 1])
        piv_par = chkfield(piv_par, 'vlDist', [1] + [2] * (piv_par['anNpasses'] - 1))
        piv_par = chkfield(piv_par, 'smMethod', 'smoothn')
        piv_par = chkfield(piv_par, 'smSigma', np.nan)
        piv_par = chkfield(piv_par, 'rpMethod', 'inpaint')
        piv_par = chkfield(piv_par, 'qvPair', {})

        if piv_par['smMethod'] == 'gauss':
            piv_par = chkfield(piv_par, 'smSize', 5)

        piv_par['ccMethod'] = ['fft'] + ['dcn'] * (piv_par['anNpasses'] - 1)
        piv_par['vlPasses'] = [2] + [1] * (piv_par['anNpasses'] - 1)
        piv_par['vlDist'] = [1] + [2] * (piv_par['anNpasses'] - 1)

        if piv_par['iaPreprocMethod'] == 'MinMax':
            piv_par = chkfield(piv_par, 'iaMinMaxSize', 7)
            piv_par = chkfield(piv_par, 'iaMinMaxLevel', 16)

        piv_par = order_fields(piv_par)

    elif action.lower() == 'defaultsseq':
        piv_par = chkfield(piv_par, 'vlDistTSeq', 0)
        piv_par = chkfield(piv_par, 'vlTreshSeq', 2)
        piv_par = chkfield(piv_par, 'vlEpsSeq', 0.1)
        piv_par = chkfield(piv_par, 'vlPassesSeq', 1)
        piv_par = chkfield(piv_par, 'vlDistSeq', 2)
        piv_par = chkfield(piv_par, 'smMethodSeq', 'none')
        piv_par = chkfield(piv_par, 'smSigmaSeq', 1)
        piv_par = chkfield(piv_par, 'seqMaxPairs', np.inf)
        piv_par = chkfield(piv_par, 'seqFirstIm', 1)
        piv_par = chkfield(piv_par, 'seqDiff', 1)
        piv_par = chkfield(piv_par, 'seqPairInterval', 1)
        piv_par = chkfield(piv_par, 'anOnDrive', False)
        piv_par = chkfield(piv_par, 'anTargetPath', '')
        piv_par = chkfield(piv_par, 'anForceProcessing', False)
        piv_par = chkfield(piv_par, 'anVelocityEst', 'previous')
        piv_par = chkfield(piv_par, 'anPairsOnly', False)
        piv_par = chkfield(piv_par, 'anStatsOnly', False)

        if piv_par['anVelocityEst'] in ['previous', 'previousSmooth']:
            piv_par = chkfield(piv_par, 'ccMethod', 'dcn')

        piv_par, piv_data = piv_params(piv_data, piv_par, 'Defaults')
        piv_par = order_fields(piv_par)

    elif action.lower() == 'defaults1px':
        piv_par, piv_data = piv_params(piv_data, piv_par_in, 'Defaults')

        piv_par = chkfield(piv_par, 'spDeltaXNeg', 8)
        piv_par = chkfield(piv_par, 'spDeltaXPos', 8)
        piv_par = chkfield(piv_par, 'spDeltaYNeg', 8)
        piv_par = chkfield(piv_par, 'spDeltaYPos', 8)
        piv_par = chkfield(piv_par, 'spDeltaAutoCorr', 3)
        piv_par = chkfield(piv_par, 'spBindX', 1)
        piv_par = chkfield(piv_par, 'spBindY', 1)
        piv_par = chkfield(piv_par, 'spStepX', min(piv_par['spBindX'], piv_par['spBindY']))
        piv_par = chkfield(piv_par, 'spStepY', min(piv_par['spBindX'], piv_par['spBindY']))
        piv_par = chkfield(piv_par, 'spAvgSmooth', 3)
        piv_par = chkfield(piv_par, 'spRmsSmooth', 5)
        piv_par = chkfield(piv_par, 'spACsource', 'both')
        piv_par = chkfield(piv_par, 'spOnDrive', True)
        piv_par = chkfield(piv_par, 'spForceProcessing', False)
        piv_par = chkfield(piv_par, 'spSaveInterval', 50)
        piv_par = chkfield(piv_par, 'spGFitNPasses', 1)
        piv_par = chkfield(piv_par, 'spGFitMaxIter', 10000)

        piv_par = chkfield(piv_par, 'spGFitMinCc', 0.05 * np.ones(piv_par['spGFitNPasses']))
        piv_par = chkfield(piv_par, 'spGFitMaxDist', 5 * np.ones(piv_par['spGFitNPasses']))
        piv_par = chkfield(piv_par, 'spVlDist', 2 * np.ones(piv_par['spGFitNPasses']))
        piv_par = chkfield(piv_par, 'spVlPasses', 2 * np.ones(piv_par['spGFitNPasses']))
        piv_par = chkfield(piv_par, 'spVlTresh', 1.8 * np.ones(piv_par['spGFitNPasses']))
        piv_par = chkfield(piv_par, 'spVlEps', 0.05 * np.ones(piv_par['spGFitNPasses']))
        piv_par = chkfield(piv_par, 'spSmSigma', np.append(2 * np.ones(piv_par['spGFitNPasses'] - 1), np.nan))

        piv_par = chkfield(piv_par, 'seqFirstIm', 1)
        piv_par = chkfield(piv_par, 'seqDiff', 1)
        piv_par = chkfield(piv_par, 'seqPairInterval', 1)

        piv_par = order_fields(piv_par)

    elif action.lower() == 'singlepass':
        kpass = args[0]
        piv_par = {}

        piv_par = copy_value(piv_par, piv_par_in, 'anNpasses', kpass, 0)
        piv_par = copy_value(piv_par, piv_par_in, 'iaSizeX', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaSizeY', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaStepX', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaStepY', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'imMask1', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'imMask2', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaMethod', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaImageToDeform', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaImageInterpolationMethod', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaPreprocMethod', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaMinMaxSize', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'iaMinMaxLevel', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'ccRemoveIAMean', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'ccMaxDisplacement', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'ccWindow', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'ccCorrectWindowBias', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'ccMethod', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'ccMaxDCNdist', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'crAmount', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'vlMinCC', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'vlTresh', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'vlEps', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'vlDist', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'vlPasses', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'vlDistTSeq', kpass, 1)
        piv_par = copy_value(piv_par, piv_par_in, 'vlTreshSeq', kpass, 1)
        piv_par = copy_value(piv_par, piv_par_in, 'vlEpsSeq', kpass, 1)
        piv_par = copy_value(piv_par, piv_par_in, 'vlDistSeq', kpass, 1)
        piv_par = copy_value(piv_par, piv_par_in, 'vlPassesSeq', kpass, 1)
        piv_par = copy_value(piv_par, piv_par_in, 'rpMethod', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'smMethod', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'smSigma', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'smSize', kpass, 2)
        piv_par = copy_value(piv_par, piv_par_in, 'smMethodSeq', kpass, 1)
        piv_par = copy_value(piv_par, piv_par_in, 'smSigmaSeq', kpass, 1)
        piv_par = copy_value(piv_par, piv_par_in, 'qvPair', kpass, 1)

        piv_par = order_fields(piv_par)

        if len(args) > 1:
            piv_data['pivPar'] = [{} for _ in range(piv_par['anNpasses'])]
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'iaSizeX', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'iaSizeY', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'iaStepX', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'iaStepY', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'ccRemoveIAMean', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'ccMaxDisplacement', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'iaMethod', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'iaImageToDeform', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'iaImageInterpolationMethod', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'rpMethod', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'smMethod', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'smSigma', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'smSize', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'smMethodSeq', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'smSigmaSeq', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlTresh', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlEps', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlDist', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlDistTSeq', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlPasses', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlTreshSeq', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlEpsSeq', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlDistSeq', kpass, 1)
            piv_data['pivPar'][kpass] = copy_value(piv_data['pivPar'][kpass], piv_par, 'vlPassesSeq', kpass, 1)

    elif action.lower() == 'defaultsjobmanagement':
        piv_par = chkfield(piv_par, 'jmParallelJobs', 4)
        piv_par = chkfield(piv_par, 'jmLockExpirationTime', 600)
        piv_par = order_fields(piv_par)

    piv_par = order_fields(piv_par)
    return piv_par, piv_data

def chkfield(piv_par: Dict[str, Any], field_name: str, default_val: Any) -> Dict[str, Any]:
    """
    Check if a field exists in pivPar; if not, set it to the default value.

    Args:
        piv_par (Dict[str, Any]): The parameter dictionary.
        field_name (str): The name of the field to check.
        default_val (Any): The default value to set if the field does not exist.

    Returns:
        Dict[str, Any]: The updated parameter dictionary.
    """
    if field_name not in piv_par:
        piv_par[field_name] = default_val
    return piv_par

def copy_value(piv_par: Dict[str, Any], piv_par_in: Dict[str, Any], field_name: str, k: int, mode: int) -> Dict[str, Any]:
    """
    Copy a field value from pivParIn to pivPar based on the specified mode.

    Args:
        piv_par (Dict[str, Any]): The target parameter dictionary.
        piv_par_in (Dict[str, Any]): The source parameter dictionary.
        field_name (str): The name of the field to copy.
        k (int): The index for array or list fields.
        mode (int): The copying mode (0: do nothing, 1: copy entire field, 2: copy k-th element).

    Returns:
        Dict[str, Any]: The updated parameter dictionary.
    """
    if mode == 1 and field_name in piv_par_in:
        piv_par[field_name] = piv_par_in[field_name]
    elif mode == 2 and field_name in piv_par_in:
        value = piv_par_in[field_name]
        if isinstance(value, (list, np.ndarray)) and len(value) > k:
            piv_par[field_name] = value[k]
        elif isinstance(value, (list, np.ndarray)):
            piv_par[field_name] = value[-1]
        else:
            piv_par[field_name] = value
    return piv_par

def order_fields(piv_par: Dict[str, Any]) -> Dict[str, Any]:
    """
    Order the fields in a dictionary alphabetically.

    Args:
        piv_par (Dict[str, Any]): The dictionary to order.

    Returns:
        Dict[str, Any]: The dictionary with fields ordered alphabetically.
    """
    return {key: piv_par[key] for key in sorted(piv_par)}
