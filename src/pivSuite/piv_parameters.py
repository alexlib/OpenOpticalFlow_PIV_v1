from dataclasses import dataclass, field
from typing import List, Union, Optional, Any, Dict, Tuple
import numpy as np

@dataclass
class PIVParameters:
    """Parameters for PIV analysis."""
    
    # Interrogation area parameters
    iaSizeX: Union[List[int], int] = field(default_factory=lambda: [64, 32, 16, 16])
    iaSizeY: Union[List[int], int] = field(default_factory=lambda: [64, 32, 16, 16])
    iaStepX: Union[List[int], int] = None  # Will be set in post_init
    iaStepY: Union[List[int], int] = None  # Will be set in post_init
    iaMethod: str = 'defspline'
    iaImageToDeform: str = 'image1'
    iaImageInterpolationMethod: str = 'spline'
    iaPreprocMethod: str = 'none'
    iaMinMaxSize: int = 7
    iaMinMaxLevel: int = 16
    
    # Cross-correlation parameters
    ccRemoveIAMean: float = 1.0
    ccMaxDisplacement: float = 0.9
    ccWindow: str = 'Welch'
    ccCorrectWindowBias: bool = False
    ccMethod: Union[List[str], str] = None  # Will be set in post_init
    ccMaxDCNdist: int = 1
    
    # Validation parameters
    vlMinCC: float = 0.3
    vlTresh: float = 2.0
    vlEps: float = 0.1
    vlPasses: Union[List[int], int] = None  # Will be set in post_init
    vlDist: Union[List[int], int] = None  # Will be set in post_init
    vlDistTSeq: int = 0
    vlTreshSeq: float = 2.0
    vlEpsSeq: float = 0.1
    vlPassesSeq: int = 1
    vlDistSeq: int = 2
    
    # Smoothing parameters
    smMethod: Union[List[str], str] = 'smoothn'
    smSigma: float = np.nan
    smSize: int = 5
    smMethodSeq: str = 'none'
    smSigmaSeq: float = 1.0
    
    # Replacement parameters
    rpMethod: str = 'inpaint'
    
    # Corrector parameters
    crAmount: float = 0.0
    
    # Analysis parameters
    anNpasses: int = 4
    anVelocityEst: str = 'previous'
    anOnDrive: bool = False
    anTargetPath: str = ''
    anForceProcessing: bool = False
    anPairsOnly: bool = False
    anStatsOnly: bool = False
    
    # Sequence parameters
    seqMaxPairs: float = np.inf
    seqFirstIm: int = 1
    seqDiff: int = 1
    seqPairInterval: int = 1
    
    # Masking parameters
    imMask1: Any = None
    imMask2: Any = None
    
    # Visualization parameters
    qvPair: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Set derived parameters after initialization."""
        # Convert scalar values to lists if needed
        self._ensure_list_parameters()
        
        # Set iaStepX and iaStepY if not provided
        if self.iaStepX is None:
            self.iaStepX = [s // 2 for s in self.iaSizeX]
        if self.iaStepY is None:
            self.iaStepY = [s // 2 for s in self.iaSizeY]
            
        # Set ccMethod if not provided
        if self.ccMethod is None:
            self.ccMethod = ['fft'] + ['dcn'] * (self.anNpasses - 1)
            
        # Set vlPasses if not provided
        if self.vlPasses is None:
            if self.anNpasses == 1:
                self.vlPasses = [2]
            else:
                self.vlPasses = [2] + [1] * (self.anNpasses - 2) + [2]
            
        # Set vlDist if not provided
        if self.vlDist is None:
            self.vlDist = [1] + [2] * (self.anNpasses - 1)
    
    def _ensure_list_parameters(self):
        """Ensure that parameters that should be lists are lists."""
        list_params = ['iaSizeX', 'iaSizeY', 'iaStepX', 'iaStepY', 
                       'ccMethod', 'vlPasses', 'vlDist', 'smMethod']
        
        for param in list_params:
            value = getattr(self, param)
            if value is not None and not isinstance(value, list):
                setattr(self, param, [value])
    
    @classmethod
    def create_single_pass(cls, **kwargs) -> 'PIVParameters':
        """Create a PIVParameters object for a single pass."""
        # Set default anNpasses to 1
        if 'anNpasses' not in kwargs:
            kwargs['anNpasses'] = 1
            
        # Ensure list parameters are single values
        for param in ['iaSizeX', 'iaSizeY', 'iaStepX', 'iaStepY', 
                      'ccMethod', 'vlPasses', 'vlDist', 'smMethod']:
            if param in kwargs and isinstance(kwargs[param], list):
                kwargs[param] = kwargs[param][0]
                
        return cls(**kwargs)
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'PIVParameters':
        """Create a PIVParameters object from a dictionary."""
        if params_dict is None:
            return cls()
            
        # Filter out keys that are not valid parameters
        valid_params = {}
        for k, v in params_dict.items():
            if k in cls.__annotations__ or hasattr(cls, k):
                valid_params[k] = v
        return cls(**valid_params)
    
    @classmethod
    def from_tuple_or_dict(cls, params: Any) -> 'PIVParameters':
        """Create a PIVParameters object from a tuple, dict, or existing PIVParameters."""
        if isinstance(params, PIVParameters):
            return params
        elif isinstance(params, dict):
            return cls.from_dict(params)
        elif isinstance(params, tuple) and len(params) > 0:
            if isinstance(params[0], dict):
                return cls.from_dict(params[0])
            elif isinstance(params[0], PIVParameters):
                return params[0]
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameters to a dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    def get_single_pass_params(self, pass_index: int) -> 'PIVParameters':
        """Get parameters for a specific pass."""
        params = self.to_dict()
        single_pass_params = {}
        
        # Handle list parameters
        for key, value in params.items():
            if isinstance(value, list) and len(value) > pass_index:
                single_pass_params[key] = value[pass_index]
            elif isinstance(value, list) and len(value) > 0:
                single_pass_params[key] = value[-1]
            else:
                single_pass_params[key] = value
        
        # Set anNpasses to 1 for single pass
        single_pass_params['anNpasses'] = 1
                
        return PIVParameters.from_dict(single_pass_params)
    
    def get_parameter(self, name: str, pass_index: int = 0) -> Any:
        """Get a parameter value, handling list parameters appropriately."""
        value = getattr(self, name, None)
        if value is None:
            return None
            
        if isinstance(value, list):
            if pass_index < len(value):
                return value[pass_index]
            elif len(value) > 0:
                return value[-1]
            else:
                return None
        return value
    
    def __getitem__(self, key):
        """Allow dictionary-like access to parameters for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return None
    
    def __setitem__(self, key, value):
        """Allow dictionary-like setting of parameters for backward compatibility."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def get(self, key, default=None):
        """Mimic dictionary get method for backward compatibility."""
        if hasattr(self, key):
            value = getattr(self, key)
            if value is not None:
                return value
        return default
