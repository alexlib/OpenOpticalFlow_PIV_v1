"""
OpenOpticalFlow PIV Package

This package provides tools for Particle Image Velocimetry (PIV) analysis
using optical flow methods.
"""

__version__ = "0.1.0"

# Import main functions to make them available at package level
from .correction_illumination import correction_illumination
from .pre_processing_a import pre_processing_a
from .OpticalFlowPhysics_fun import OpticalFlowPhysics_fun
from .shift_image_fun_refine_1 import shift_image_fun_refine_1
from .horn_schunk_estimator import horn_schunk_estimator
from .liu_shen_estimator import liu_shen_estimator
