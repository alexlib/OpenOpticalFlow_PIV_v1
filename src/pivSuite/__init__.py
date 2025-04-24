"""
PIVsuite Package

This package provides tools for Particle Image Velocimetry (PIV) analysis
using cross-correlation methods.
"""

__version__ = "0.1.0"

# Import main functions to make them available at package level
from .piv_parameters import PIVParameters
from .pivAnalyzeImagePair import piv_analyze_image_pair
