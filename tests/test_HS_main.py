"""
Test script for the Horn-Schunck optical flow implementation
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# Add the project root directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import required functions
from HS_PIV.HS_Pyramids import HS_Pyramids
from HS_PIV.tools.plotFlow_Cai import plotFlow_Cai
from HS_PIV.tools.flowColorCode.readFlowFile import readFlowFile
from HS_PIV.tools.flowAngErr import flowAngErr

# Test data configuration
TEST_IMAGE_PAIRS = [
    'uniform_00001',
    'DNS_turbulence_00001',
    'vortexPair'
]

@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory"""
    return PROJECT_ROOT / 'data'

@pytest.fixture
def test_parameters():
    """Fixture providing default test parameters"""
    return {
        'pyramid_level': 3,
        'warp_iter': 2,
        'ite': 400,
        'boundaryCondition': 'periodical',
        'interpolation_method': 'spline',
        'isMedianFilter': True,
        'sizeOfMF': [5, 5]
    }

@pytest.mark.flow
def test_hs_pyramids_computation(test_data_dir, test_parameters):
    """Test basic Horn-Schunck pyramid computation"""
    # Load test images
    im1_path = test_data_dir / 'uniform_00001_img1.tif'
    im2_path = test_data_dir / 'uniform_00001_img2.tif'
    
    assert im1_path.exists(), f"Test image not found: {im1_path}"
    assert im2_path.exists(), f"Test image not found: {im2_path}"
    
    im1 = io.imread(im1_path)
    im2 = io.imread(im2_path)
    
    # Compute flow
    lambda_param = 10
    u, v = HS_Pyramids(im1, im2, lambda_param, test_parameters)
    
    # Basic assertions
    assert u.shape == im1.shape, "Flow field U shape mismatch"
    assert v.shape == im1.shape, "Flow field V shape mismatch"
    assert not np.isnan(u).any(), "Flow field U contains NaN values"
    assert not np.isnan(v).any(), "Flow field V contains NaN values"

@pytest.mark.flow
def test_hs_pyramids_against_ground_truth(test_data_dir, test_parameters):
    """Test Horn-Schunck results against ground truth"""
    # Load images and ground truth
    im1 = io.imread(test_data_dir / 'uniform_00001_img1.tif')
    im2 = io.imread(test_data_dir / 'uniform_00001_img2.tif')
    uv_gt = readFlowFile(test_data_dir / 'uniform_00001_flow.flo')
    
    # Compute flow
    lambda_param = 10
    u, v = HS_Pyramids(im1, im2, lambda_param, test_parameters)
    
    # Compare with ground truth
    rmse = flowAngErr(uv_gt[:,:,0], uv_gt[:,:,1], u, v, margin=0)
    
    # Assert RMSE is within acceptable range
    assert rmse < 1.0, f"RMSE too high: {rmse:.3f}"

@pytest.mark.visual
def test_flow_visualization(test_data_dir, test_parameters, tmp_path):
    """Test flow field visualization"""
    # Load images
    im1 = io.imread(test_data_dir / 'uniform_00001_img1.tif')
    im2 = io.imread(test_data_dir / 'uniform_00001_img2.tif')
    
    # Compute flow
    lambda_param = 10
    u, v = HS_Pyramids(im1, im2, lambda_param, test_parameters)
    
    # Generate visualization
    plt.figure(figsize=(10, 8))
    plotFlow_Cai(u, v, None, 0.5)
    output_path = tmp_path / 'test_output_flow.png'
    plt.savefig(output_path)
    plt.close()
    
    assert output_path.exists(), "Flow visualization was not saved"

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
