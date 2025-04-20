import numpy as np
from imageio.v3 import imread
from optical_flow_physics import optical_flow_physics
from preprocessing import (pre_processing, correction_illumination,
                         shift_image_refine)
from plots_set_1 import plots_set_1
from plots_set_2 import plots_set_2

def flow_diagnostics_run(image1_path, image2_path, params=None):
    """
    Main function for optical flow analysis of image pairs.
    Matches MATLAB implementation exactly.
    """
    # Default parameters
    if params is None:
        params = {
            'lambda_1': 20,
            'lambda_2': 2000,
            'scale_im': 0.5,
            'size_filter': 4,
            'size_average': 20,
            'no_iteration': 1
        }

    # Read images
    Im1 = imread(image1_path)
    Im2 = imread(image2_path)

    # Convert to double precision float (0-255)
    I1_original = Im1.astype(float)
    I2_original = Im2.astype(float)

    # Correct illumination
    window_shifting = [0, Im1.shape[1], 0, Im1.shape[0]]  # [x1,x2,y1,y2]
    I1, I2 = correction_illumination(I1_original, I2_original,
                                   window_shifting, params['size_average'])

    # Pre-processing
    I1, I2 = pre_processing(I1, I2, params['scale_im'], params['size_filter'])
    I_region1, I_region2 = I1.copy(), I2.copy()

    # Initial optical flow calculation
    ux0, uy0, vor, ux_horn, uy_horn, error1 = optical_flow_physics(
        I_region1, I_region2, params['lambda_1'], params['lambda_2'])

    # Iterative refinement
    ux_corr, uy_corr = ux0.copy(), uy0.copy()

    for k in range(params['no_iteration']):
        # Shift image based on current flow field
        Im1_shift, uxI, uyI = shift_image_refine(ux_corr, uy_corr, Im1, Im2)

        I1 = Im1_shift.astype(float)
        I2 = Im2.astype(float)

        # Calculate correction
        dux, duy, vor, _, _, error2 = optical_flow_physics(
            I1, I2, params['lambda_1'], params['lambda_2'])

        # Update flow field
        ux_corr = uxI + dux
        uy_corr = uyI + duy

    # Final velocity field
    ux, uy = ux_corr, uy_corr

    # Visualizations
    plots_set_1(I_region1, I_region2, ux0, uy0, Im1, Im2, ux, uy)
    plots_set_2(ux, uy)

    return ux, uy, vor

if __name__ == "__main__":
    # Example usage
    # image1_path = "img/White_Oval_1.tif"
    # image2_path = "img/White_Oval_2.tif"
    # image1_path = "img/wall_jet_1.tif"
    # image2_path = "img/wall_jet_2.tif"
    image1_path = "img/vortex_pair_particles_1.tif"
    image2_path = "img/vortex_pair_particles_2.tif"

    ux, uy, vor = flow_diagnostics_run(image1_path, image2_path)
