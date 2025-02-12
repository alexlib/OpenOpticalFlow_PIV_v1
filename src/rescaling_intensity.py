import numpy as np

def rescaling_intensity(Im1, Im2, max_intensity_value):
    Imax1 = np.max(Im1)
    Imin1 = np.min(Im1)
    Im1a = (Im1 - Imin1) / (Imax1 - Imin1)

    Imax2 = np.max(Im2)
    Imin2 = np.min(Im2)
    Im2a = (Im2 - Imin2) / (Imax2 - Imin2)

    Im1 = Im1a * max_intensity_value
    Im2 = Im2a * max_intensity_value

    I1 = Im1.astype(np.float64)
    I2 = Im2.astype(np.float64)

    return I1, I2



