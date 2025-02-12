import numpy as np

def piv_min_max_filter(img_in, mask, kernel, mindiff):
    # Convert input image to float
    img_in = img_in.astype(np.float32)
    img_in[mask == 0] = np.nan
    aux_img = np.full(np.array(img_in.shape) + (kernel - 1), np.nan, dtype=np.float32)
    aux_img[(kernel - 1) // 2:-(kernel + 1) // 2, (kernel - 1) // 2:-(kernel + 1) // 2] = img_in
    aux_min = np.zeros_like(aux_img)
    aux_max = np.zeros_like(aux_img)
    aux_min2 = np.zeros_like(aux_img)
    aux_max2 = np.zeros_like(aux_img)

    # Get minima and maxima
    for ky in range((kernel - 1) // 2, aux_img.shape[0] - (kernel + 1) // 2):
        for kx in range((kernel - 1) // 2, aux_img.shape[1] - (kernel + 1) // 2):
            aux = aux_img[ky - (kernel - 1) // 2:ky + (kernel - 1) // 2 + 1, kx - (kernel - 1) // 2:kx + (kernel - 1) // 2 + 1]
            aux = aux.flatten()
            aux = aux[~np.isnan(aux)]
            if aux.size > 1:
                aux_min[ky, kx] = np.min(aux)
                aux_max[ky, kx] = np.max(aux)
            else:
                aux_min[ky, kx] = np.nan
                aux_max[ky, kx] = np.nan

    # Filter minima and maxima
    for ky in range((kernel - 1) // 2, aux_img.shape[0] - (kernel + 1) // 2):
        for kx in range((kernel - 1) // 2, aux_img.shape[1] - (kernel + 1) // 2):
            aux = aux_min[ky - (kernel - 1) // 2:ky + (kernel - 1) // 2 + 1, kx - (kernel - 1) // 2:kx + (kernel - 1) // 2 + 1]
            aux = aux.flatten()
            aux = aux[~np.isnan(aux)]
            if aux.size > 1:
                aux_min2[ky, kx] = np.sum(aux) / aux.size
            else:
                aux_min2[ky, kx] = np.nan
            aux = aux_max[ky - (kernel - 1) // 2:ky + (kernel - 1) // 2 + 1, kx - (kernel - 1) // 2:kx + (kernel - 1) // 2 + 1]
            aux = aux.flatten()
            aux = aux[~np.isnan(aux)]
            if aux.size > 1:
                aux_max2[ky, kx] = np.sum(aux) / aux.size
            else:
                aux_max2[ky, kx] = np.nan

    diff = aux_max2 - aux_min2
    diff[diff < mindiff] = mindiff
    aux_max2 = aux_min2 + diff

    aux_min2 = aux_min2[(kernel - 1) // 2:-(kernel + 1) // 2, (kernel - 1) // 2:-(kernel + 1) // 2]
    aux_max2 = aux_max2[(kernel - 1) // 2:-(kernel + 1) // 2, (kernel - 1) // 2:-(kernel + 1) // 2]

    img_out = (img_in - aux_min2) / (aux_max2 - aux_min2) * 128
    return img_out
