import numpy as np
from scipy.ndimage import convolve

def generate_invmatrix(I, alpha, h):
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # partial derivative
    M = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4  # mixed partial derivatives
    F = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]) / 4  # average
    D2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])  # partial derivative
    H = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    r, c = I.shape

    cmtx = convolve(np.ones_like(I), H / (h * h), mode='constant')

    A11 = I * (convolve(I, D2 / (h * h), mode='nearest') - 2 * I / (h * h)) - alpha * cmtx
    A22 = I * (convolve(I, D2.T / (h * h), mode='nearest') - 2 * I / (h * h)) - alpha * cmtx
    A12 = I * convolve(I, M / (h * h), mode='nearest')

    DetA = A11 * A22 - A12 * A12

    B11 = A22 / DetA
    B12 = -A12 / DetA
    B22 = A11 / DetA

    return B11, B12, B22





# % D = [0, 0, 0; 0,-1,0;0,1,0]; %%% partial derivative 
# % M = [0, 0, 0; 0,1,-1;0,-1,1]; %%% mixed partial derivatives
# % F = [0, 0, 0; 0,1,1;0,1,1]/4; %%% average
# % D2 =  [0, 1, 0; 0,-2,0;0,1,0]; %%% partial derivative
# % H = [1, 1, 1; 1,0,1;1,1,1]; 
# % 
# % [r,c]=size(I);
# % 
# % cmtx = imfilter(ones(size(I)), H, 'same');
# % 
# % A11 = I.*(imfilter(I, D2, 'replicate',  'same') - ... 
# %     2*imfilter(I, D, 'replicate',  'same') - 2*I) - alpha*cmtx;
# % A22 = I.*(imfilter(I, D2', 'replicate',  'same') - ... 
# %     2*imfilter(I, D', 'replicate',  'same') - 2*I) - alpha*cmtx;
# % A12 = I.*(imfilter(I, M, 'replicate',  'same') - ... 
# %     imfilter(I, D, 'replicate', 'same')-imfilter(I, D', 'replicate', 'same') + I);
# % 
# % DetA = A11.*A22-A12.*A12;
# % 
# % B11 = A22./DetA;
# % B12 = -A12./DetA;
# % B22 = A11./DetA;
# % 
