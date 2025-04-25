import numpy as np
from scipy.ndimage import convolve

"""
function [grad_mag]=gradient(Vx, Vy)

% Vx = imfilter(Vx, [1 1 1 1 1]'*[1 1 1 1 1]/25,'symmetric');
% Vy = imfilter(Vy, [1 1 1 1 1]'*[1,1 1 1,1]/25,'symmetric');

dx=1;
D = [0, -1, 0; 0,0,0; 0,1,0]/2; %%% partial derivative 
Vy_x = imfilter(Vy, D'/dx, 'symmetric',  'same'); 
Vx_y = imfilter(Vx, D/dx, 'symmetric',  'same');
grad_mag=(Vy_x.^2+Vx_y.^2).^0.5;
"""

def gradient(Vx, Vy):
    dx = 1
    D = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2  # partial derivative

    Vy_x = convolve(Vy, D.T / dx, mode='reflect')
    Vx_y = convolve(Vx, D / dx, mode='reflect')
    
    grad_mag = np.sqrt(Vy_x**2 + Vx_y**2)
    
    return grad_mag




















