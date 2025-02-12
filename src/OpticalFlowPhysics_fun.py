"""
function [ux,uy,vor,ux_horn,uy_horn,error1]=OpticalFlowPhysics_fun(I1,I2,lambda_1,lambda_2)


% Horn's solution as an initial approximation of u and v
D1 = [0, 0, 0; 0,-1,-1;0,1,1]/2;
F1 = [0, 0, 0; 0,1,1;0,1,1]/4;

Ix = imfilter((I1+I2)/2, D1, 'symmetric',  'same'); 
Iy = imfilter((I1+I2)/2, D1', 'symmetric',  'same');
It = imfilter(I2-I1, F1, 'symmetric',  'same');

maxnum_1=500;
tol_1 = 10^(-12);
% lambda_1 = 10;

[u,v] = horn_schunk_estimator(Ix, Iy, It, lambda_1, tol_1, maxnum_1);
ux_horn = v;
uy_horn = u;


% new model
Dm=0*10^(-3);
f=Dm*laplacian(I1,1);

maxnum=60;
tol = 0.00000001;
% lambda_2 = 4000; % 2000 for dI0 

dx=1; 
dt=1; % unit time
[u,v,error1] = liu_shen_estimator(I1, I2, f, dx, dt, lambda_2, tol, maxnum, uy_horn, ux_horn);

ux=v;
uy=u;

vor=vorticity(ux, uy);


"""

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import laplace

def optical_flow_physics_fun(I1, I2, lambda_1, lambda_2):
    # Define derivative filters
    D1 = np.array([[0, 0, 0], [0, -1, -1], [0, 1, 1]]) / 2
    F1 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]) / 4

    # Compute image gradients
    Ix = convolve((I1 + I2) / 2, D1, mode='reflect')
    Iy = convolve((I1 + I2) / 2, D1.T, mode='reflect')
    It = convolve(I2 - I1, F1, mode='reflect')

    # Horn-Schunck parameters
    maxnum_1 = 500
    tol_1 = 1e-12

    # Initial optical flow estimation using Horn-Schunck method
    u, v = horn_schunk_estimator(Ix, Iy, It, lambda_1, tol_1, maxnum_1)
    ux_horn = v
    uy_horn = u

    # New model
    Dm = 0 * 1e-3
    f = Dm * laplace(I1, mode='reflect')

    # Liu-Shen parameters
    maxnum = 60
    tol = 1e-8
    dx = 1
    dt = 1

    # Refined optical flow estimation using Liu-Shen method
    u, v, error1 = liu_shen_estimator(I1, I2, f, dx, dt, lambda_2, tol, maxnum, uy_horn, ux_horn)
    ux = v
    uy = u

    # Compute vorticity
    vor = vorticity(ux, uy)

    return ux, uy, vor, ux_horn, uy_horn, error1
