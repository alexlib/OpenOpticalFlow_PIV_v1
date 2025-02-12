import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def load_data(filename):
    return np.loadtxt(filename)

Ux_exact_1 = load_data('Ux_vortexpair_true_dt0p01.dat')
Uy_exact_1 = load_data('Uy_vortexpair_true_dt0p01.dat')

Ux_exact_2 = load_data('Ux_vortexpair_true_dt0p02.dat')
Uy_exact_2 = load_data('Uy_vortexpair_true_dt0p02.dat')

Ux_exact_3 = load_data('Ux_vortexpair_true_dt0p03.dat')
Uy_exact_3 = load_data('Uy_vortexpair_true_dt0p03.dat')

Ux_exact_4 = load_data('Ux_vortexpair_true_dt0p05.dat')
Uy_exact_4 = load_data('Uy_vortexpair_true_dt0p05.dat')

Ux_exact_5 = load_data('Ux_vortexpair_true_dt0p1.dat')
Uy_exact_5 = load_data('Uy_vortexpair_true_dt0p1.dat')

Ux_exact_6 = load_data('Ux_vortexpair_true_dt0p2.dat')
Uy_exact_6 = load_data('Uy_vortexpair_true_dt0p2.dat')

Ux_liu_1 = load_data('Ux_vortexpair_hybrid_dt0p01.dat')
Uy_liu_1 = load_data('Uy_vortexpair_hybrid_dt0p01.dat')

Ux_liu_2 = load_data('Ux_vortexpair_hybrid_dt0p02.dat')
Uy_liu_2 = load_data('Uy_vortexpair_hybrid_dt0p02.dat')

Ux_liu_3 = load_data('Ux_vortexpair_hybrid_dt0p03.dat')
Uy_liu_3 = load_data('Uy_vortexpair_hybrid_dt0p03.dat')

Ux_liu_4 = load_data('Ux_vortexpair_hybrid_dt0p05.dat')
Uy_liu_4 = load_data('Uy_vortexpair_hybrid_dt0p05.dat')

Ux_liu_5 = load_data('Ux_vortexpair_hybrid_dt0p1.dat')
Uy_liu_5 = load_data('Uy_vortexpair_hybrid_dt0p1.dat')

Ux_liu_6 = load_data('Ux_vortexpair_hybrid_dt0p2.dat')
Uy_liu_6 = load_data('Uy_vortexpair_hybrid_dt0p2.dat')

Ux_corr_1 = load_data('Ux_vortexpair_corr_dt0p01.dat')
Uy_corr_1 = load_data('Uy_vortexpair_corr_dt0p01.dat')

Ux_corr_2 = load_data('Ux_vortexpair_corr_dt0p02.dat')
Uy_corr_2 = load_data('Uy_vortexpair_corr_dt0p02.dat')

Ux_corr_3 = load_data('Ux_vortexpair_corr_dt0p03.dat')
Uy_corr_3 = load_data('Uy_vortexpair_corr_dt0p03.dat')

Ux_corr_4 = load_data('Ux_vortexpair_corr_dt0p05.dat')
Uy_corr_4 = load_data('Uy_vortexpair_corr_dt0p05.dat')

Ux_corr_5 = load_data('Ux_vortexpair_corr_dt0p1.dat')
Uy_corr_5 = load_data('Uy_vortexpair_corr_dt0p1.dat')

Ux_corr_6 = load_data('Ux_vortexpair_corr_dt0p2.dat')
Uy_corr_6 = load_data('Uy_vortexpair_corr_dt0p2.dat')

m, n = Ux_exact_2.shape

Ux_corr_a_2 = zoom(Ux_corr_2, (m / Ux_corr_2.shape[0], n / Ux_corr_2.shape[1]))
Uy_corr_a_2 = zoom(Uy_corr_2, (m / Uy_corr_2.shape[0], n / Uy_corr_2.shape[1]))

Ux_corr_a_3 = zoom(Ux_corr_3, (m / Ux_corr_3.shape[0], n / Ux_corr_3.shape[1]))
Uy_corr_a_3 = zoom(Uy_corr_3, (m / Uy_corr_3.shape[0], n / Uy_corr_3.shape[1]))

Ux_corr_a_4 = zoom(Ux_corr_4, (m / Ux_corr_4.shape[0], n / Ux_corr_4.shape[1]))
Uy_corr_a_4 = zoom(Uy_corr_4, (m / Uy_corr_4.shape[0], n / Uy_corr_4.shape[1]))

Ux_corr_a_5 = zoom(Ux_corr_5, (m / Ux_corr_5.shape[0], n / Ux_corr_5.shape[1]))
Uy_corr_a_5 = zoom(Uy_corr_5, (m / Uy_corr_5.shape[0], n / Uy_corr_5.shape[1]))

Ux_corr_a_6 = zoom(Ux_corr_6, (m / Ux_corr_6.shape[0], n / Ux_corr_6.shape[1]))
Uy_corr_a_6 = zoom(Uy_corr_6, (m / Uy_corr_6.shape[0], n / Uy_corr_6.shape[1]))

x0 = 250
y = np.arange(1, m + 1)

plt.figure(2)
plt.plot(y, Ux_exact_6[:m, x0], '.k', label='Truth')
plt.plot(y, Ux_liu_6[:, x0], '-r', label='Optical Flow')
plt.plot(y[::8], Ux_corr_a_6[::8, x0], '+b', label='Correlation')
plt.grid()
plt.xlabel('y (pixels)')
plt.ylabel('u_x (pixels/unit time)')
plt.legend()

plt.figure(3)
gx = 30
offset = 1
plt.quiver(Ux_liu_6[::gx, ::gx], Uy_liu_6[::gx, ::gx], color='red')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()

plt.figure(4)
x, y = np.meshgrid(np.arange(n), np.arange(m))
plt.streamplot(x, y, Ux_liu_6, Uy_liu_6, color='blue')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()

plt.figure(5)
plt.quiver(Ux_corr_a_6[::gx, ::gx], Uy_corr_a_6[::gx, ::gx], color='red')
plt.xlabel('x (regions)')
plt.ylabel('y (regions)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()

plt.figure(6)
plt.streamplot(x[::8, ::8], y[::8, ::8], Ux_corr_a_6[::8, ::8], Uy_corr_a_6[::8, ::8], color='blue')
plt.xlabel('x (regions)')
plt.ylabel('y (regions)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()

grad_avg = []
U_max = []
error_angle_liu = []
error_angle_corr = []

for i in range(2, 7):
    Ux_liu = eval(f'Ux_liu_{i}')
    Uy_liu = eval(f'Uy_liu_{i}')
    
    Ux_corr = eval(f'Ux_corr_a_{i}')
    Uy_corr = eval(f'Uy_corr_a_{i}')
    
    Ux_exact = eval(f'Ux_exact_{i}')
    Uy_exact = eval(f'Uy_exact_{i}')
    
    n = 450
    angle_liu = np.arctan2(Uy_liu[50:n, 50:n], Ux_liu[50:n, 50:n])
    angle_corr = np.arctan2(Uy_corr[50:n, 50:n], Ux_corr[50:n, 50:n])
    angle_exact = np.arctan2(Uy_exact[50:n, 50:n], Ux_exact[50:n, 50:n])
    
    dangle_liu = np.sqrt((angle_liu - angle_exact) ** 2)
    dangle_corr = np.sqrt((angle_corr - angle_exact) ** 2)
    
    U_mag = np.sqrt(Ux_exact ** 2 + Uy_exact ** 2)
    grad_mag = np.gradient(Ux_exact[50:n, 50:n], Uy_exact[50:n, 50:n])
    
    grad_avg.append(np.mean(grad_mag))
    U_max.append(np.max(U_mag))
    
    error_angle_liu.append(np.mean(dangle_liu) * 180 / np.pi)
    error_angle_corr.append(np.mean(dangle_corr) * 180 / np.pi)

data_err_ang_Umax = load_data('data err_ang_Umax.dat')

plt.figure(20)
plt.plot(U_max, error_angle_liu, 'o-k', label='Hybrid Method')
plt.plot(data_err_ang_Umax[:, 0], data_err_ang_Umax[:, 1], '-sk', label='Optical Flow Method')
plt.plot(U_max, error_angle_corr, '>-k', label='Correlation Method 1')
plt.plot(data_err_ang_Umax[:, 0], data_err_ang_Umax[:, 2], '-dk', label='Correlation Method 2')
plt.grid()
plt.xlabel('Max Displacement (pixels)')
plt.ylabel('RMS Angular Error (deg)')
plt.legend()

plt.figure(21)
plt.imshow(dangle_liu, extent=[0, 500, 0, 500], origin='lower')
plt.colorbar()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Angular Error (deg)')

plt.figure(22)
plt.imshow(dangle_corr, extent=[0, 62, 0, 62], origin='lower')
plt.colorbar()
plt.xlabel('x (regions)')
plt.ylabel('y (regions)')
plt.title('Angular Error (deg)')

data_err_ang_Ugrad = load_data('data err_ang_Ugrad.dat')

plt.figure(30)
plt.plot(grad_avg, error_angle_liu, 'o-k', label='Hybrid Method')
plt.plot(data_err_ang_Ugrad[:, 0], data_err_ang_Ugrad[:, 1], '-sk', label='Optical Flow Method')
plt.plot(grad_avg, error_angle_corr, '>-', label='Correlation Method 1')
plt.plot(data_err_ang_Ugrad[:, 0], data_err_ang_Ugrad[:, 2], '-dk', label='Correlation Method 2')
plt.grid()
plt.xlabel('Velocity Gradient (1/unit time)')
plt.ylabel('RMS Angular Error (deg)')
plt.legend()

plt.show()
