import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Load data
Ux_exact_1 = np.loadtxt('Ux_vortexpair_true_dt0p01.dat')
Uy_exact_1 = np.loadtxt('Uy_vortexpair_true_dt0p01.dat')

Ux_exact_2 = np.loadtxt('Ux_vortexpair_true_dt0p02.dat')
Uy_exact_2 = np.loadtxt('Uy_vortexpair_true_dt0p02.dat')

Ux_exact_3 = np.loadtxt('Ux_vortexpair_true_dt0p03.dat')
Uy_exact_3 = np.loadtxt('Uy_vortexpair_true_dt0p03.dat')

Ux_exact_4 = np.loadtxt('Ux_vortexpair_true_dt0p05.dat')
Uy_exact_4 = np.loadtxt('Uy_vortexpair_true_dt0p05.dat')

Ux_exact_5 = np.loadtxt('Ux_vortexpair_true_dt0p1.dat')
Uy_exact_5 = np.loadtxt('Uy_vortexpair_true_dt0p1.dat')

Ux_exact_6 = np.loadtxt('Ux_vortexpair_true_dt0p2.dat')
Uy_exact_6 = np.loadtxt('Uy_vortexpair_true_dt0p2.dat')

Ux_liu_1 = np.loadtxt('Ux_vortexpair_hybrid_dt0p01.dat')
Uy_liu_1 = np.loadtxt('Uy_vortexpair_hybrid_dt0p01.dat')

Ux_liu_2 = np.loadtxt('Ux_vortexpair_hybrid_dt0p02.dat')
Uy_liu_2 = np.loadtxt('Uy_vortexpair_hybrid_dt0p02.dat')

Ux_liu_3 = np.loadtxt('Ux_vortexpair_hybrid_dt0p03.dat')
Uy_liu_3 = np.loadtxt('Uy_vortexpair_hybrid_dt0p03.dat')

Ux_liu_4 = np.loadtxt('Ux_vortexpair_hybrid_dt0p05.dat')
Uy_liu_4 = np.loadtxt('Uy_vortexpair_hybrid_dt0p05.dat')

Ux_liu_5 = np.loadtxt('Ux_vortexpair_hybrid_dt0p1.dat')
Uy_liu_5 = np.loadtxt('Uy_vortexpair_hybrid_dt0p1.dat')

Ux_liu_6 = np.loadtxt('Ux_vortexpair_hybrid_dt0p2.dat')
Uy_liu_6 = np.loadtxt('Uy_vortexpair_hybrid_dt0p2.dat')

Ux_corr_1 = np.loadtxt('Ux_vortexpair_corr_dt0p01.dat')
Uy_corr_1 = np.loadtxt('Uy_vortexpair_corr_dt0p01.dat')

Ux_corr_2 = np.loadtxt('Ux_vortexpair_corr_dt0p02.dat')
Uy_corr_2 = np.loadtxt('Uy_vortexpair_corr_dt0p02.dat')

Ux_corr_3 = np.loadtxt('Ux_vortexpair_corr_dt0p03.dat')
Uy_corr_3 = np.loadtxt('Uy_vortexpair_corr_dt0p03.dat')

Ux_corr_4 = np.loadtxt('Ux_vortexpair_corr_dt0p05.dat')
Uy_corr_4 = np.loadtxt('Uy_vortexpair_corr_dt0p05.dat')

Ux_corr_5 = np.loadtxt('Ux_vortexpair_corr_dt0p1.dat')
Uy_corr_5 = np.loadtxt('Uy_vortexpair_corr_dt0p1.dat')

Ux_corr_6 = np.loadtxt('Ux_vortexpair_corr_dt0p2.dat')
Uy_corr_6 = np.loadtxt('Uy_vortexpair_corr_dt0p2.dat')

# Resize correlation data
m, n = Ux_exact_2.shape

Ux_corr_a_1 = zoom(Ux_corr_1, (m / Ux_corr_1.shape[0], n / Ux_corr_1.shape[1]))
Uy_corr_a_1 = zoom(Uy_corr_1, (m / Uy_corr_1.shape[0], n / Uy_corr_1.shape[1]))

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

# Plotting
x0 = 250
y = np.arange(1, m + 1)

plt.figure(2)
plt.plot(y, Ux_exact_4[:, x0], '.k', y, Ux_liu_4[:, x0], '-k', y[::8], Ux_corr_a_4[::8, x0], '+k')
plt.grid()
plt.axis([0, 500, -3, 5])
plt.xlabel('y (pixels)')
plt.ylabel('u_x (pixels/unit time)')
plt.legend(['Truth', 'Hybrid Method', 'Correlation Method 1'])

# Plot velocity vector field
plt.figure(3)
gx = 30
offset = 1
plt.quiver(Ux_liu_4[::gx, ::gx], Uy_liu_4[::gx, ::gx], scale=3, color='black')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.title('Velocity Field (Hybrid Method)')

# Plot streamlines
plt.figure(4)
x, y = np.meshgrid(np.arange(n), np.arange(m))
dn = 10
dm = 10
sx, sy = np.meshgrid(np.arange(1, n, dn), np.arange(1, m, dm))
plt.streamplot(x, y, Ux_liu_4, Uy_liu_4, color='black')
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.title('Streamlines (Hybrid Method)')

# Plot velocity vector field
plt.figure(5)
plt.quiver(Ux_corr_a_4[::8, ::8], Uy_corr_a_4[::8, ::8], scale=3, color='black')
plt.xlabel('x (regions)')
plt.ylabel('y (regions)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.title('Velocity Field (Correlation Method 1)')

# Plot streamlines
plt.figure(6)
plt.streamplot(x[::8, ::8], y[::8, ::8], Ux_corr_a_4[::8, ::8], Uy_corr_a_4[::8, ::8], color='black')
plt.xlabel('x (regions)')
plt.ylabel('y (regions)')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.title('Streamlines (Correlation Method 1)')

U_max = []
error_liu = []
error_corr = []
grad_avg = []

for i in range(2, 7):
    Ux_liu = eval(f'Ux_liu_{i}')
    Uy_liu = eval(f'Uy_liu_{i}')
    
    Ux_corr = eval(f'Ux_corr_a_{i}')
    Uy_corr = eval(f'Uy_corr_a_{i}')
    
    Ux_exact = eval(f'Ux_exact_{i}')
    Uy_exact = eval(f'Uy_exact_{i}')
    
    dU_liu = np.sqrt((Ux_liu[50:, 50:] - Ux_exact[50:, 50:])**2 + (Uy_liu[50:, 50:] - Uy_exact[50:, 50:])**2)
    dU_corr = np.sqrt((Ux_corr[::8, ::8] - Ux_exact[::8, ::8])**2 + (Uy_corr[::8, ::8] - Uy_exact[::8, ::8])**2)
    U_mag = np.sqrt(Ux_exact**2 + Uy_exact**2)
    U_max.append(np.max(U_mag))
    
    grad_mag = np.gradient(Ux_exact[50:, 50:], Uy_exact[50:, 50:])
    grad_avg.append(np.mean(np.sqrt(grad_mag[0]**2 + grad_mag[1]**2)))
    
    error_liu.append(np.mean(dU_liu))
    error_corr.append(np.mean(dU_corr))

data_err = np.loadtxt('errors_OF_lavision.dat')

plt.figure(20)
plt.plot(U_max, error_liu, 'o-k', label='Hybrid Method')
plt.plot(data_err[:, 0], data_err[:, 1], '-sk', label='Optical Flow Method')
plt.plot(U_max, error_corr, '>-k', label='Correlation Method 1')
plt.plot(data_err[:, 0], data_err[:, 2], '-dk', label='Correlation Method 2')
plt.grid()
plt.xlabel('Max Displacement (pixels)')
plt.ylabel('RMS Error (pixels)')
plt.legend()

plt.figure(21)
plt.plot(U_max, np.array(error_liu) / np.array(U_max), 'o-', label='Hybrid Method')
plt.plot(U_max, np.array(error_corr) / np.array(U_max), '>-k', label='Correlation Method')
plt.grid()
plt.xlabel('Max Displacement (pixels)')
plt.ylabel('Relative RMS Error (pixels)')
plt.legend()

plt.figure(22)
plt.imshow(dU_liu, extent=[1, 500, 1, 500], vmin=0, vmax=1.5)
plt.colorbar()
plt.xlabel('x (pixels)')
plt.ylabel('y (pixels)')
plt.title('Error (pixels/unit time) - Hybrid Method')

plt.figure(23)
plt.imshow(dU_corr, extent=[1, 62, 1, 62], vmin=0, vmax=1.5)
plt.colorbar()
plt.xlabel('x (regions)')
plt.ylabel('y (regions)')
plt.title('Error (pixels/unit time) - Correlation Method')

plt.figure(30)
plt.plot(grad_avg, error_liu, 'o-k', label='Hybrid Method')
plt.plot(grad_avg, data_err[:, 1], '-sk', label='Optical Flow Method')
plt.plot(grad_avg, error_corr, '>-k', label='Correlation Method 1')
plt.plot(grad_avg, data_err[:, 2], '-dk', label='Correlation Method 2')
plt.grid()
plt.xlabel('Velocity Gradient Magnitude (1/unit time)')
plt.ylabel('RMS Error (pixels)')
plt.legend()

plt.show()
