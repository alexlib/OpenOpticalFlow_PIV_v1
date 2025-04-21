import numpy as np
from scipy.ndimage import convolve

def invariant2_factor(Vx: np.ndarray, Vy: np.ndarray, factor_x: float, factor_y: float) -> np.ndarray:
    """
    Compute the invariant QQ (Q-criterion) using convolution filters.
    
    The Q-criterion identifies vortices as areas where the rotation rate exceeds
    the strain rate. Positive values indicate vortex cores.
    
    Args:
        Vx (np.ndarray): x-component of velocity field
        Vy (np.ndarray): y-component of velocity field
        factor_x (float): Conversion factor from pixel to m (m/pixel) in x
        factor_y (float): Conversion factor from pixel to m (m/pixel) in y
    
    Returns:
        np.ndarray: Computed Q-criterion field
    """
    # Define derivative kernel
    dx = 1
    d_kernel = np.array([[0, -1, 0], 
                        [0, 0, 0], 
                        [0, 1, 0]]) / 2
    
    # Calculate velocity gradients
    # Note: d_kernel.T for x-derivatives, d_kernel for y-derivatives
    Vx_x = convolve(Vx, d_kernel.T/dx, mode='reflect') / factor_x
    Vx_y = convolve(Vx, d_kernel/dx, mode='reflect') / factor_y
    Vy_x = convolve(Vy, d_kernel.T/dx, mode='reflect') / factor_x
    Vy_y = convolve(Vy, d_kernel/dx, mode='reflect') / factor_y
    
    # Calculate Q-criterion components (vectorized implementation)
    # Symmetric part (strain tensor)
    s11 = Vx_x
    s12 = 0.5 * (Vx_y + Vy_x)
    s21 = s12
    s22 = Vy_y
    
    # Antisymmetric part (rotation tensor)
    q12 = 0.5 * (Vx_y - Vy_x)
    q21 = -q12
    
    # Calculate Q-criterion: 0.5 * (||Ω||² - ||S||²)
    # where Ω is the antisymmetric part and S is the symmetric part
    QQ = 2 * (q12 * q21) - (s11**2 + s12*s21 + s21*s12 + s22**2)
    
    return QQ

def invariant2_factor_loop(Vx: np.ndarray, Vy: np.ndarray, factor_x: float, factor_y: float) -> np.ndarray:
    """
    Compute the invariant QQ (Q-criterion) using a loop-based approach.
    This implementation is more explicit but slower than the vectorized version.
    
    Args:
        Vx (np.ndarray): x-component of velocity field
        Vy (np.ndarray): y-component of velocity field
        factor_x (float): Conversion factor from pixel to m (m/pixel) in x
        factor_y (float): Conversion factor from pixel to m (m/pixel) in y
    
    Returns:
        np.ndarray: Computed Q-criterion field
    """
    # Define derivative kernel
    dx = 1
    d_kernel = np.array([[0, -1, 0], 
                        [0, 0, 0], 
                        [0, 1, 0]]) / 2
    
    # Calculate velocity gradients
    Vx_x = convolve(Vx, d_kernel.T/dx, mode='reflect') / factor_x
    Vx_y = convolve(Vx, d_kernel/dx, mode='reflect') / factor_y
    Vy_x = convolve(Vy, d_kernel.T/dx, mode='reflect') / factor_x
    Vy_y = convolve(Vy, d_kernel/dx, mode='reflect') / factor_y
    
    # Get dimensions
    M, N = Vx.shape
    QQ = np.zeros((M, N))
    
    # Calculate Q-criterion at each point
    for i in range(M):
        for j in range(N):
            # Construct velocity gradient tensor
            u = np.array([[Vx_x[i,j], Vx_y[i,j]],
                         [Vy_x[i,j], Vy_y[i,j]]])
            
            # Calculate symmetric and antisymmetric parts
            s = 0.5 * (u + u.T)  # Symmetric part (strain tensor)
            q = 0.5 * (u - u.T)  # Antisymmetric part (rotation tensor)
            
            # Calculate Q-criterion
            QQ[i,j] = 0.5 * (np.trace(q @ q.T) - np.trace(s @ s.T))
    
    return QQ

# Example usage
if __name__ == "__main__":
    # Create sample velocity fields
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    vx = -y  # Rigid body rotation
    vy = x
    
    # Set conversion factors
    factor_x = 0.001  # 1 mm/pixel
    factor_y = 0.001  # 1 mm/pixel
    
    # Calculate Q-criterion using both methods
    import time
    
    start = time.time()
    qq1 = invariant2_factor(vx, vy, factor_x, factor_y)
    print(f"Vectorized time: {time.time() - start:.6f} seconds")
    
    start = time.time()
    qq2 = invariant2_factor_loop(vx, vy, factor_x, factor_y)
    print(f"Loop time: {time.time() - start:.6f} seconds")
    
    # Compare results
    diff = np.abs(qq1 - qq2).max()
    print(f"Maximum difference: {diff:.8f}")
    
    # Plot results
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.quiver(x[::5, ::5], y[::5, ::5], vx[::5, ::5], vy[::5, ::5])
        plt.title('Velocity Field')
        plt.axis('equal')
        
        plt.subplot(132)
        plt.imshow(qq1, cmap='RdBu_r')
        plt.colorbar(label='Q-criterion (Vectorized)')
        plt.title('Q-criterion (Vectorized)')
        
        plt.subplot(133)
        plt.imshow(qq2, cmap='RdBu_r')
        plt.colorbar(label='Q-criterion (Loop)')
        plt.title('Q-criterion (Loop)')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")
