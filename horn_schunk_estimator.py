import numpy as np

def horn_schunk_estimator(Ix, Iy, It, lambda_, tol, maxnum):
    r, c = Ix.shape

    chorizontal = np.array([3] + [8] * (c - 2) + [3])
    cvertical = np.array([3] + [8] * (r - 2) + [3])
    cmtx = 8 * np.ones((r, c))
    cmtx[0, :] = chorizontal
    cmtx[-1, :] = chorizontal
    cmtx[:, 0] = cvertical
    cmtx[:, -1] = cvertical

    uv = (Ix * Iy) / (cmtx * (Ix**2 + Iy**2) + lambda_ * cmtx**2)
    u1 = (Iy**2 + lambda_ * cmtx) / (cmtx * (Ix**2 + Iy**2) + lambda_ * cmtx**2)
    u2 = (Ix * It) / (Ix**2 + Iy**2 + lambda_ * cmtx)
    v1 = (Ix**2 + lambda_ * cmtx) / (cmtx * (Ix**2 + Iy**2) + lambda_ * cmtx**2)
    v2 = (Iy * It) / (Ix**2 + Iy**2 + lambda_ * cmtx)

    k = 0
    total_error = 1e8
    u = np.zeros((r, c))
    v = np.zeros((r, c))

    while total_error > tol and k < maxnum:
        unew = np.zeros((r, c))
        vnew = np.zeros((r, c))
        for n in range(c):
            for m in range(r):
                if n == 1:
                    if m == 1:
                        tmpu = u[m+1, n] + u[m, n+1] + u[m+1, n+1]
                        tmpv = v[m+1, n] + v[m, n+1] + v[m+1, n+1]
                    elif m == r - 1:
                        tmpu = u[m-1, n] + u[m, n+1] + u[m-1, n+1]
                        tmpv = v[m-1, n] + v[m, n+1] + v[m-1, n+1]
                    else:
                        tmpu = u[m-1, n] + u[m+1, n] + u[m-1, n+1] + u[m, n+1] + u[m+1, n+1]
                        tmpv = v[m-1, n] + v[m+1, n] + v[m-1, n+1] + v[m, n+1] + v[m+1, n+1]
                elif n == c - 1:
                    if m == 1:
                        tmpu = u[m+1, n] + u[m, n-1] + u[m+1, n-1]
                        tmpv = v[m+1, n] + v[m, n-1] + v[m+1, n-1]
                    elif m == r - 1:
                        tmpu = u[m-1, n] + u[m, n-1] + u[m-1, n-1]
                        tmpv = v[m-1, n] + v[m, n-1] + v[m-1, n-1]
                    else:
                        tmpu = u[m-1, n] + u[m+1, n] + u[m-1, n-1] + u[m, n-1] + u[m+1, n-1]
                        tmpv = v[m-1, n] + v[m+1, n] + v[m-1, n-1] + v[m, n-1] + v[m+1, n-1]
                else:
                    if m == 1:
                        tmpu = u[m, n-1] + u[m, n+1] + u[m+1, n-1] + u[m+1, n] + u[m+1, n+1]
                        tmpv = v[m, n-1] + v[m, n+1] + v[m+1, n-1] + v[m+1, n] + v[m+1, n+1]
                    elif m == r - 1:
                        tmpu = u[m, n-1] + u[m, n+1] + u[m-1, n-1] + u[m-1, n] + u[m-1, n+1]
                        tmpv = v[m, n-1] + v[m, n+1] + v[m-1, n-1] + v[m-1, n] + v[m-1, n+1]
                    else:
                        tmpu = (u[m-1, n-1] + u[m-1, n] + u[m-1, n+1] + u[m, n-1] + 
                                u[m, n+1] + u[m+1, n-1] + u[m+1, n] + u[m+1, n+1])
                        tmpv = (v[m-1, n-1] + v[m-1, n] + v[m-1, n+1] + v[m, n-1] + 
                                v[m, n+1] + v[m+1, n-1] + v[m+1, n] + v[m+1, n+1])
                unew[m, n] = u1[m, n] * tmpu - uv[m, n] * tmpv - u2[m, n]
                vnew[m, n] = v1[m, n] * tmpv - uv[m, n] * tmpu - v2[m, n]
        
        total_error = (np.linalg.norm(unew - u, 'fro') + np.linalg.norm(vnew - v, 'fro')) / (r * c)
        u = unew
        v = vnew
        k += 1

    return u, v
