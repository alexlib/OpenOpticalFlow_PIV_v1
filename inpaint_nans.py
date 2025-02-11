import numpy as np
from scipy.sparse import spdiags, linalg

def inpaint_nans(A, method=0):
  n, m = A.shape
  A = A.flatten()
  nm = n * m
  k = np.isnan(A)

  nan_list = np.where(k)[0]
  known_list = np.where(~k)[0]
  nan_count = len(nan_list)
  nr, nc = np.unravel_index(nan_list, (n, m))
  nan_list = np.column_stack((nan_list, nr, nc))

  if method not in range(6):
    raise ValueError('If supplied, method must be one of: {0,1,2,3,4,5}.')

  if method == 0:
    talks_to = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    neighbors_list = identify_neighbors(n, m, nan_list, talks_to)
    all_list = np.vstack((nan_list, neighbors_list))

    L = (all_list[:, 1] > 1) & (all_list[:, 1] < n)
    nl = np.sum(L)
    if nl > 0:
      fda = spdiags([np.ones(nl), -2 * np.ones(nl), np.ones(nl)], [-1, 0, 1], nm, nm)
    else:
      fda = spdiags([], [], nm, nm)

    L = (all_list[:, 2] > 1) & (all_list[:, 2] < m)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -2 * np.ones(nl), np.ones(nl)], [-n, 0, n], nm, nm)

    rhs = -fda[:, known_list] @ A[known_list]
    k = np.any(fda[:, nan_list[:, 0]], axis=1)

    B = A.copy()
    B[nan_list[:, 0]] = linalg.spsolve(fda[k][:, nan_list[:, 0]], rhs[k])

  elif method == 1:
    i, j = np.meshgrid(np.arange(1, n-1), np.arange(m))
    ind = i.flatten() + j.flatten() * n
    np_ = (n-2) * m
    fda = spdiags([np.ones(np_), -2 * np.ones(np_), np.ones(np_)], [-1, 0, 1], nm, nm)

    i, j = np.meshgrid(np.arange(n), np.arange(1, m-1))
    ind = i.flatten() + j.flatten() * n
    np_ = n * (m-2)
    fda += spdiags([np.ones(np_), -2 * np.ones(np_), np.ones(np_)], [-n, 0, n], nm, nm)

    rhs = -fda[:, known_list] @ A[known_list]
    k = np.any(fda[:, nan_list[:, 0]], axis=1)

    B = A.copy()
    B[nan_list[:, 0]] = linalg.spsolve(fda[k][:, nan_list[:, 0]], rhs[k])

  elif method == 2:
    L = (nan_list[:, 1] > 1) & (nan_list[:, 1] < n)
    nl = np.sum(L)
    if nl > 0:
      fda = spdiags([np.ones(nl), -2 * np.ones(nl), np.ones(nl)], [-1, 0, 1], nm, nm)
    else:
      fda = spdiags([], [], nm, nm)

    L = (nan_list[:, 2] > 1) & (nan_list[:, 2] < m)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -2 * np.ones(nl), np.ones(nl)], [-n, 0, n], nm, nm)

    if 1 in nan_list[:, 0]:
      fda[0, [0, 1, n]] = [-2, 1, 1]
    if n in nan_list[:, 0]:
      fda[n-1, [n-1, n-2, 2*n-1]] = [-2, 1, 1]
    if nm-n+1 in nan_list[:, 0]:
      fda[nm-n, [nm-n, nm-n+1, nm-2*n]] = [-2, 1, 1]
    if nm in nan_list[:, 0]:
      fda[nm-1, [nm-1, nm-2, nm-n-1]] = [-2, 1, 1]

    rhs = -fda[:, known_list] @ A[known_list]

    B = A.copy()
    k = nan_list[:, 0]
    B[k] = linalg.spsolve(fda[k][:, k], rhs[k])

  elif method == 3:
    talks_to = np.array([[-2, 0], [-1, -1], [-1, 0], [-1, 1], [0, -2], [0, -1], [0, 1], [0, 2], [1, -1], [1, 0], [1, 1], [2, 0]])
    neighbors_list = identify_neighbors(n, m, nan_list, talks_to)
    all_list = np.vstack((nan_list, neighbors_list))

    L = (all_list[:, 1] >= 3) & (all_list[:, 1] <= n-2) & (all_list[:, 2] >= 3) & (all_list[:, 2] <= m-2)
    nl = np.sum(L)
    if nl > 0:
      fda = spdiags([np.ones(nl), 2 * np.ones(nl), -8 * np.ones(nl), 2 * np.ones(nl), np.ones(nl), -8 * np.ones(nl), 20 * np.ones(nl), -8 * np.ones(nl), np.ones(nl), 2 * np.ones(nl), -8 * np.ones(nl), 2 * np.ones(nl), np.ones(nl)], [-2*n, -n-1, -n, -n+1, -2, -1, 0, 1, 2, n-1, n, n+1, 2*n], nm, nm)
    else:
      fda = spdiags([], [], nm, nm)

    L = (((all_list[:, 1] == 2) | (all_list[:, 1] == n-1)) & (all_list[:, 2] >= 2) & (all_list[:, 2] <= m-1)) | (((all_list[:, 2] == 2) | (all_list[:, 2] == m-1)) & (all_list[:, 1] >= 2) & (all_list[:, 1] <= n-1))
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), np.ones(nl), -4 * np.ones(nl), np.ones(nl), np.ones(nl)], [-n, -1, 0, 1, n], nm, nm)

    L = ((all_list[:, 1] == 1) | (all_list[:, 1] == n)) & (all_list[:, 2] >= 2) & (all_list[:, 2] <= m-1)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -2 * np.ones(nl), np.ones(nl)], [-n, 0, n], nm, nm)

    L = ((all_list[:, 2] == 1) | (all_list[:, 2] == m)) & (all_list[:, 1] >= 2) & (all_list[:, 1] <= n-1)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -2 * np.ones(nl), np.ones(nl)], [-1, 0, 1], nm, nm)

    rhs = -fda[:, known_list] @ A[known_list]
    k = np.any(fda[:, nan_list[:, 0]], axis=1)

    B = A.copy()
    B[nan_list[:, 0]] = linalg.spsolve(fda[k][:, nan_list[:, 0]], rhs[k])

  elif method == 4:
    hv_list = np.array([[-1, -1, 0], [1, 1, 0], [-n, 0, -1], [n, 0, 1]])
    hv_springs = []
    for i in range(4):
      hvs = nan_list + hv_list[i]
      k = (hvs[:, 1] >= 1) & (hvs[:, 1] <= n) & (hvs[:, 2] >= 1) & (hvs[:, 2] <= m)
      hv_springs.append(np.column_stack((nan_list[k, 0], hvs[k, 0])))

    hv_springs = np.unique(np.sort(np.vstack(hv_springs), axis=1), axis=0)
    nhv = hv_springs.shape[0]
    springs = spdiags([np.ones(nhv), -np.ones(nhv)], [0, 1], nhv, nm)

    rhs = -springs[:, known_list] @ A[known_list]

    B = A.copy()
    B[nan_list[:, 0]] = linalg.spsolve(springs[:, nan_list[:, 0]], rhs)

  elif method == 5:
    fda = spdiags([], [], nm, nm)

    L = (nan_list[:, 1] > 1) & (nan_list[:, 2] > 1)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [-n-1, 0], nm, nm)

    L = nan_list[:, 2] > 1
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [-n, 0], nm, nm)

    L = (nan_list[:, 1] < n) & (nan_list[:, 2] > 1)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [-n+1, 0], nm, nm)

    L = nan_list[:, 1] > 1
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [-1, 0], nm, nm)

    L = nan_list[:, 1] < n
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [1, 0], nm, nm)

    L = (nan_list[:, 1] > 1) & (nan_list[:, 2] < m)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [n-1, 0], nm, nm)

    L = nan_list[:, 2] < m
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [n, 0], nm, nm)

    L = (nan_list[:, 1] < n) & (nan_list[:, 2] < m)
    nl = np.sum(L)
    if nl > 0:
      fda += spdiags([np.ones(nl), -np.ones(nl)], [n+1, 0], nm, nm)

    rhs = -fda[:, known_list] @ A[known_list]

    B = A.copy()
    k = nan_list[:, 0]
    B[k] = linalg.spsolve(fda[k][:, k], rhs[k])

  return B.reshape((n, m))

def identify_neighbors(n, m, nan_list, talks_to):
  if nan_list.size == 0:
    return np.array([])

  nan_count = nan_list.shape[0]
  talk_count = talks_to.shape[0]

  nn = np.zeros((nan_count * talk_count, 2), dtype=int)
  j = [0, nan_count]
  for i in range(talk_count):
    nn[j[0]:j[1], :] = nan_list[:, 1:3] + talks_to[i]
    j = [j[1], j[1] + nan_count]

  L = (nn[:, 0] < 0) | (nn[:, 0] >= n) | (nn[:, 1] < 0) | (nn[:, 1] >= m)
  nn = nn[~L]

  neighbors_list = np.column_stack((np.ravel_multi_index((nn[:, 0], nn[:, 1]), (n, m)), nn))

  neighbors_list = np.unique(neighbors_list, axis=0)
  neighbors_list = neighbors_list[np.isin(neighbors_list[:, 0], nan_list[:, 0], invert=True)]

  return neighbors_list
















