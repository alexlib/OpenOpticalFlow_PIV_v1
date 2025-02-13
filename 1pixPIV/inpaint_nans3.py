import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def inpaint_nans3(A, method=0):
  """
  INPAINT_NANS3: in-paints over nans in a 3-D array
  usage: B = inpaint_nans3(A)          # default method (0)
  usage: B = inpaint_nans3(A, method)  # specify method used

  Solves approximation to a boundary value problem to
  interpolate and extrapolate holes in a 3-D array.
  """
  NA = A.shape
  A = A.flatten()
  nt = np.prod(NA)
  k = np.isnan(A)

  nan_list = np.where(k)[0]
  known_list = np.where(~k)[0]
  nan_count = len(nan_list)

  n1, n2, n3 = np.unravel_index(nan_list, NA)
  nan_list = np.column_stack((nan_list, n1, n2, n3))

  if method not in [0, 1]:
    raise ValueError('If supplied, method must be one of: {0,1}.')

  if method == 0:
    talks_to = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
    neighbors_list = identify_neighbors(NA, nan_list, talks_to)

    all_list = np.vstack((nan_list, neighbors_list))

    L = np.where((all_list[:, 1] > 1) & (all_list[:, 1] < NA[0]))[0]
    nL = len(L)
    if nL > 0:
      fda = lil_matrix((nt, nt))
      fda[all_list[L, 0], all_list[L, 0]] = -2
      fda[all_list[L, 0], all_list[L, 0] - 1] = 1
      fda[all_list[L, 0], all_list[L, 0] + 1] = 1
    else:
      fda = lil_matrix((nt, nt))

    L = np.where((all_list[:, 2] > 1) & (all_list[:, 2] < NA[1]))[0]
    nL = len(L)
    if nL > 0:
      fda[all_list[L, 0], all_list[L, 0]] -= 2
      fda[all_list[L, 0], all_list[L, 0] - NA[0]] = 1
      fda[all_list[L, 0], all_list[L, 0] + NA[0]] = 1

    L = np.where((all_list[:, 3] > 1) & (all_list[:, 3] < NA[2]))[0]
    nL = len(L)
    if nL > 0:
      ntimesm = NA[0] * NA[1]
      fda[all_list[L, 0], all_list[L, 0]] -= 2
      fda[all_list[L, 0], all_list[L, 0] - ntimesm] = 1
      fda[all_list[L, 0], all_list[L, 0] + ntimesm] = 1

    rhs = -fda[:, known_list] @ A[known_list]
    k = np.where(np.any(fda[:, nan_list[:, 0]], axis=1))[0]

    B = A.copy()
    B[nan_list[:, 0]] = spsolve(fda[k, :][:, nan_list[:, 0]], rhs[k])

  elif method == 1:
    hv_list = np.array([[-1, -1, 0, 0], [1, 1, 0, 0], [-NA[0], 0, -1, 0], [NA[0], 0, 1, 0],
              [-NA[0] * NA[1], 0, 0, -1], [NA[0] * NA[1], 0, 0, 1]])
    hv_springs = []
    for i in range(hv_list.shape[0]):
      hvs = nan_list + hv_list[i, :]
      k = (hvs[:, 1] >= 1) & (hvs[:, 1] <= NA[0]) & (hvs[:, 2] >= 1) & (hvs[:, 2] <= NA[1]) & (hvs[:, 3] >= 1) & (hvs[:, 3] <= NA[2])
      hv_springs.append(np.column_stack((nan_list[k, 0], hvs[k, 0])))

    hv_springs = np.vstack(hv_springs)
    hv_springs = np.unique(np.sort(hv_springs, axis=1), axis=0)

    nhv = hv_springs.shape[0]
    springs = lil_matrix((nhv, nt))
    springs[np.arange(nhv), hv_springs[:, 0]] = 1
    springs[np.arange(nhv), hv_springs[:, 1]] = -1

    rhs = -springs[:, known_list] @ A[known_list]

    B = A.copy()
    B[nan_list[:, 0]] = spsolve(springs[:, nan_list[:, 0]], rhs)

  B = B.reshape(NA)
  return B

def identify_neighbors(NA, nan_list, talks_to):
  if nan_list.size == 0:
    return np.array([])

  nan_count = nan_list.shape[0]
  talk_count = talks_to.shape[0]

  nn = np.zeros((nan_count * talk_count, 3), dtype=int)
  j = [0, nan_count]
  for i in range(talk_count):
    nn[j[0]:j[1], :] = nan_list[:, 1:4] + talks_to[i, :]
    j = [j[1], j[1] + nan_count]

  L = (nn[:, 0] < 1) | (nn[:, 0] > NA[0]) | (nn[:, 1] < 1) | (nn[:, 1] > NA[1]) | (nn[:, 2] < 1) | (nn[:, 2] > NA[2])
  nn = nn[~L, :]

  neighbors_list = np.column_stack((np.ravel_multi_index((nn[:, 0] - 1, nn[:, 1] - 1, nn[:, 2] - 1), NA), nn))

  neighbors_list = np.unique(neighbors_list, axis=0)
  neighbors_list = neighbors_list[~np.isin(neighbors_list[:, 0], nan_list[:, 0])]

  return neighbors_list















