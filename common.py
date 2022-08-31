from typing import Callable
from numpy import zeros, inf, ndarray, linalg


def dtw(x : ndarray, y : ndarray, dist : Callable) -> float:
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    x: N*K array
    y: M*K array
    dist: distance used as cost measure
    
    returns dtw distance.
    """
    
    n, m = len(x), len(y)
    
    D0 = zeros((n + 1, m + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    
    for i in range(n):
        for j in range(m):
            D1[i, j] = dist(x[i], y[j])
    
    for i in range(n):
        for j in range(m):
            min_list = [D0[i, j]]
            i_k = min(i + 1, n)
            j_k = min(j + 1, m)
            min_list += [D0[i_k, j], D0[i, j_k]]
            D1[i, j] += min(min_list)
            
    return D1[-1, -1]


def epsilon_norm(x : ndarray, y : ndarray, epsilon : float = 168.0) -> float:
    if linalg.norm(x - y) < epsilon:
        return 0
    else:
        return linalg.norm(x-y)
