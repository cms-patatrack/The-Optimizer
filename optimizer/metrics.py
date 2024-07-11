#from .mopso.mopso import get_dominated
from numba import njit, jit
import numpy as np
import copy


def hyper_volume(pareto_front, ref_point, real_pareto = None, hv_real = 1):
    # if real_pareto is not None: hv_real = wfg(real_pareto, ref_point)
    
    return wfg(sorted(pareto_front, key=lambda x: x[0]), ref_point) / hv_real

def wfg(pareto_front, ref_point):
    """
    Calculate the hypervolume for a given set of points.

    Parameters
    ----------
    pareto_front : list
        A list of points constituting the pareto front to be evaluated

    Returns
    -------
    float
        The hypervolume, which is the sum of exclusive hypervolumes of the provided points.
    """
    if len(pareto_front) == 0: 
        return 0
    else:
        # return np.sum([exclhv(pareto_front, k, ref_point) for k in range(len(pareto_front))])
        sum = 0
        for k in range(len(pareto_front)):
            sum = sum + exclhv(pareto_front, k, ref_point)
        return sum

def exclhv(pareto_front, k, ref_point):
    """
    The exclusive hypervolume of a point p relative to an underlying set S
    is the size of the part of objective space that is dominated by p but is 
    not dominated by any member of S
    """
    limited_set = limitset(pareto_front, k)
    result = inclhv(pareto_front[k], ref_point)
    if len(limited_set) > 0:
        result = result - wfg(nds(limited_set), ref_point)
    return  result

@njit
def inclhv(p, ref_point):
    volume = 1
    for i in range(len(p)):
        volume = volume * abs(p[i] - ref_point[i])
    return volume

@njit
def limitset(pareto_front, k):
    # n = 2 #? Dimension of each point in the pareto
    # ql = np.full((len(pareto_front) - k, n), -np.inf)
    # for i in range(1, len(pareto_front) - k):
    #     for j in range(1, n):
    #         ql[i][j] = np.max((pareto_front[k][j], pareto_front[k + i][j]))
    # ql = ql[1:][:,1]
    # return np.array([[max(p,q) for (p,q) in zip(pareto_front[k], pareto_front[j+k+1])]
    #         for j in range(len(pareto_front)-k-1)])

    m = len(pareto_front) - k - 1
    n = len(pareto_front[0])
    result = np.empty((m, n))
    for j in range(m):
        l = np.empty(n)
        for i in range(n):
            p = pareto_front[k][i]
            q = pareto_front[j+k+1][i]
            l[i] = p if p > q else q
        result[j] = l
    return result

@njit
def nds(front):
    """
    return the nondominated solutions from a set of points
    """
    # front = np.array(front)
    # if len(front) == 0:
    #     return np.array([], dtype=np.float64)
    if len(front) == 1:
        return front
    else:
        return front[np.invert(get_dominated(front, 0))]
     
@njit
def get_dominated(particles, pareto_lenght):
    dominated_particles = np.full(len(particles), False)
    for i in range(len(particles)):
        dominated = False
        for j in range(pareto_lenght, len(particles)):
            if np.any(particles[i] > particles[j]) and \
                    np.all(particles[i] >= particles[j]):
                dominated = True
                break
        dominated_particles[i] = dominated  
    return dominated_particles

def first(x):
    return x[0]

def stupid_hv(pareto_front, ref_point):
    # pareto_front_copy = copy.deepcopy(pareto_front)
    # pareto_front_copy = pareto_front_copy.sort(key=first)
    pareto_front_sorted = sorted(pareto_front, key=lambda x: x[0])
    hv = (ref_point[0] - pareto_front_sorted[0][0]) * (ref_point[1] - pareto_front_sorted[0][1])
    for i in range(1, len(pareto_front_sorted)):
        hv = hv + (ref_point[0] - pareto_front_sorted[i][0]) * (pareto_front_sorted[i - 1][1] - pareto_front_sorted[i][1])
    return hv