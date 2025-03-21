import numpy as np
from .util import njit, get_dominated

# Generational distance
def generational_distance(pareto_front, reference_front):
    """
    This function calculates the generational distance metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    generational_distance : float
        The generational distance metric value.
    """
    return np.mean(np.min(np.linalg.norm(pareto_front - reference_front, axis=1), axis=0))

# Inverted generational distance
def inverted_generational_distance(pareto_front, reference_front):
    """
    This function calculates the inverted generational distance metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_front : numpy array
        Represents the true pareto front.
    Returns:
    inverted_generational_distance : float
        The inverted generational distance metric value.
    """
    return np.mean(np.min(np.linalg.norm(reference_front - pareto_front, axis=1), axis=0))

# Hypervolume
def hypervolume_indicator(pareto_front, reference_point, reference_hv=1):
    """
    This function calculates the hypervolume indicator metric, for any dimension of the pareto front.
    Parameters:
    pareto_front : numpy array
        Represents the pareto front obtained from the optimization algorithm.
    reference_point : numpy array
        Represents the reference point for the hypervolume calculation.
    Returns:
    hypervolume : float
        The hypervolume indicator metric value.
    """
    return wfg(sorted(pareto_front, key=lambda x: x[0]), reference_point)/reference_hv


def wfg(pareto_front, reference_point):
    if len(pareto_front) == 0: 
        return 0
    else:
        # return np.sum([exclhv(pareto_front, k, reference_point) for k in range(len(pareto_front))])
        sum = 0
        for k in range(len(pareto_front)):
            sum = sum + exclhv(pareto_front, k, reference_point)
        return sum

def exclhv(pareto_front, k, reference_point):
    """
    The exclusive hypervolume of a point p relative to an underlying set S
    is the size of the part of objective space that is dominated by p but is 
    not dominated by any member of S
    """
    limited_set = limitset(pareto_front, k)
    result = inclhv(pareto_front[k], reference_point)
    if len(limited_set) > 0:
        result = result - wfg(nds(limited_set), reference_point)
    return  result

@njit
def inclhv(p, reference_point):
    volume = 1
    for i in range(len(p)):
        volume = volume * abs(p[i] - reference_point[i])
    return volume

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
