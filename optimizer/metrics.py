#from .mopso.mopso import get_dominated
import numpy as np

def hyper_volume(pareto_front, ref_point, real_pareto = None):
    hv_real = wfg(real_pareto, ref_point) if real_pareto is not None else 1
    return wfg(pareto_front, ref_point) / hv_real
    
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
    return sum([exclhv(pareto_front, k, ref_point) for k in range(len(pareto_front))])

def exclhv(pareto_front, k, ref_point):
    """
    The exclusive hypervolume of a point p relative to an underlying set S
    is the size of the part of objective space that is dominated by p but is 
    not dominated by any member of S
    """
    return inclhv(pareto_front[k], ref_point) - wfg(nds(limitset(pareto_front, k)), ref_point)

def inclhv(p, ref_point):
    volume = 1
    for i in range(len(p)):
        volume *= abs(p[i] - ref_point[i])
    return volume

def limitset(pareto_front, k):
    # n = 2 #? Dimension of each point in the pareto
    # ql = np.full((len(pareto_front) - k, n), -np.inf)
    # for i in range(1, len(pareto_front) - k):
    #     for j in range(1, n):
    #         ql[i][j] = np.max((pareto_front[k][j], pareto_front[k + i][j]))
    # ql = ql[1:][:,1]
    return [[max(p,q) for (p,q) in zip(pareto_front[k], pareto_front[j+k+1])]
            for j in range(len(pareto_front)-k-1)]

def nds(front):
    """
    return the nondominated solutions from a set of points
    """
    # archive = []

    # for row in front:
    #     asize = len(archive)
    #     ai = -1
    #     while ai < asize - 1:
    #         ai += 1
    #         adominate = False
    #         sdominate = False
    #         nondominate = False
    #         for arc, sol in zip(archive[ai], row):
    #             if arc < sol:
    #                 adominate = True
    #                 if sdominate:
    #                     nondominate = True
    #                     break # stop comparing objectives
    #             elif arc > sol:
    #                 sdominate = True
    #                 if adominate:
    #                     nondominate = True
    #                     break # stop comparing objectives
    #         if nondominate:
    #             continue # compare next archive solution
    #         if adominate:
    #             break    # row is dominated
    #         if sdominate:
    #             archive.pop(ai)
    #             ai -= 1
    #             asize -= 1
    #             continue # compare next archive solution
    #     # if the solution made it all the way through, keep it
    #     archive.append(row)
    # return archive
    return np.array(front)[np.invert(get_dominated(front, 0))]

def get_dominated(particles, pareto_lenght):
    particles = np.array(particles)
    dominated_particles = np.full(len(particles), False, dtype=bool)
    for i in range(len(particles)):
        dominated = False
        for j in range(pareto_lenght, len(particles)):
            if np.any(particles[i] > particles[j]) and \
                    np.all(particles[i] >= particles[j]):
                dominated = True
                break
        dominated_particles[i] = dominated

    return dominated_particles
