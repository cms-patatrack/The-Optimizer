import numpy as np
import math
from numba import njit, prange

# State functions

@njit
def distance_from_cluster(v, mod_v, pos, points, angle_deg, max_dist):
    angle_rad = angle_deg * np.pi / 180
    mask = np.full(len(points), False)
    num_points_inside = 0
    num_points = len(points)
    for i in prange(num_points):
        u = points[i] - pos
        mod_u = np.linalg.norm(u)
        if mod_u != 0: # if the point is in the pareto is also inside the cone
            angle = math.acos(round(np.dot(v, u) / (mod_v * mod_u), 2))
        else:
            angle = 0
        if angle < angle_rad:
            mask[i] = True
            num_points_inside = num_points_inside + 1
    
    if  num_points_inside > 1:
        # mean_position = [points[i] for i in prange(num_points) if mask[i]][np.random.randint(num_points_inside)]
        mean_position = np.empty(len(pos))
        for j in prange(len(points[0])):
            for i in prange(num_points):
                mean_position[j] = mean_position[j] + points[i][j]
            mean_position[j] = mean_position[j] / num_points                     
        return np.linalg.norm(pos - mean_position), num_points_inside
    
    elif num_points_inside == 1:
        for i in prange(num_points):
            if mask[i]: 
                return np.linalg.norm(pos - points[i]), 1
            
    else:
        return max_dist, 0
    
@njit
def sphere(position, radius, points_positions):
    mask = np.full(len(points_positions), False)
    for i, p in enumerate(points_positions): 
        if np.linalg.norm(position - p) < radius:
            mask[i] = True
    return np.sum(mask)

def observe_list(pso, good_points_positions, bad_points_positions, radius, max_dist, pso_iterations):
        
        # Try to include:
        # particle.num_skips
        # mean distance
        # progress
        observe_list = [[] for i in range(len(pso.particles))]

        positions = [p.position for p in pso.particles]
        progress = pso.iteration / pso_iterations
        
        for i, particle in enumerate(pso.particles):
            bad_points_in_sphere = sphere(particle.position, radius, bad_points_positions) if len(bad_points_positions) > 0 else 0
            good_points_in_sphere = sphere(particle.position, radius, good_points_positions) if len(good_points_positions) > 0 else 0

            distance = np.linalg.norm(positions - positions[i], axis=1)
            mean_distance = np.sum(distance) / (len(pso.particles) - 1) / max_dist

            particle_observation = [
                        bad_points_in_sphere,
                        good_points_in_sphere,
                        particle.iterations_with_no_improvement,
                        mean_distance,
                        progress,
                        # distance_good_points,
                        # num_good_points,
                        # distance_bad_points,
                        # num_bad_points,
                        # # distance_best,
                        # particle.iteration_from_best_position,
                        # particle.num_skips,
                        # # progress
                    ]           
            observe_list[i] = particle_observation
        return observe_list

# Other utilities
def find_new_bad_points(particles, dominated_list, actions):
    new_bad_points = []
    for i in range(len(dominated_list)):
        if dominated_list[i] and actions[i]:
            new_bad_points.append(particles[i].position.copy())
    return new_bad_points