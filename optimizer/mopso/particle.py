import numpy as np
from optimizer import Randomizer
from util import get_dominated

class Particle:
    """
    Represents a particle in the Multi-Objective Particle Swarm Optimization (MOPSO) algorithm.

    Parameters:
        lower_bound (numpy.ndarray): Lower bound for the particle's position.
        upper_bound (numpy.ndarray): Upper bound for the particle's position.
        num_objectives (int): Number of objectives in the optimization problem.

    Attributes:
        position (numpy.ndarray): Current position of the particle.
        num_objectives (int): Number of objectives in the optimization problem.
        velocity (numpy.ndarray): Current velocity of the particle.
        best_position (numpy.ndarray): Best position the particle has visited.
        best_fitness (numpy.ndarray): Best fitness values achieved by the particle.
        fitness (numpy.ndarray): Current fitness values of the particle.
    """

    def __init__(self, lower_bound, num_objectives, num_particles, id, topology):
        self.position = np.asarray(lower_bound)
        self.num_objectives = num_objectives
        self.num_particles = num_particles
        self.velocity = np.zeros_like(self.position)

        self.fitness = np.full(self.num_objectives, np.inf)
        self.local_best_fitnesses = []
        self.local_best_positions = []
        self.iterations_with_no_improvement = 0
        self.id = id
        self.topology = topology

    def update_velocity(self,
                        pareto_front,
                        crowding_distances,
                        inertia_weight=0.5,
                        cognitive_coefficient=1, 
                        social_coefficient=1):
        """
        Update the particle's velocity based on its best position and the global best position.

        Parameters:
            global_best_position (numpy.ndarray): Global best position in the swarm.
            inertia_weight (float): Inertia weight controlling the impact of the previous velocity
                                    (default is 0.5).
            cognitive_coefficient (float): Cognitive coefficient controlling the impact of personal
                                           best (default is 1).
            social_coefficient (float): Social coefficient controlling the impact of global best
                                        (default is 1).
        """
        leader = self.get_pareto_leader(pareto_front, crowding_distances)
        best_position = Randomizer.rng.choice(self.local_best_positions)
        cognitive_random = Randomizer.rng.uniform(0, 1)
        social_random = Randomizer.rng.uniform(0, 1)
        cognitive = cognitive_coefficient * cognitive_random * \
            (best_position - self.position)
        social = social_coefficient * social_random * \
            (leader.position - self.position)
        self.velocity = inertia_weight * self.velocity + cognitive + social

    def update_position(self, lower_bound, upper_bound):
        """
        Update the particle's position based on its current velocity and bounds.

        Parameters:
            lower_bound (numpy.ndarray): Lower bound for the particle's position.
            upper_bound (numpy.ndarray): Upper bound for the particle's position.
        """
        new_position = np.empty_like(self.position)
        for i in range(len(lower_bound)):
            if type(lower_bound[i]) == int or type(lower_bound[i]) == bool:
                new_position[i] = np.round(self.position[i] + self.velocity[i])
            else:
                new_position[i] = self.position[i] + self.velocity[i]
        self.position = np.clip(new_position, lower_bound, upper_bound)

    def set_fitness(self, fitness):
        """
        Set the fitness values of the particle and calls `update_best` method to update the
        particle's fitness and best position.

        Parameters:
            fitness (numpy.ndarray): The fitness values of the particle for each objective.
        """
        self.fitness = fitness
        self.update_best()

    def set_position(self, position):
        self.position = position

    def set_state(self, velocity, position, best_position, fitness, best_fitness):
        self.velocity = velocity
        self.position = position
        self.best_position = best_position
        self.fitness = fitness
        self.best_fitness = best_fitness

    def update_best(self):
        """
        Update particle's fitness and best position
        """
        len_fitness = len(self.local_best_fitnesses)
        fitnesses = np.array([f for f in self.local_best_fitnesses] + [self.fitness])
        positions = np.array([p for p in self.local_best_positions] + [self.position])

        dominated = get_dominated(fitnesses, len_fitness)

        new_local_best_fitnesses = [fitnesses[i] for i in range(len(fitnesses)) if not dominated[i]]
        
        # if the new fitness is different from the old one, reset the counter
        if np.array_equal(new_local_best_fitnesses, self.local_best_fitnesses):
            self.iterations_with_no_improvement += 1
        else:
            self.iterations_with_no_improvement = 0
        self.local_best_fitnesses = new_local_best_fitnesses
        self.local_best_positions = [positions[i] for i in range(len(positions)) if not dominated[i]]

    def get_pareto_leader(self, pareto_front, crowding_distances):
        if self.topology == "random":
            return Randomizer.rng.choice(pareto_front)
        elif self.topology == "lower_weighted_crowding_distance":
            return weighted_crowding_distance_topology(pareto_front, crowding_distances, higher = False)
        elif self.topology == "round_robin":
            return round_robin_topology(pareto_front, self.id)
        else:
            raise ValueError(
                f"MOPSO: {self.topology} not implemented!")

def weighted_crowding_distance_topology(pareto_front, crowding_distances, higher):
    pdf = boltzmann(crowding_distances, higher)
    return Randomizer.rng.choice(pareto_front, p = pdf)

def round_robin_topology(pareto_front, id):
    index = id % len(pareto_front)
    return pareto_front[index]

def boltzmann(crowding_distances, higher):
    cd_list = list(crowding_distances.values())
    len_cd = len(cd_list)
    if len_cd == 1: return [1]
    if len_cd == 2: return Randomizer.rng.choice([[1,0], [0, 1]])
    pdf = np.empty(len_cd)
    if higher: cd_list = [1 / (cd + 1e-9) for cd in cd_list]
    for i in range(len_cd):
        pdf[i] = np.exp(-cd_list[i])
    pdf = pdf / (np.sum(pdf))
    return pdf