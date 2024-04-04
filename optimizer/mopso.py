"""
mopso.py

This module implements the Multi-Objective Particle Swarm Optimization (MOPSO) algorithm,
a heuristic optimization algorithm used to find the Pareto front of non-dominated solutions
for multi-objective optimization problems.

The MOPSO algorithm works by maintaining a population of particles, each representing a potential
solution to the optimization problem. The particles are iteratively updated by considering their
velocity and position in the search space, aiming to converge towards the Pareto front of the
problem's solution space.

This module contains two classes:
    - Particle: Represents a particle in the MOPSO algorithm.
    - MOPSO: Multi-Objective Particle Swarm Optimization algorithm.

Both classes are designed to be used in conjunction to perform the MOPSO optimization process and
find the Pareto front of non-dominated solutions.
"""
from copy import copy
import itertools
import math
import numpy as np
import math
from optimizer import Optimizer, FileManager, Randomizer, Logger
from numba import njit, jit
import scipy.stats as stats


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

    def __init__(self, lower_bound, upper_bound, num_objectives, num_particles):
        self.position = np.asarray(lower_bound)
        self.num_objectives = num_objectives
        self.num_particles = num_particles
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position
        self.best_fitness = np.array([np.inf]*self.num_objectives)
        self.fitness = np.ones(self.num_objectives)

    def update_velocity(self,
                        pareto_front, inertia_weight=0.5,
                        cognitive_coefficient=1, social_coefficient=1,):
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
        leader = Randomizer.rng.choice(pareto_front)
        cognitive_random = Randomizer.rng.uniform(0, 1)
        social_random = Randomizer.rng.uniform(0, 1)
        cognitive = cognitive_coefficient * cognitive_random * \
            (self.best_position - self.position)
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
        # if np.any(self.best_fitness == np.zeros(self.num_objectives)):
        #     self.fitness = np.ones(self.num_objectives)
        if np.all(self.fitness <= self.best_fitness):
            self.best_fitness = self.fitness
            self.best_position = self.position


class MOPSO(Optimizer):
    """
    Multi-Objective Particle Swarm Optimization (MOPSO) algorithm.

    Parameters:
        objective_functions (list): List of objective functions to be minimized.
        lower_bound (numpy.ndarray): Lower bound for the particles' positions.
        upper_bound (numpy.ndarray): Upper bound for the particles' positions.
        num_particles (int): Number of particles in the swarm (default is 50).
        inertia_weight (float): Inertia weight controlling the impact of the previous velocity
                                (default is 0.5).
        cognitive_coefficient (float): Cognitive coefficient controlling the impact of personal
                                       best (default is 1).
        social_coefficient (float): Social coefficient controlling the impact of global best
                                    (default is 1).
        num_iterations (int): Number of iterations for the optimization process (default is 100).
        optimization_mode (str): Mode for updating particle fitness: 'individual' or 'global'
                                 (default is 'individual').
        max_iter_no_improv (int): Maximum number of iterations without improvement
                                  (default is None).
        num_objectives (int): Number of objectives in the optimization problem (default is None,
                              calculated from objective_functions).
        checkpoint_dir (str): Path to the folder where the a checkpoint is saved (optional). 
                              If this is specified, the checkpoint will be restored 
                              and the optimization will continue from this point.

    Attributes:
        objective_functions (list): List of objective functions to be minimized.
        num_objectives (int): Number of objectives in the optimization problem.
        num_particles (int): Number of particles in the swarm.
        num_params (int): Number of parameters to optimize
        lower_bounds (numpy.ndarray): Lower bound for the particles' positions.
        upper_bounds (numpy.ndarray): Upper bound for the particles' positions.
        inertia_weight (float): Inertia weight controlling the impact of the previous velocity.
        cognitive_coefficient (float): Cognitive coefficient controlling the impact of
                                       personal best.
        social_coefficient (float): Social coefficient controlling the impact of global best.
        num_iterations (int): Number of iterations for the optimization process.
        max_iter_no_improv (int): Maximum number of iterations without improvement.
        optimization_mode (str): Mode for updating particle fitness: 'individual' or 'global'.
        particles (list): List of Particle objects representing the swarm.
        global_best_position (numpy.ndarray): Global best position in the swarm.
        global_best_fitness (list): Global best fitness values achieved in the swarm.
        iteration (int): the current iteration
        pareto_front (list): List of Particle objects representing the Pareto front of non-dominated 
                             solutions across all iterations.
        history (list): List to store the global best fitness values at each iteration.

    Methods:
        optimize():
            Perform the MOPSO optimization process and return the Pareto front of non-dominated
            solutions.
        update_global_best():
            Update the global best position and fitness based on the swarm's particles.
        get_pareto_front():
            Get the Pareto front of non-dominated solutions.
        calculate_crowding_distance(pareto_front):
            Calculate the crowding distance for particles in the Pareto front.
    """

    def __init__(self,
                 objective,
                 lower_bounds, upper_bounds, num_particles=50,
                 inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=1,
                 incremental_pareto=True, initial_particles_position='random', default_point=None):
        self.objective = objective
        if FileManager.loading_enabled:
            try:
                self.load_checkpoint()
                return
            except FileNotFoundError as e:
                Logger.warning("Checkpoint not found. Fallback to standard construction.")
        else:
            Logger.debug("Loading disabled. Starting standard construction.")
        self.num_particles = num_particles

        if len(lower_bounds) != len(upper_bounds):
            Logger.warning(f"Warning: lower_bounds and upper_bounds have different lengths."
                          f"The lowest length ({min(len(lower_bounds), len(upper_bounds))}) is taken.")
        self.num_params = min(len(lower_bounds), len(upper_bounds))
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.check_types()

        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.particles = [Particle(lower_bounds, upper_bounds, objective.num_objectives, num_particles)
                          for _ in range(num_particles)]
        VALID_INITIAL_PARTICLES_POSITIONS = {
            'spread', 'lower_bounds', 'upper_bounds', 'random', 'gaussian'}

        Logger.debug(f"Setting initial particles position")
        
        if initial_particles_position == 'spread':
            Logger.warning(f"Initial distribution forced to 'random'.")
            initial_particles_position = 'random'
            # self.spread_particles()

        if initial_particles_position == 'lower_bounds':
            [particle.set_position(self.lower_bounds)
             for particle in self.particles]
        elif initial_particles_position == 'upper_bounds':
            [particle.set_position(self.upper_bounds)
             for particle in self.particles]
        elif initial_particles_position == 'random':
            def random_position():
                positions = []
                for i in range(self.num_params):
                    if type(self.lower_bounds[i]) == int:
                        position = Randomizer.rng.integers(
                            self.lower_bounds[i], self.upper_bounds[i])
                    elif type(self.lower_bounds[i]) == float:
                        position = Randomizer.rng.uniform(
                            self.lower_bounds[i], self.upper_bounds[i])
                    elif type(self.lower_bounds[i]) == bool:
                        position = Randomizer.rng.choice([True, False])
                    else:
                        raise ValueError(
                            f"Type {type(self.lower_bounds[i])} not supported")
                    positions.append(position)
                return np.array(positions)

            [particle.set_position(random_position())
             for particle in self.particles]

        elif initial_particles_position == 'gaussian':
            if default_point is None:
                default_point = np.mean(
                    [self.lower_bounds, self.upper_bounds], axis=0)
            else:
                default_point = np.array(default_point)
            # See scipy truncnorm rvs documentation for more information
            a_trunc = np.array(self.lower_bounds)
            b_trunc = np.array(self.upper_bounds)
            loc = default_point
            scale = (np.array(self.upper_bounds) -
                     np.array(self.lower_bounds))/4
            a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
            [particle.set_position(stats.truncnorm.rvs(a, b, loc, scale))
             for particle in self.particles]
            
            for particle in self.particles:
                for i in range(self.num_params):
                    if type(lower_bounds[i]) == int or type(lower_bounds[i]) == bool:
                        particle.position[i] = int(round(particle.position[i]))

        elif initial_particles_position not in VALID_INITIAL_PARTICLES_POSITIONS:
            raise ValueError(
                f"MOPSO: initial_particles_position must be one of {VALID_INITIAL_PARTICLES_POSITIONS}")

        if default_point is not None:
            self.particles[0].set_position(default_point)
            
        self.iteration = 0
        self.incremental_pareto = incremental_pareto
        self.pareto_front = []

    def check_types(self):
        lb_types = [type(lb) for lb in self.lower_bounds]
        ub_types = [type(ub) for ub in self.upper_bounds]

        acceptable_types = [int, float, bool]

        for i in range(self.num_params):
            if lb_types[i] not in acceptable_types:
                raise ValueError(f"Lower bound type {lb_types[i]} for "
                                 f"Lower bound {i} is not acceptable")
            if ub_types[i] not in acceptable_types:
                raise ValueError(f"Upper bound type {ub_types[i]} for "
                                 f"Upper bound {i} is not acceptable")

        if lb_types != ub_types:
            Logger.warning(
                "lower_bounds and upper_bounds are of different types")
            Logger.warning("Keeping the least restrictive type")
            for i in range(self.num_params):
                if lb_types[i] == float or ub_types[i] == float:
                    self.lower_bounds[i] = float(self.lower_bounds[i])
                    self.upper_bounds[i] = float(self.upper_bounds[i])
                elif lb_types[i] == int or ub_types[i] == int:
                    self.upper_bounds[i] = int(self.upper_bounds[i])
                    self.lower_bounds[i] = int(self.lower_bounds[i])

    def insert_nodes(self, param_list, is_bool=False):
        indices = [i for i in range(len(param_list) - 1)]
        is_int = any(isinstance(x, int) for x in param_list)
        is_float = any(isinstance(x, float) for x in param_list)
        if is_float:
            new_values = [(param_list[idx] + param_list[idx + 1]
                           ) / 2 for idx in indices]
        elif is_int:
            new_values = [math.floor(
                (param_list[idx] + param_list[idx + 1]) / 2) for idx in indices]
        for new_value in new_values:
            for idx, val in enumerate(param_list[:-1]):
                if val <= new_value < param_list[idx + 1]:
                    if is_bool:
                        param_list.insert(idx + 1, bool(new_value))
                    else:
                        param_list.insert(idx + 1, new_value)
                    break
        return param_list

    def get_nodes(self):
        
        def ndcube(*args):
            return list(itertools.product(*map(lambda x: [x[0], x[1]], args)))
        
        bounds = list(zip(self.lower_bounds, self.upper_bounds))
        all_nodes = ndcube(*bounds)
        print(len(all_nodes))
        exit()
        indices_with_bool = [idx for idx, node in enumerate(
            all_nodes) if any(isinstance(val, bool) for val in node)]
        all_nodes = [[2 if isinstance(val, bool) and val else 0 if isinstance(
            val, bool) and not val else val for val in node] for node in all_nodes]

        if self.num_particles < self.num_params:
            Logger.warning(f"Warning: not enough particles, now you are running with {len(all_nodes[0])} particles")

        particle_count = len(all_nodes[0])
        while particle_count < self.num_particles:
            for idx in range(self.num_params):
                nodes = all_nodes[idx]
                len_before = len(nodes)
                if idx in indices_with_bool:
                    nodes = self.insert_nodes(nodes, is_bool=True)
                else:
                    nodes = self.insert_nodes(nodes)
                len_after = len(nodes)
                particle_count += (len_after - len_before) / self.num_params
        for idx in indices_with_bool:
            all_nodes[idx][0] = False
            all_nodes[idx][len(all_nodes[idx]) - 1] = True
        combinations = itertools.product(*all_nodes)
        return np.array([np.array(combo, dtype=object) for combo in combinations])

    def spread_particles(self):
        positions = self.get_nodes()
        np.random.shuffle(positions)
        [particle.set_position(point) for particle,
         point in zip(self.particles, positions)]

    def save_attributes(self):
        """
        Save PSO attributes in a json file, which can be loaded later to continue the optimization
        from a checkpoint.

        Parameters:
            checkpoint_dir (str): Path to the folder where the json file is saved.
        """
        Logger.debug("Saving PSO attributes")
        pso_attributes = {
            'lower_bounds': self.lower_bounds,
            'upper_bounds': self.upper_bounds,
            'num_particles': self.num_particles,
            'num_params': self.num_params,
            'inertia_weight': self.inertia_weight,
            'cognitive_coefficient': self.cognitive_coefficient,
            'social_coefficient': self.social_coefficient,
            'incremental_pareto': self.incremental_pareto,
            'iteration': self.iteration
        }
        FileManager.save_json(pso_attributes, "checkpoint/pso_attributes.json")

    def save_state(self):
        """
        Save the current state of PSO in csv files, which can be loaded later to continue the 
        optimization from a checkpoint.
        There will be 4 files:
            - individual_states.csv: Containing the current state (position, velocity, 
                                     best_position, best_fitness) of each particle
            - global_states.csv: Containing the current state (global_best_position, 
                                 global_best_fitness, iteration) of PSO.
            - pareto_front.csv: Containing non-dominated solutions up until this point
            - history.csv: Containing the best global fitness each iteration              

        Parameters:
            checkpoint_dir (str): Path to the folder where the csv files are saved.
        """
        Logger.debug("Saving PSO state")
        FileManager.save_csv([np.concatenate([particle.position,
                                              particle.velocity,
                                              particle.best_position,
                                              np.ravel(particle.best_fitness)])
                             for particle in self.particles],
                             'checkpoint/individual_states.csv')

        FileManager.save_csv([np.concatenate([particle.position, np.ravel(particle.fitness)])
                             for particle in self.pareto_front],
                             'checkpoint/pareto_front.csv')

    def load_checkpoint(self):
        """
        Load a checkpoint in order to continue a previous run.            

        Parameters:
            checkpoint_dir (str): Path to the folder where the checkpoint is saved.
            num_additional_iterations: Number of additional iterations to run. 
        """
        # load saved data
        Logger.debug("Loading checkpoint")
        
        pso_attributes = FileManager.load_json(
            'checkpoint/pso_attributes.json')
        individual_states = FileManager.load_csv(
            'checkpoint/individual_states.csv')
        pareto_front = FileManager.load_csv('checkpoint/pareto_front.csv')

        # restore pso attributes
        Logger.debug("Restoring PSO attributes")
        self.lower_bounds = pso_attributes['lower_bounds']
        self.upper_bounds = pso_attributes['upper_bounds']
        self.num_particles = pso_attributes['num_particles']
        self.num_params = pso_attributes['num_params']
        self.inertia_weight = pso_attributes['inertia_weight']
        self.cognitive_coefficient = pso_attributes['cognitive_coefficient']
        self.social_coefficient = pso_attributes['social_coefficient']
        self.incremental_pareto = pso_attributes['incremental_pareto']
        self.iteration = pso_attributes['iteration']

        # restore particles
        Logger.debug("Restoring particles")
        self.particles = []
        for i in range(self.num_particles):
            particle = Particle(self.lower_bounds, self.upper_bounds,
                                num_objectives=self.objective.num_objectives,
                                num_particles=self.num_particles)
            particle.set_state(
                position=np.array(
                    individual_states[i][:self.num_params], dtype=float),
                velocity=np.array(
                    individual_states[i][self.num_params:2*self.num_params], dtype=float),
                best_position=np.array(
                    individual_states[i][2*self.num_params:3*self.num_params], dtype=float),
                fitness=[np.inf] * self.objective.num_objectives,
                best_fitness=np.array(
                    individual_states[i][3*self.num_params:], dtype=float)
            )
            self.particles.append(particle)

        # restore pareto front
        Logger.debug("Restoring pareto front")
        self.pareto_front = []
        for i in range(len(pareto_front)):
            particle = Particle(self.lower_bounds, self.upper_bounds,
                                num_objectives=self.objective.num_objectives,
                                num_particles=self.num_particles)
            particle.set_state(position=pareto_front[i][:self.num_params],
                               fitness=pareto_front[i][self.num_params:],
                               velocity=None,
                               best_position=None,
                               best_fitness=None)
            self.pareto_front.append(particle)

    def optimize(self, num_iterations=100):
        """
        Perform the MOPSO optimization process and return the Pareto front of non-dominated
        solutions. If `history_dir` is specified, the position and fitness of all particles 
        are saved every iteration. If `checkpoint_dir` is specified, a checkpoint will be 
        saved at the end. The checkpoint can be loaded later to continue the optimization.

        Parameters:
            history_dir (str): Path to the folder where history is saved (optional).
            checkpoint_dir (str): Path to the folder where checkpoint is saved (optional).

        Returns:
            list: List of Particle objects representing the Pareto front of non-dominated solutions.
        """
        Logger.info(f"Starting MOPSO optimization with {num_iterations} iterations")
        for _ in range(num_iterations):
            Logger.debug(f"Iteration {self.iteration}")
            optimization_output = self.objective.evaluate(
                [particle.position for particle in self.particles])
            [particle.set_fitness(optimization_output[p_id])
             for p_id, particle in enumerate(self.particles)]
            FileManager.save_csv([np.concatenate([particle.position, np.ravel(
                                 particle.fitness)]) for particle in self.particles],
                                 'history/iteration' + str(self.iteration) + '.csv')

            self.update_pareto_front()

            for particle in self.particles:
                particle.update_velocity(self.pareto_front,
                                         self.inertia_weight,
                                         self.cognitive_coefficient,
                                         self.social_coefficient)
                particle.update_position(self.lower_bounds, self.upper_bounds)

            self.iteration += 1
        Logger.info("MOPSO optimization finished")
        self.save_attributes()
        self.save_state()

        return self.pareto_front

    def update_pareto_front(self):
        """
        Update the Pareto front of non-dominated solutions across all iterations.
        """
        # Given the array of particles with n fitness values, pareto_fitness is an array with n rows of num_particles columns
        Logger.debug("Updating Pareto front")
        pareto_lenght = len(self.pareto_front)
        particles = self.pareto_front + self.particles
        particle_fitnesses = np.array([particle.fitness for particle in particles])
        dominanted = get_dominated(particle_fitnesses, pareto_lenght)

        if self.incremental_pareto:
            self.pareto_front = [copy(particles[i]) for i in range(
                len(particles)) if not dominanted[i]]
        else:
            self.pareto_front = [copy(particles[i]) for i in range(
                pareto_lenght, len(particles)) if not dominanted[i]]
        Logger.debug(f"New pareto front size: {len(self.pareto_front)}")
        crowding_distances = self.calculate_crowding_distance(
            self.pareto_front)
        self.pareto_front.sort(
            key=lambda x: crowding_distances[x], reverse=True)

    def calculate_crowding_distance(self, pareto_front):
        """
        Calculate the crowding distance for particles in the Pareto front.

        Parameters:
            pareto_front (list): List of Particle objects representing the Pareto front.

        Returns:
            dict: A dictionary with Particle objects as keys and their corresponding crowding
                  distances as values.
        """
        if len(pareto_front) == 0:
            return []

        num_objectives = len(np.ravel(pareto_front[0].fitness))
        distances = [0] * len(pareto_front)
        point_to_distance = {}

        for i in range(num_objectives):
            # Sort by objective i
            sorted_front = sorted(
                pareto_front, key=lambda x: np.ravel(x.fitness)[i])

            # Set the boundary points to infinity
            distances[0] = float('inf')
            distances[-1] = float('inf')

            # Normalize the objective values for calculation
            min_obj = np.ravel(sorted_front[0].fitness)[i]
            max_obj = np.ravel(sorted_front[-1].fitness)[i]
            norm_denom = max_obj - min_obj if max_obj != min_obj else 1

            for j in range(1, len(pareto_front) - 1):
                distances[j] += (np.ravel(sorted_front[j + 1].fitness)[i] -
                                 np.ravel(sorted_front[j - 1].fitness)[i]) / norm_denom

        for i, point in enumerate(pareto_front):
            point_to_distance[point] = distances[i]

        return point_to_distance


@njit
def get_dominated(particles, pareto_lenght):
    dominated_particles = np.zeros(len(particles))
    for i in range(len(particles)):
        dominated = False
        for j in range(pareto_lenght, len(particles)):
            if np.any(particles[i] > particles[j]) and \
                    np.all(particles[i] >= particles[j]):
                dominated = True
                break
        dominated_particles[i] = dominated
    return dominated_particles
