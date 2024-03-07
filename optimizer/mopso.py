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
import copy
import numpy as np
from optimizer import Optimizer, FileManager, Randomizer
from numba import njit, jit


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
        leader = Randomizer.rng.choice(pareto_front[:int(self.num_particles)])
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
        self.position = np.clip(
            self.position + self.velocity, lower_bound, upper_bound)

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
    @njit
    def is_dominated(self, others):
        """
        Check if the particle is dominated by any other particle.

        Parameters:
            others (list): List of other particles.

        Returns:
            bool: True if the particle is dominated, False otherwise
        """
        for particle in others:
            if np.any(self.fitness > particle.fitness) and \
                np.all(self.fitness >= particle.fitness):
                return True
        return False


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
                 incremental_pareto=True, initial_particles_position='spread'):
        self.objective = objective
        if FileManager.loading_enabled:
            try:
                self.load_checkpoint()
                return
            except FileNotFoundError as e:
                print("Checkpoint not found. Fallback to standard construction.")
        self.num_particles = num_particles
        self.num_params = len(lower_bounds)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.inertia_weight = inertia_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.particles = [Particle(lower_bounds, upper_bounds, objective.num_objectives, num_particles)
                          for _ in range(num_particles)]
        VALID_INITIAL_PARTICLES_POSITIONS = {
            'spread', 'lower_bounds', 'upper_bounds', 'random'}

        if initial_particles_position == 'spread':
            self.spread_particles()
        elif initial_particles_position == 'lower_bounds':
            [particle.set_position(self.lower_bounds)
             for particle in self.particles]
        elif initial_particles_position == 'upper_bounds':
            [particle.set_position(self.upper_bounds)
             for particle in self.particles]
        elif initial_particles_position == 'random':
            [particle.set_position(Randomizer.rng.uniform(
                self.lower_bounds, self.upper_bounds)) for particle in self.particles]
        else:
            raise ValueError(
                f"MOPSO: initial_particles_position must be one of {VALID_INITIAL_PARTICLES_POSITIONS}")

        self.iteration = 0
        self.incremental_pareto = incremental_pareto
        self.pareto_front = []

    def spread_particles(self):
        mesh = np.meshgrid(*[np.linspace(l_b, u_b, num=int(self.num_particles**(1/self.num_params)))
                             for l_b, u_b in zip(self.lower_bounds, self.upper_bounds)])
        points = np.vstack([dim.flatten() for dim in mesh]).T
        [particle.set_position(point)
         for particle, point in zip(self.particles, points)]

    def save_attributes(self):
        """
        Save PSO attributes in a json file, which can be loaded later to continue the optimization
        from a checkpoint.

        Parameters:
            checkpoint_dir (str): Path to the folder where the json file is saved.
        """
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
        pso_attributes = FileManager.load_json(
            'checkpoint/pso_attributes.json')
        individual_states = FileManager.load_csv(
            'checkpoint/individual_states.csv')
        pareto_front = FileManager.load_csv('checkpoint/pareto_front.csv')

        # restore pso attributes
        self.lower_bounds = pso_attributes['lower_bounds']
        self.upper_bounds = pso_attributes['upper_bounds']
        self.num_particles = pso_attributes['num_particles']
        self.num_params = pso_attributes['num_params']
        self.inertia_weight = pso_attributes['inertia_weight']
        self.cognitive_coefficient = pso_attributes['cognitive_coefficient']
        self.social_coefficient = pso_attributes['social_coefficient']
        self.max_iter_no_improv = pso_attributes['max_iter_no_improv']
        self.optimization_mode = pso_attributes['optimization_mode']
        self.incremental_pareto = pso_attributes['incremental_pareto']
        self.iteration = pso_attributes['iteration']

        # restore particles
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

    def optimize(self, num_iterations=100, max_iter_no_improv=None):
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
        for _ in range(num_iterations):
            optimization_output = self.objective.evaluate(
                [particle.position for particle in self.particles])
            [particle.set_fitness(optimization_output[:, p_id])
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

        self.save_attributes()
        self.save_state()

        return self.pareto_front

    def update_pareto_front(self):
        """
        Update the Pareto front of non-dominated solutions across all iterations.
        """
        # Given the array of particles with n fitness values, pareto_fitness is an array with n rows of num_particles columns
        pareto_lenght = len(self.pareto_front)
        particles = self.pareto_front + self.particles
        particle_fitnesses = np.array([particle.fitness for particle in self.pareto_front]+ [particle.fitness for particle in self.particles])
        dominanted = get_dominated(particle_fitnesses, pareto_lenght)
        
        if self.incremental_pareto:
            self.pareto_front = [copy.deepcopy(particles[i]) for i in range(len(particles)) if not dominanted[i]]
        else:
            self.pareto_front = [copy.deepcopy(particles[i]) for i in range(pareto_lenght, len(particles)) if not dominanted[i]]
            
        crowding_distances = self.calculate_crowding_distance(
            self.pareto_front)
        self.pareto_front.sort(key=lambda x: crowding_distances[x], reverse=True)

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