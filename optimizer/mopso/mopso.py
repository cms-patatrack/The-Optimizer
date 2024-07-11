from copy import copy
import itertools
import math
import numpy as np
import math
from optimizer import Optimizer, FileManager, Randomizer, Logger
import scipy.stats as stats
from .particle import Particle
from util import get_dominated

class MOPSO(Optimizer):
    def __init__(self,
                 objective,
                 lower_bounds, upper_bounds, num_particles=50,
                 inertia_weight=0.5, cognitive_coefficient=1, social_coefficient=1,
                 incremental_pareto=True, initial_particles_position='random', default_point=None,
                 exploring_particles=False, topology = 'random'):
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
        self.particles = [Particle(lower_bounds, objective.num_objectives, num_particles)
                          for _ in range(num_particles)]
        
        self.exploring_particles = exploring_particles
        VALID_INITIAL_PARTICLES_POSITIONS = {
            'spread', 'lower_bounds', 'upper_bounds', 'random', 'gaussian'}
        
        VALID_TOPOLOGIES = {
            'random', 'higher_crowding_distance', 'lower_crowding_distance', 'higher_weighted_crowding_distance',
            'lower_weighted_crowding_distance', 'round_robin', 'higher_crowding_distance_random_ring', 
            'lower_crowding_distance_random_ring'}

        if topology not in VALID_TOPOLOGIES:
            raise ValueError(
                f"MOPSO: topology must be one of {VALID_TOPOLOGIES}")

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
        # Randomizer.rng = np.random.default_rng(seed)  
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
        Logger.debug("Saving PSO state")
        FileManager.save_csv([np.concatenate([particle.position,
                                              particle.velocity])
                             for particle in self.particles],
                             'checkpoint/individual_states.csv')

        FileManager.save_csv([np.concatenate([particle.position, np.ravel(particle.fitness)])
                             for particle in self.pareto_front],
                             'checkpoint/pareto_front.csv')

    def load_checkpoint(self):
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

    def step(self):
        Logger.debug(f"Iteration {self.iteration}")
        optimization_output = self.objective.evaluate(
            [particle.position for particle in self.particles])
        [particle.set_fitness(optimization_output[p_id])
            for p_id, particle in enumerate(self.particles)]
        FileManager.save_csv([np.concatenate([particle.position, np.ravel(
                                particle.fitness)]) for particle in self.particles],
                                'history/iteration' + str(self.iteration) + '.csv')

        crowding_distances = self.update_pareto_front()

        for particle in self.particles:
            particle.update_velocity(self.pareto_front,
                                     crowding_distances,
                                     self.inertia_weight,
                                     self.cognitive_coefficient,
                                     self.social_coefficient)
            if self.exploring_particles and particle.iterations_with_no_improvement > 10:
                self.scatter_particle(particle)
            particle.update_position(self.lower_bounds, self.upper_bounds)

        self.iteration += 1
        
    def optimize(self, num_iterations=100, max_iter_no_improv=None):
        Logger.info("Starting MOPSO optimization")
        for _ in range(num_iterations):
            self.step()
        Logger.info("MOPSO optimization finished")

        self.save_attributes()
        self.save_state()

        return self.pareto_front

    def update_pareto_front(self):
        Logger.debug("Updating Pareto front")
        pareto_lenght = len(self.pareto_front)
        particles = self.pareto_front + self.particles
        particle_fitnesses = np.array(
            [particle.fitness for particle in particles])
        dominanted = get_dominated(particle_fitnesses, pareto_lenght)
        
        self.pareto_front =[copy(particles[i]) for i in range(len(particles)) if not dominanted[i]]
        crowding_distances = self.calculate_crowding_distance(self.pareto_front)
        self.pareto_front.sort(key=lambda x: crowding_distances[x], reverse=True)

        max_pareto_len = 500
        self.pareto_front = self.pareto_front[: max_pareto_len]
        Logger.debug(f"New pareto front size: {len(self.pareto_front)}")

        crowding_distances = self.calculate_crowding_distance(self.pareto_front)
        return crowding_distances
    
    def calculate_crowding_distance(self, pareto_front):
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

    def scatter_particle(self, particle: Particle):
        Logger.debug(f"Particle {particle} did not improve for 10 iterations. Scattering.")
        for i in range(len(self.lower_bounds)):
            lower_count = sum([1 for p in self.particles if p.position[i] < particle.position[i]])
            upper_count = sum([1 for p in self.particles if p.position[i] > particle.position[i]])
            if lower_count > upper_count:
                particle.velocity[i] = 1
            else:
                particle.velocity[i] = -1



