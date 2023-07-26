import numpy as np


class Particle:
    def __init__(self, lb=-10, ub=10, num_objectives=2):
        self.position = np.random.uniform(lb, ub)
        self.num_objectives = num_objectives
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position
        self.best_fitness = np.ones(self.num_objectives)
        self.fitness = np.ones(self.num_objectives)

    def update_velocity(self, global_best_position, w=0.5, c1=1, c2=1):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, lb, ub):
        self.position = np.clip(self.position + self.velocity, lb, ub)

    def set_fitness(self, fitness):
        self.fitness = np.array(fitness)
        if np.any(self.best_fitness == np.zeros(self.num_objectives)):
            self.fitness = np.ones(self.num_objectives)
        if np.all(self.fitness < self.best_fitness):
            self.best_fitness = self.fitness
            self.best_position = self.position

    def evaluate_fitness(self, objective_functions):
        fitness = [obj_func(self.position) for obj_func in objective_functions]
        self.set_fitness(fitness)


class MOPSO:
    def __init__(self, objective_functions,
                 lb, ub, num_particles=50,
                 w=0.5, c1=1, c2=1,
                 num_iterations=100,
                 optimization_mode='individual',
                 max_iter_no_improv=None,
                 num_objectives=None):
        self.objective_functions = objective_functions
        if num_objectives == None:
            self.num_objectives = len(self.objective_functions)
        else:
            self.num_objectives = num_objectives
        self.num_particles = num_particles
        self.lower_bounds = lb
        self.upper_bounds = ub
        self.inertia_weight = w
        self.cognitive_coefficient = c1
        self.social_coefficient = c2
        self.num_iterations = num_iterations
        self.max_iter_no_improv = max_iter_no_improv
        self.optimization_mode = optimization_mode
        self.particles = [Particle(lb, ub, self.num_objectives)
                          for _ in range(num_particles)]
        self.global_best_position = np.zeros_like(lb)
        self.global_best_fitness = [np.inf] * self.num_objectives
        self.history = []

    def optimize(self):
        for i in range(self.num_iterations):
            if self.optimization_mode == 'global':
                optimization_output = [objective_function([particle.position for particle in self.particles])
                                       for objective_function in self.objective_functions]
            for particle in self.particles:
                if self.optimization_mode == 'individual':
                    particle.evaluate_fitness(self.objective_functions)
                if self.optimization_mode == 'global':
                    particle.set_fitness([output[i]
                                         for output in optimization_output])

                for i, particle in enumerate(self.particles):
                    if np.all(particle.fitness < self.global_best_fitness):
                        self.global_best_fitness = particle.fitness
                        self.global_best_position = particle.position

                    self.update_particle_best(particle)
                    self.update_global_best()

                    particle.update_velocity(self.global_best_position,
                                             self.inertia_weight,
                                             self.cognitive_coefficient,
                                             self.social_coefficient)
                    particle.update_position(
                        self.lower_bounds, self.upper_bounds)

            self.history.append(self.global_best_fitness)

        pareto_front = self.get_pareto_front()
        return pareto_front

    def update_particle_best(self, particle):
        if np.any(particle.fitness < particle.best_fitness):
            particle.best_fitness = particle.fitness
            particle.best_position = particle.position

    def update_global_best(self):
        for particle in self.particles:
            if np.all(particle.fitness <= self.global_best_fitness):
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position

    def get_pareto_front(self):
        pareto_front = []
        for particle in self.particles:
            dominated = False
            for other_particle in self.particles:
                if np.all(particle.fitness >= other_particle.fitness) and np.any(particle.fitness > other_particle.fitness):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(particle)
        # Sort the Pareto front by crowding distance
        crowding_distances = self.calculate_crowding_distance(pareto_front)
        pareto_front.sort(key=lambda x: crowding_distances[x], reverse=True)
        return pareto_front

    def calculate_crowding_distance(self, pareto_front):
        crowding_distances = {particle: 0 for particle in pareto_front}
        for objective_index in range(self.num_objectives):
            # Sort the Pareto front by the current objective function
            pareto_front_sorted = sorted(
                pareto_front, key=lambda x, i=objective_index: x.fitness[i])
            crowding_distances[pareto_front_sorted[0]] = np.inf
            crowding_distances[pareto_front_sorted[-1]] = np.inf
            for i in range(1, len(pareto_front_sorted)-1):
                crowding_distances[pareto_front_sorted[i]] += (
                    pareto_front_sorted[i+1].fitness[objective_index] -
                    pareto_front_sorted[i-1].fitness[objective_index]
                )
        return crowding_distances
