import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import warnings
warnings.filterwarnings("error")

num_agents = 100
num_iterations = 100
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('DEBUG')

def zdt1_objective(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2

use_reinforcement_learning = 0

optimizer.Randomizer.rng = np.random.default_rng(46)

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective(zdt1_objective, 2)

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2,
                      initial_particles_position='random', exploring_particles=True, max_pareto_lenght=2*num_agents)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=5)

print(len(pso.pareto_front))
fig, ax = plt.subplots()

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]
real_x = (np.linspace(0, 1, n_pareto_points))
real_y = 1-np.sqrt(real_x)
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

if not os.path.exists('tmp'):
    os.makedirs('tmp')
plt.savefig('tmp/pf.png')
plt.close()
