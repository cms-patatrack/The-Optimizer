import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 100
num_iterations = 100
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params
p_names = [f"x{i}" for i in range(num_params)]

optimizer.Logger.setLevel('DEBUG')

def zdt1_objective(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2


optimizer.Randomizer.rng = np.random.default_rng(46)

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = True
optimizer.FileManager.headers_enabled = True

objective = optimizer.ElementWiseObjective(zdt1_objective, 2, objective_names=['f1', 'f2'])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub, param_names=p_names,
                      num_particles=num_agents,
                      inertia_weight=0.4, cognitive_coefficient=1.5, social_coefficient=2,
                      initial_particles_position='random', exploring_particles=True, max_pareto_lenght=2*num_agents)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=5)

fig, ax = plt.subplots()
pso.tight_plot(plot_true_pareto=True, label="ZDT1")
plt.show()
plt.savefig('tmp/pf.png')

fig, ax = plt.subplots()
pso.parallel_coordinates(ax=ax, highlighted_particle=None)
plt.show()
plt.savefig('tmp/pc.png')

fig = plt.figure()
pso.scatter(fig)
plt.show()
plt.savefig('tmp/sc.png')
