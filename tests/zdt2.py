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


def zdt2_objective1(x):
    return x[0]


def zdt2_objective2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.power((f1 * 1.0 / g), 2)
    f2 = g * h
    return f2

def true_pareto(x):
    f1 = x
    f2 = 1 - np.power(x, 2)
    return f1, f2

if not os.path.exists('tmp'):
    os.makedirs('tmp')

optimizer.FileManager.working_dir = "tmp/zdt2/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = True

objective = optimizer.ElementWiseObjective([zdt2_objective1, zdt2_objective2], true_pareto=true_pareto)

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.4, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', topology="random")

# run the optimization algorithm
pso.optimize(num_iterations)

fig, ax = plt.subplots()
pso.tight_plot(plot_true_pareto=True, label="ZDT2")
plt.show()

plt.savefig('tmp/pf.png')
