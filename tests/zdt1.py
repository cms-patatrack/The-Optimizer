import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


num_agents = 100
num_iterations = 200
num_params = 30

lb = [0] * num_params
ub = [1] * num_params

def zdt1_objective1(x):
    return x[0]

def zdt1_objective2(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

optimizer.FileManager.working_dir="tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2])

pso = optimizer.MOPSO(objective=objective,lower_bounds=lb, upper_bounds=ub, 
            num_particles=num_agents, num_iterations=num_iterations, 
            inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, 
            max_iter_no_improv=None)

# run the optimization algorithm
pso.optimize()

fig, ax = plt.subplots()

pareto_front = pso.get_current_pareto_front()
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]
real_x = (np.linspace(0, 1, n_pareto_points))
real_y = 1-np.sqrt(real_x)
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig('tmp/pf.png')

