import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


num_agents = 60
num_iterations = 200
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

def zdt2_objective1(x):
    return x[0]

def zdt2_objective2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    h = 1.0 - np.power((f1 *1.0 / g),2)
    f2 = g * h
    return f2

optimizer.FileManager.working_dir="tmp/zdt2/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = True

objective = optimizer.ElementWiseObjective([zdt2_objective1, zdt2_objective2])

pso = optimizer.MOPSO(objective=objective,lower_bounds=lb, upper_bounds=ub, 
            num_particles=num_agents,
            inertia_weight=0.4, cognitive_coefficient=0.1, social_coefficient=2
            , initial_particles_position='random')

# run the optimization algorithm
pso.optimize(num_iterations)

fig, ax = plt.subplots()

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = (np.linspace(0, 1, n_pareto_points))
real_y = 1 - np.power(real_x, 2)
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig('tmp/pf.png')

