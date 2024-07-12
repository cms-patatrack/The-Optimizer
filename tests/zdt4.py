import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

num_agents = 200
num_iterations = 600
num_params = 10

lb = [0.] + [-5.] * (num_params - 1)
ub = [1.] + [5.] * (num_params - 1)

optimizer.Logger.setLevel('DEBUG')

optimizer.Randomizer.rng = np.random.default_rng(46)

def zdt4_objective1(x):
    return x[0]


def zdt4_objective2(x):
    f1 = x[0]
    g = 1.0 + 10 * (len(x) - 1) + sum([i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:]])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

optimizer.FileManager.working_dir = "tmp/zdt4/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = True

if not os.path.exists(optimizer.FileManager.working_dir):
    os.makedirs(optimizer.FileManager.working_dir)

objective = optimizer.ElementWiseObjective([zdt4_objective1, zdt4_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.1, cognitive_coefficient=1, social_coefficient=0.5,
                      initial_particles_position='random', topology = 'random',
                      exploring_particles=True, max_pareto_lenght=2*num_agents)

# run the optimization algorithm
pso.optimize(num_iterations, max_iterations_without_improvement=10)

fig, ax = plt.subplots()

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

real_x = (np.linspace(0, 1, 100))
real_y = 1 - np.sqrt(real_x)
plt.scatter(real_x, real_y, s=5, c='red')
plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig(optimizer.FileManager.working_dir + 'pf_with_exploration.png')

metrics = [pd.read_csv('tmp/zdt4/history/iteration' + str(i) + '.csv',
                       header=None,
                       usecols=[num_params, num_params + 1]).transpose().to_numpy()
           for i in range(num_iterations)]


fig, ax = plt.subplots()

def animate(i):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 50)
    ax.scatter(metrics[i][0], metrics[i][1], c='red', s=10)
    ax.set_title(str(i))


ani = animation.FuncAnimation(
    fig, animate, interval=200, frames=range(num_iterations))
ani.save('tmp/zdt4/checkpoint/metrics.gif', writer='pillow')
