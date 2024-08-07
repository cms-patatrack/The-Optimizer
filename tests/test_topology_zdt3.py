import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import copy
from pymoo.indicators.hv import HV

num_agents = 100
num_iterations = 100
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('ERROR')

def zdt3_objective1(x):
    return x[0]


def zdt3_objective2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    h = (1.0 - np.power(f1 * 1.0 / g, 0.5) -
         (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))
    f2 = g * h
    return f2

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([zdt3_objective1, zdt3_objective2])

topologies = [
            'random', 'higher_crowding_distance', 'lower_crowding_distance', 'higher_weighted_crowding_distance',
            'lower_weighted_crowding_distance', 'round_robin', 'higher_crowding_distance_random_ring', 
            'lower_crowding_distance_random_ring'
            ]

paretos = []
seeds = list(range(50,150))
# markers = ['.', 's', '1', 'x', 'v']
ref_point = [5, 5]
ind = HV(ref_point=ref_point)
hvs = np.empty((len(topologies), len(seeds)))

inertia_weight = 0.4
cognitive_coefficient = 1
social_coefficient = 2
for i, t in enumerate(topologies):
    # fig, ax = plt.subplots()
    # plt.scatter(real_x, real_y, s=5, c='red', marker = "*",label = "Real pareto")
    for j, s in enumerate(seeds):
        print(f"Starting {t} topology with seed {s}")
        optimizer.Randomizer.rng = np.random.default_rng(s)  
        pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                            num_particles=num_agents,
                            inertia_weight=inertia_weight, cognitive_coefficient=cognitive_coefficient, social_coefficient=social_coefficient, 
                            initial_particles_position='random', incremental_pareto=False,
                            topology = t, seed = s)

        # run the optimization algorithm
        pso.optimize(num_iterations)
        pareto_front = pso.pareto_front
        hv = ind(np.array([p.fitness for p in pareto_front]))
        hvs[i][j] = hv
        # paretos.append(copy.deepcopy(pso.pareto_front))
        # pareto_x = [particle.fitness[0] for particle in pareto_front]
        # pareto_y = [particle.fitness[1] for particle in pareto_front]
        # plt.scatter(pareto_x, pareto_y, s=20, label=f"seed {s}", marker = m)
        # print(f"Len for {t} topology: {len(pso.pareto_front)}")
    # plt.xlabel("Objective 1")
    # plt.ylabel("Objective 2")
    # plt.legend()
    # plt.savefig(f"./plots/topology_{t}")
    # plt.close()
name = f"hyper_volumes_zdt3_agents_{num_agents}_iter_{num_iterations}_inertia_{inertia_weight}_cognitive_{cognitive_coefficient}_social_{social_coefficient}.npy"
np.save(name, hvs)
mean = np.mean(hvs, axis = 1)
err = np.std(hvs, axis = 1)
for i,t in enumerate(topologies):
    print(f"{t}: {mean[i]} +- {err[i] / np.sqrt(len(seeds))}")
    # print(f"std: {err[i]}")

regions = [[0, 0.0830015349],
           [0.182228780, 0.2577623634],
           [0.4093136748, 0.4538821041],
           [0.6183967944, 0.6525117038],
           [0.8233317983, 0.8518328654]]

pf = []

for r in regions:
    x1 = np.linspace(r[0], r[1], int(100 / len(regions)))
    x2 = 1 - np.sqrt(x1) - x1 * np.sin(10 * np.pi * x1)
    pf.append([x1, x2])

real_x = np.concatenate([x for x, _ in pf])
real_y = np.concatenate([y for _, y in pf])

print(f"True hv: {ind(np.array([np.array([real_x[i], real_y[i]]) for i in range(len(real_x))]))}")

if not os.path.exists('tmp'):
    os.makedirs('tmp')