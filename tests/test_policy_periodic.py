import optimizer
import numpy as np
from pymoo.indicators.hv import HV
from matplotlib import pyplot as plt

num_agents = 20
num_iterations = 2
num_params = 2

lb = [-10.] * num_params
ub = [10.] * num_params


optimizer.Logger.setLevel('INFO')

def objective1(x):
    return 3 * np.cos(x[0])

def objective2(x):
    return 3 * np.cos(x[0] + np.pi / 2) + 1

optimizer.FileManager.working_dir = "tmp/policy_easy/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2], sleep_time = 0.)

# test(objective, num_agents, num_iterations, lb, ub)
# test_time(objective, num_agents, num_iterations, lb, ub, [0.1, 0.5, 1, 2, 5, 10])
# test_random(objective, num_agents, num_iterations, lb, ub, max_time= 5)

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', topology="round_robin")

# run the optimization algorithm
pso.optimize(num_iterations)
ind = HV(ref_point=[5,5])
hv_pso = ind(np.array([p.fitness.tolist() for p in pso.pareto_front]))
plt.figure()
print(round(hv_pso / 54.06236516259962, 2))

pareto_front = pso.pareto_front
n_pareto_points = len(pareto_front)
pareto_x = [particle.fitness[0] for particle in pareto_front]
pareto_y = [particle.fitness[1] for particle in pareto_front]

plt.scatter(pareto_x, pareto_y, s=5)

plt.savefig('Pareto_try.png')