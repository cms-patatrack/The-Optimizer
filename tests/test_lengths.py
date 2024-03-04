import optimizer
import numpy as np

num_agents = 100
num_iterations = 200

lb = [1., 2, 1, 0]
ub = [2, 1, 1]

def zdt1_objective1(x):
    return x[0]

optimizer.Randomizer.rng = np.random.default_rng(42)

optimizer.FileManager.working_dir = "tmp/test-length/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([zdt1_objective1])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub)
