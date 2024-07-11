import optimizer
import numpy as np
from optimizer.tester import test, test_time,test_random

num_agents = 10
num_iterations = 70
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


objective = optimizer.ElementWiseObjective([objective1, objective2], sleep_time = 0.01)

# test(objective, num_agents, num_iterations, lb, ub)
# test_time(objective, num_agents, num_iterations, lb, ub, [0.1, 0.5, 1, 2, 5, 10])
test_random(objective, num_agents, num_iterations, lb, ub, max_time= 5)