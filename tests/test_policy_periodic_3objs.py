import optimizer
import numpy as np
from optimizer.tester import test_model, print_results, plot_paretos, explainability

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

def objective3(x):
    return 3 * np.cos(0.6 * x[0] + 3 * np.pi / 4) -1

optimizer.FileManager.working_dir = "tmp/policy_easy/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False




mopso_parameters = {'lower_bounds'        : lb,
                    'upper_bounds'        : ub,
                    'num_particles'       : num_agents,
                    'topology'            : 'round_robin',
                    'exploring_particles' : True,   
                    'radius'              : None     
                    }

objective = optimizer.ElementWiseObjective([objective1, objective2, objective3], sleep_time = 0)
rl_model = './models/model_periodic/model'
# rl_model = './models/model_zdt1/model'
# rl_model = './models/model_periodic_parallel/model'
ref_point = [5, 5, 5]
seeds = list(range(50, 51))
print(seeds)
name = f"results_zdt1_agents_{num_agents}_iterations_{num_iterations}"

test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, name, plot_paretos_enabled = True, time_limit = 3, verbose = 2)