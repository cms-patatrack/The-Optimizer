import optimizer
import numpy as np
from matplotlib import pyplot as plt
from optimizer.tester import test_model, print_results, plot_paretos, explainability

# for the cool plot in the slides
num_agents = 50
num_iterations = 100

num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('ERROR')
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

def zdt1_objective1(x):
    return x[0]

def zdt1_objective2(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2

optimizer.FileManager.working_dir = "tmp/zdt1/"

mopso_parameters = {'lower_bounds'        : lb,
                    'upper_bounds'        : ub,
                    'num_particles'       : num_agents,
                    'topology'            : 'round_robin',
                    'exploring_particles' : True,   
                    'radius'              : None     
                    }

objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2], sleep_time = 0.001)
rl_model = './models/model_periodic/model'
# rl_model = './models/model_zdt1/model'
# rl_model = './models/model_periodic_parallel/model'
ref_point = [5, 5]
seeds = list(range(50, 150))
print(seeds)
name = f"results_zdt1_agents_{num_agents}_iterations_{num_iterations}"
test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, name, plot_paretos_enabled = True, time_limit = 3, verbose = 2)
explainability(rl_model, 100)
# print_results("results_zdt1_50_100.json")