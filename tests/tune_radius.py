from optimizer import pso_environment_AEC
import optimizer
import numpy as np
from stable_baselines3 import PPO, TD3
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy, MultiInputPolicy
import time
from matplotlib import pyplot as plt
from optimizer import callback
import os
import torch as th
from stable_baselines3.common.vec_env import VecMonitor
import pdb
from optimizer.trainer import train
from optimizer.tester import test_model, explainability
from joblib import Parallel, delayed
from objective_functions import zdt1
from multiprocessing import Pool

num_agents = 20
num_iterations = 100
num_params = 2

lb = [-10.] * num_params
ub = [10.] * num_params
lb_zdt1 = [0.] * 30
ub_zdt1 = [1.] * 30

optimizer.Logger.setLevel('INFO')

def objective1(x):
    return 3 * np.cos(x[0])

def objective2(x):
    return 3 * np.cos(x[0] + np.pi / 2) + 1


optimizer.FileManager.working_dir = "tmp/periodic_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([objective1, objective2])
objective_zdt1 = optimizer.ElementWiseObjective(zdt1.zdt1_objective, 2)
directory = f"./models/tune_radius_60_0_1/"

def evaluate_radius(radius):
    models_to_test = ['pso', 'pso_trained_policy', 'pso_random_policy']

    name = f"zdt1_radius_{radius}"
    path = directory + name
    print('########################################################')
    print(f"Testing radius {radius}")
    print('########################################################')
    if not os.path.isfile(path + "_model"):
        pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                            num_particles=num_agents,
                            inertia_weight=0.6, cognitive_coefficient=0.5, social_coefficient=1, initial_particles_position='random', exploring_particles=False,
                            rl_model=None, radius_scaler=radius)

        env_fn = pso_environment_AEC
        env_kwargs = {'pso' : pso,
                    'pso_iterations' : num_iterations,
                    'metric_reward' :  1 / 54.06236516259962 * 60,
                    'metric_reward_hv_diff' : 0, #1 / 54.06236516259962 * scaler,
                    'evaluation_penalty' : -1,
                    'not_dominated_reward' : 2,
                    'radius_scaler' : radius,
                    'render_mode' : 'None'
                        }
        
        train(env_fn, steps=1e6, seed=0, name = path, **env_kwargs)

    mopso_parameters = {'lower_bounds': lb_zdt1,
                'upper_bounds'        : ub_zdt1,
                'num_particles'       : num_agents,
                'topology'            : 'round_robin',
                'exploring_particles' : True,   
                'radius_scaler'       : radius 
                }
    
    rl_model = f"{path}_model"
    print(f"Model {rl_model}")
    ref_point = [5, 5]
    seeds = list(range(50, 150))
    explainability(rl_model, 50)
    results = test_model(objective_zdt1, mopso_parameters, num_iterations, rl_model, ref_point, seeds, path, plot_paretos_enabled = False, models_to_test = models_to_test, verbose = 2)
    return results


def main():
    seeds = list(range(50, 150))
    num_seeds = len(seeds)
    radiuses = np.linspace(0.01, 0.04, 60)
    print(f"Radiuses {radiuses}")
    num_rad = len(radiuses)
    hv_means_trained = np.zeros(num_rad)
    hv_stds_trained = np.zeros(num_rad)
    hv_means_random = np.zeros(num_rad)
    hv_stds_random = np.zeros(num_rad)
    evaluations_means_trained = np.zeros(num_rad)
    evaluations_stds_trained = np.zeros(num_rad)
    evaluations_means_random = np.zeros(num_rad)
    evaluations_stds_random = np.zeros(num_rad)

    res_objs = []
    with Pool(20) as p:
       res_objs = p.map(evaluate_radius, radiuses)

    for r, res in enumerate(res_objs):
        # results = evaluate_radius(radius)    
        hv_means_trained[r] = res.get_metric_means('hyper_volume')['pso_trained_policy']
        hv_means_random[r] = res.get_metric_means('hyper_volume')['pso_random_policy']
        hv_stds_trained[r] = res.get_metric_stds('hyper_volume')['pso_trained_policy'] / np.sqrt(num_seeds)
        hv_stds_random[r] = res.get_metric_stds('hyper_volume')['pso_random_policy'] / np.sqrt(num_seeds)

        evaluations_means_trained[r] = res.get_metric_means('evaluations')['pso_trained_policy']
        evaluations_means_random[r] = res.get_metric_means('evaluations')['pso_random_policy']
        evaluations_stds_trained[r] = res.get_metric_stds('evaluations')['pso_trained_policy'] / np.sqrt(num_seeds)
        evaluations_stds_random[r] = res.get_metric_stds('evaluations')['pso_random_policy'] / np.sqrt(num_seeds)

    fig, axs = plt.subplots(2, 1, figsize = (10, 7))
    axs[0].errorbar(radiuses, hv_means_trained, hv_stds_trained, label='Trained policy')
    axs[0].errorbar(radiuses, hv_means_random, hv_stds_random, label='Random policy')
    axs[0].set_xlabel('Radius\' scaler')
    axs[0].set_ylabel('Mean hyper volume')
    axs[0].legend()

    axs[1].errorbar(radiuses, evaluations_means_trained, evaluations_stds_trained, label='Trained policy')
    axs[1].errorbar(radiuses, evaluations_means_random, evaluations_stds_random, label='Random policy')
    axs[1].set_xlabel('Radius\' scaler')
    axs[1].set_ylabel('Mean number of evaluations')
    axs[1].legend()

    plt.savefig(f"radius_tuning_60_0_1_periodic.png")

    np.save(f"{directory}_hv_means_trained.npy", hv_means_trained)
    np.save(f"{directory}_hv_stds_trained.npy", hv_stds_trained)
    np.save(f"{directory}_hv_means_random.npy", hv_means_random)
    np.save(f"{directory}_hv_stds_random.npy", hv_stds_random)

if __name__ == "__main__":
    main()