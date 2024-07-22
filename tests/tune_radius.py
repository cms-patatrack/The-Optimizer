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

num_agents = 50
num_iterations = 100
num_params = 2

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('INFO')

def objective1(x):
    return 3 * np.cos(x[0])

def objective2(x):
    return 3 * np.cos(x[0] + np.pi / 2) + 1

def zdt1_objective(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f1, f2


optimizer.FileManager.working_dir = "tmp/periodic_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2])
objective_zdt1 = optimizer.ElementWiseObjective(zdt1_objective, 2)

def main():

    radiuses = np.linspace(0.01, 0.1, 20)
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

    models_to_test = ['pso_trained_policy', 'pso_random_policy']

    directory = f"./models/tune_radius/"
    for r, radius in enumerate(radiuses):
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
                        'metric_reward' :  num_iterations / 22 / 2,
                        'evaluation_penalty' : -1,
                        'not_dominated_reward' : 2,
                        'radius_scaler' : radius,
                        'render_mode' : 'None'
                            }
            
            train(env_fn, steps=1e6, seed=0, name = path, **env_kwargs)

        mopso_parameters = {'lower_bounds': lb,
                    'upper_bounds'        : ub,
                    'num_particles'       : num_agents,
                    'topology'            : 'round_robin',
                    'exploring_particles' : True,   
                    'radius_scaler'       : radius 
                    }
        
        rl_model = f"{path}_model"
        print(f"Model {rl_model}")
        ref_point = [5, 5]
        seeds = list(range(50, 51))
        results = test_model(objective_zdt1, mopso_parameters, num_iterations, rl_model, ref_point, seeds, path, plot_paretos_enabled = False, models_to_test = models_to_test, verbose = 2)
        explainability(rl_model, 50)
        
        hv_means_trained[r] = results.get_metric_means('hyper_volume')['pso_trained_policy']
        hv_means_random[r] = results.get_metric_means('hyper_volume')['pso_random_policy']
        hv_stds_trained[r] = results.get_metric_stds('hyper_volume')['pso_trained_policy']
        hv_stds_random[r] = results.get_metric_stds('hyper_volume')['pso_random_policy']

        evaluations_means_trained[r] = results.get_metric_means('evaluations')['pso_trained_policy']
        evaluations_means_random[r] = results.get_metric_means('evaluations')['pso_random_policy']
        evaluations_stds_trained[r] = results.get_metric_stds('evaluations')['pso_trained_policy']
        evaluations_stds_random[r] = results.get_metric_stds('evaluations')['pso_random_policy']

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

    plt.savefig("radius_tuning.png")

    np.save(f"{directory}_hv_means_trained.npy", hv_means_trained)
    np.save(f"{directory}_hv_stds_trained.npy", hv_stds_trained)
    np.save(f"{directory}_hv_means_random.npy", hv_means_random)
    np.save(f"{directory}_hv_stds_random.npy", hv_stds_random)

if __name__ == "__main__":
    main()