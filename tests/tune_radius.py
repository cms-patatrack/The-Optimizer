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
from optimizer.tester import test_model

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


optimizer.FileManager.working_dir = "tmp/periodic_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2])

def main():

    radiuses = np.linspace(0.01, 0.1, 20)
    num_rad = len(radiuses)
    means_trained = np.empty(num_rad)
    stds_trained = np.empty(num_rad)
    means_random = np.empty(num_rad)
    stds_random = np.empty(num_rad)

    models_to_test = ['pso_trained_policy', 'pso_random_policy']

    for r, radius in enumerate(radiuses):
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
        name = f"zdt1_radius_{radius}"
        path = f"./models/tune_radius/" + name
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
        seeds = list(range(50, 150))
        results = test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, path, plot_paretos_enabled = False, time_limit = 3, models_to_test = models_to_test, verbose = 2)
        
        means_trained[r] = results.get_metric_means('hyper_volume')['pso_trained_policy']
        means_random[r] = results.get_metric_means('hyper_volume')['pso_random_policy']
        stds_trained[r] = results.get_metric_stds('hyper_volume')['pso_trained_policy']
        stds_random[r] = results.get_metric_stds('hyper_volume')['pso_random_policy']

    plt.figure()
    plt.errorbar(radiuses, means_trained,stds_trained,label='Trained policy')
    plt.errorbar(radiuses, means_random,stds_random,label='Random policy')
    plt.legend()
    plt.savefig("radius_tuning.png")

if __name__ == "__main__":
    main()