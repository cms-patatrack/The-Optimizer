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
from optimizer.tester import test_model, print_results, plot_paretos, explainability

num_agents = 50
num_iterations = 200
num_params = 30

lb = [0.] * num_params
ub = [1.] * num_params

optimizer.Logger.setLevel('INFO')

def zdt1_objective1(x):
    return x[0]

def zdt1_objective2(x):
    f1 = x[0]
    g = 1 + 9.0 / (len(x)-1) * sum(x[1:])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2


optimizer.FileManager.working_dir = "tmp/periodic_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2])

def main():

    radiuses = np.linspace(0.01, 0.1, 0.01)
    num_rad = len(radiuses)
    means_trained = np.empty(num_rad)
    stds_trained = np.empty(num_rad)
    means_random = np.empty(num_rad)
    stds_random = np.empty(num_rad)

    for r in radiuses:
        pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                            num_particles=num_agents,
                            inertia_weight=0.4, cognitive_coefficient=4, social_coefficient=2, initial_particles_position='random', exploring_particles=False,
                            rl_model=None)

        env_fn = pso_environment_AEC
        env_kwargs = {'pso' : pso,
                    'pso_iterations' : num_iterations,
                    'metric_reward' : 3,
                    'evaluation_penalty' : -1,
                    'not_dominated_reward' : 2,
                    'radius_scaler' : r,
                    'render_mode' : 'None'
                        }
        name = f"./models/tune_radius/zdt1_radius_{r}"
        train(env_fn, steps=2000000, seed=0, name = name, **env_kwargs)

        mopso_parameters = {'lower_bounds': lb,
                    'upper_bounds'        : ub,
                    'num_particles'       : num_agents,
                    'topology'            : 'round_robin',
                    'exploring_particles' : True,   
                    'radius'              : r  
                    }
        
        rl_model = f"{name}_model"
        ref_point = [5, 5]
        seeds = list(range(50, 150))
        results = test_model(objective, mopso_parameters, num_iterations, rl_model, ref_point, seeds, name, plot_paretos_enabled = False, time_limit = 3, verbose = 2)
        
        means_trained.append(results.get_metric_means('hyper_volume')['pso_trained_policy'])
        means_random.append(results.get_metric_means('hyper_volume')['pso_random_policy'])
        stds_trained.append(results.get_metric_stds('hyper_volume')['pso_trained_policy'])
        stds_random.append(results.get_metric_stds('hyper_volume')['pso_random_policy'])

    plt.figure()
    plt.errorbar(radiuses, means_trained,stds_trained,label='Trained policy')
    plt.errorbar(radiuses, means_random,stds_random,label='Random policy')
    plt.legend()
    plt.savefig("radius_tuning.png")

if __name__ == "__main__":
    main()