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

num_agents = 50
num_iterations = 200
num_params = 2

lb = [-10.] * num_params
ub = [10.] * num_params

optimizer.Logger.setLevel('INFO')

def objective1(x):
    return 3 * np.cos(x[0])

def objective2(x):
    return 3 * np.cos(x[0] + np.pi / 2) + 1

use_reinforcement_learning = 0

optimizer.FileManager.working_dir = "tmp/periodic_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2])

def main():

    pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=0.5, social_coefficient=1, initial_particles_position='random', 
                        exploring_particles = True, rl_model=None, topology = 'round_robin')

    env_fn = pso_environment_AEC
    env_kwargs = {'pso' : pso,
                'pso_iterations' : num_iterations,
                'metric_reward' : 1,
                'evaluation_penalty' : -1,
                'not_dominated_reward' : 2,
                'render_mode' : 'None'
                    }

    train(env_fn, steps=5000000, seed=0, **env_kwargs)

if __name__ == "__main__":
    main()