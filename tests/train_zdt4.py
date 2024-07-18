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
num_params = 30

lb = [0.] + [-5.] * (num_params - 1)
ub = [1.] + [5.] * (num_params - 1)

optimizer.Logger.setLevel('INFO')

def zdt4_objective1(x):
    return x[0]

def zdt4_objective2(x):
    f1 = x[0]
    g = 1.0 + 10 * (len(x) - 1) + sum([i**2 - 10 * np.cos(4 * np.pi * i) for i in x[1:]])
    h = 1.0 - np.sqrt(f1 / g)
    f2 = g * h
    return f2


optimizer.FileManager.working_dir = "tmp/periodic_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([zdt4_objective1, zdt4_objective2])

def main():

    pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                        num_particles=num_agents,
                        inertia_weight=0.4, cognitive_coefficient=4, social_coefficient=2, initial_particles_position='random', exploring_particles=False,
                        rl_model=None)

    env_fn = pso_environment_AEC
    env_kwargs = {'pso' : pso,
                'pso_iterations' : num_iterations,
                'metric_reward' : 100 , #num_iterations / 24.66408110242748 / 3,
                'evaluation_penalty' : -1,
                'not_dominated_reward' : 2,
                'render_mode' : 'None'
                    }

    name = f"model_long"
    train(env_fn, steps=5000000,name=name, seed=0, **env_kwargs)

if __name__ == "__main__":
    main()