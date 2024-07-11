import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from stable_baselines3 import PPO
from optimizer import pso_environment_AEC
import supersuit as ss


import warnings
warnings.filterwarnings("error")

num_agents = 100
num_iterations = 100
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

use_reinforcement_learning = 1

optimizer.Randomizer.rng = np.random.default_rng(43)

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                      use_reinforcement_learning=use_reinforcement_learning)

# run the optimization algorithm

env_kwargs = {'pso' : pso,
                'pso_iterations' : 100,
                'metric_reward' : 100,
                'evaluation_penalty' : -1,
                'not_dominated_reward' : 5,
                'render_mode' : None
                  }
env = pso_environment_AEC.env(**env_kwargs)

print(f"Starting training on {str(env.metadata['name'])}.")

# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 1, num_cpus=2, base_class="stable_baselines3")
model = PPO.load("model")

rewards = {"particle_" + str(i) for i in range(num_agents)}
env.reset()
num_actions = num_agents
for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            # print("Observation ", obs)

            # for a in env.agents:
            #     rewards[agent] += env.rewards[agent]
            if termination or truncation:
                plt.figure()
                fitnesses = np.array([p.fitness for p in env.env.pso.pareto_front])
                plt.scatter(fitnesses[:,0],fitnesses[:,1], s=5)
                n_pareto_points = len(env.env.pso.pareto_front)
                real_x = (np.linspace(0, 1, n_pareto_points))
                real_y = 1-np.sqrt(real_x)
                plt.scatter(real_x, real_y, s=5, c='red')
                plt.savefig("paretoRL.png")
                break
            else:
                actions = model.predict(obs, deterministic=True)[0]
                print(actions)
                num_actions += np.sum(actions)
                # print("Action ", act)

            env.step(actions)
            print("Iteration ", env.env.pso.iteration)
print("Tot evaluations: ", num_actions)
env.close()