from optimizer import pso_environment_AEC
import optimizer
import numpy as np
from stable_baselines3 import PPO
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy
import time

from pettingzoo.test import parallel_api_test, parallel_seed_test


num_agents = 2
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

use_reinforcement_learning = 0

optimizer.Randomizer.rng = np.random.default_rng(43)

optimizer.FileManager.working_dir = "tmp/zdt1/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False

objective = optimizer.ElementWiseObjective([zdt1_objective1, zdt1_objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                      use_reinforcement_learning=use_reinforcement_learning)

def main():
    env_kwargs = {'pso' : pso,
                'pso_iterations' : 10,
                'metric_reward' : 10,
                'evaluation_penalty' : -1,
                'not_dominated_reward' : 10,
                'render_mode' : None
                  }
    
    env_fn = pso_environment_AEC.parallel_env
    env = pso_environment_AEC.parallel_env(**env_kwargs)
    
    # parallel_seed_test(env_fn, num_cycles=10)
    print("Test started")
    # parallel_api_test(env, num_cycles=10)
    env.close()

    print("TESTING")
    env = pso_environment_AEC.parallel_env(**env_kwargs)
    env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}

        observations, rewards, terminations, truncations, infos = env.step(actions)
        print("Iteration ", env.aec_env.env.pso.iteration)
        print(rewards)
        print(terminations)
        input()
    env.close()


    # for agent in env.agent_iter():
    #     obs, reward, termination, truncation, info = env.last()
    #     print("Observation ", obs)
    #     if termination or truncation:
    #         break
    #     else:
    #         actions = np.random.choice([0,1])

    #     env.step(actions)
    #     print("Iteration ", env.env.pso.iteration)
    # env.close()
    

if __name__ == "__main__":
    main()
