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

# import warnings
# warnings.filterwarnings("error")

def train(env_fn, steps: int = 1e4, seed: int = 0, **env_kwargs):
    env = env_fn.parallel_env(**env_kwargs)
    env.reset(seed=seed)
    print(f"Starting training on {str(env.metadata['name'])}.")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs = 1, num_cpus=1, base_class="stable_baselines3")
    env = VecMonitor(env)
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[5, 5], vf=[5, 5]))
    
    model = PPO(
        MlpPolicy,
        env,
        verbose=2,
        learning_rate=1e-4,
        gamma = 1,
        n_steps= int(0.2 * env_kwargs['pso_iterations']),
        batch_size=100,
        n_epochs = 20,
        # vf_coef=0.1,
        # ent_coef = 0.1,
        # max_grad_norm=0.01,
        # policy_kwargs=policy_kwargs
    )

    # print("MODEL:")
    # print(model.policy)
    model.learn(total_timesteps=steps, progress_bar=True, callback=callback.CustomCallback())
    model.save("model")
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    env.close()


num_agents = 50
num_iterations = 100
num_params = 2

lb = [-10.] * num_params
ub = [10.] * num_params

optimizer.Logger.setLevel('INFO')

def objective1(x):
    return (x[0] + 2) ** 2 - 10

def objective2(x):
    return (x[0] - 2) ** 2 + 20

use_reinforcement_learning = 0

optimizer.FileManager.working_dir = "tmp/easy_problem/"
optimizer.FileManager.loading_enabled = False
optimizer.FileManager.saving_enabled = False


objective = optimizer.ElementWiseObjective([objective1, objective2])

pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                      num_particles=num_agents,
                      inertia_weight=0.6, cognitive_coefficient=1, social_coefficient=2, initial_particles_position='random', incremental_pareto=True, 
                      use_reinforcement_learning=use_reinforcement_learning)

def main():
    # global pso
    # # run the optimization algorithm
    # pso.optimize(num_iterations)

    # fig, ax = plt.subplots()

    # pareto_front = pso.pareto_front
    # n_pareto_points = len(pareto_front)
    # pareto_x = [particle.fitness[0] for particle in pareto_front]
    # pareto_y = [particle.fitness[1] for particle in pareto_front]
    # print(pareto_x, pareto_y)
    # plt.scatter(pareto_x, pareto_y, s=5, color = "red")
    # if not os.path.exists('tmp'):
    #     os.makedirs('tmp')
    # plt.savefig('tmp/pf_'+str(use_reinforcement_learning)+'.png')

    print("NEXT STEP")
    pso = optimizer.MOPSO(objective=objective, lower_bounds=lb, upper_bounds=ub,
                        num_particles=num_agents,
                        inertia_weight=0.6, cognitive_coefficient=0.5, social_coefficient=1, initial_particles_position='random', incremental_pareto=True, 
                        use_reinforcement_learning=use_reinforcement_learning)

    env_fn = pso_environment_AEC
    env_kwargs = {'pso' : pso,
                'pso_iterations' : num_iterations,
                'metric_reward' : 10,
                'evaluation_penalty' : -1,
                'not_dominated_reward' : 0,
                'render_mode' : 'human'
                    }

    train(env_fn, steps=2e6, seed=0, **env_kwargs)

    # print("TEST")
    # env = pso_environment_AEC.env(**env_kwargs)
    # model = PPO.load("model")

    # rewards = {"particle_" + str(i) for i in range(num_agents)}
    # env.reset()
    # num_actions = num_agents
    # for agent in env.agent_iter():
    #             obs, reward, termination, truncation, info = env.last()
    #             # print("Observation ", obs)

    #             # for a in env.agents:
    #             #     rewards[agent] += env.rewards[agent]
    #             if termination or truncation:
    #                 plt.figure()
    #                 fitnesses = np.array([p.fitness for p in env.env.pso.pareto_front])
    #                 plt.scatter(fitnesses[:,0],fitnesses[:,1], s=5)
    #                 n_pareto_points = len(env.env.pso.pareto_front)
    #                 real_x = (np.linspace(0, 1, n_pareto_points))
    #                 real_y = 1-np.sqrt(real_x)
    #                 plt.scatter(real_x, real_y, s=5, c='red')
    #                 plt.savefig("paretoRL.png")
    #                 break
    #             else:
    #                 actions = model.predict(obs, deterministic=True)[0]
    #                 print(actions)
    #                 num_actions += np.sum(actions)
    #                 # print("Action ", act)

    #             env.step(actions)
    #             print("Iteration ", env.env.pso.iteration)
    # print("Tot evaluations: ", num_actions)
    # env.close()

if __name__ == "__main__":
    main()