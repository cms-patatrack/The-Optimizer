from optimizer import pso_environment_AEC
import optimizer
import numpy as np
from stable_baselines3 import PPO
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy
import time
from matplotlib import pyplot as plt
from optimizer import callback
from tqdm import tqdm
import torch as th
from optimizer.masked_actor_critic import MaskedActorCriticPolicy

def train(env_fn, steps: int = 1e4, seed: int = 0, name = 'Model', **env_kwargs):
    env = env_fn.parallel_env(**env_kwargs)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs = 1, num_cpus=1, base_class="stable_baselines3")
    env.reset()
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)

    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[5, 5], vf=[5, 5]))
    
    model = PPO(
        MaskedActorCriticPolicy,
        env,
        verbose=2,
        learning_rate=1e-5,
        gamma = 1,
        n_steps= int(0.2 * env_kwargs['pso_iterations']),
        batch_size=100,
        n_epochs = 10,
        policy_kwargs=policy_kwargs
    )
    print("-" * 100)
    print("MODEL:")
    print(model.policy)
    print("-" * 100)
    print("Started training")
    model.learn(total_timesteps=steps, progress_bar=True, callback=callback.CustomCallback(name=name))
    model.save(f"{name}_model")
    print("Model has been saved.")
    print("Training complete")
    env.close()
