from optimizer import pso_environment_AEC
import optimizer
import numpy as np
from stable_baselines3 import PPO
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy
import time
from matplotlib import pyplot as plt
from optimizer import callback


num_agents = 100
pso_iterations = 100
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

def train_butterfly_supersuit(env_fn, steps: int = 1e4, seed: int = 0, **env_kwargs):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn.parallel_env(**env_kwargs)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs = 4, num_cpus=1, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = PPO(
        MlpPolicy,
        env,
        verbose=2,
        learning_rate=1e-2,
        n_steps=2048,
        batch_size=2048,
        n_epochs = 100,
    )

    model.learn(total_timesteps=steps, progress_bar=True, callback=callback.CustomCallback())

    # model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    model.save("model")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")


    env.close()

def main():
    env_fn = pso_environment_AEC
    env_kwargs = {'pso' : pso,
                'pso_iterations' : pso_iterations,
                'metric_reward' : 10,
                'evaluation_penalty' : -1,
                'not_dominated_reward' : 5,
                'render_mode' : None
                  }

    # Train a model (takes ~3 minutes on GPU)
    # Steps should be a multiple of (n_steps, num_env, num_agents)
    train_butterfly_supersuit(env_fn, steps=1e8, seed=0, **env_kwargs)

if __name__ == "__main__":
    main()
