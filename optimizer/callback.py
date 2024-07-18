from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from matplotlib import pyplot as plt

class CustomCallback(BaseCallback):
    """
    Those variables will be accessible in the callback
    (they are defined in the base class)
    The RL model
    self.model = None  # type: BaseAlgorithm
    An alias for self.model.get_env(), the environment used for training
    self.training_env # type: VecEnv
    Number of time the callback was called
    self.n_calls = 0  # type: int
    num_timesteps = n_envs * n times env.step() was called
    self.num_timesteps = 0  # type: int
    local and global variables
    self.locals = {}  # type: Dict[str, Any]
    self.globals = {}  # type: Dict[str, Any]
    The logger object, used to report things in the terminal
    self.logger # type: stable_baselines3.common.logger.Logger
    Sometimes, for event callback, it is useful
    to have access to the parent object
    self.parent = None  # type: Optional[BaseCallback]
    """
    def __init__(self, name = '', verbose: int = 0):
        super().__init__(verbose)
        self.name = name

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.keys = self.locals["self"].env.unwrapped.vec_envs[0].par_env.agents
        self.cumulative_episode_reward = {k : [] for k in self.keys}
        self.rewards = {k : [] for k in self.keys}
        self.num_timesteps 

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # hv = self.locals["model"].env.unwrapped.vec_envs[0].par_env.aec_env.env.hv
        # self.hvs.append(hv)
        for i, k in enumerate(self.rewards.keys()):
                self.rewards[k].append(self.locals["rewards"][i])
        if np.any(self.locals["dones"]):
            for k in self.cumulative_episode_reward.keys():
                self.cumulative_episode_reward[k].append(np.sum(self.rewards[k]))
            self.rewards = {k : [] for k in self.keys}

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print("Finished training")
        for k in self.cumulative_episode_reward.keys():
            plt.plot(self.cumulative_episode_reward[k], alpha = 0.5)

        matrix = np.array(list(self.cumulative_episode_reward.values()))
        mean = np.mean(matrix, axis = 0)
        np.save(f"{self.name}_mean_reward.npy", mean)
        np.save(f"{self.name}_rewards.npy", matrix)
        plt.plot(mean, color = 'black', label = "mean")
        plt.xlabel("Episodes")
        plt.ylabel("Agents reward")
        plt.legend(loc = 'lower right')
        plt.savefig(f"{self.name}_Cumulative_episodes_rewards.png")
        plt.close()

        # plt.figure()
        # plt.plot(self.hvs)
        # plt.savefig(f"{self.name}_hvs.png")
        # plt.close()