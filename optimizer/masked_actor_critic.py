import torch as T
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3 import PPO
from torch.distributions.categorical import Categorical
import numpy as np

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(MaskedActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.action_dist = CategoricalDistribution(action_space.n)

    def forward(self, obs, deterministic=False):
        # obs_tensor, mask = obs['obs'], obs['mask']
        # print(obs)
        obs_tensor = obs
        mask = [[1 if o[0] > 0 else 0, 1 if o[1] > 0 else 0] for o in obs]
        # print("OBS ", obs)
        # print("MASK ", mask)
        features = self.extract_features(obs_tensor)
        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        # print("Logits ", logits)
        # mask = T.tensor(mask, dtype=T.float32, device=self.device)

        # s = mask.sum(dim=1, keepdim=True)
        # l = ((logits * (1 - mask)).sum(dim=1, keepdim=True) / s)
        # logits = (logits + l) * mask

        for i in range(len(logits)):
            if mask[i][0] == 0 and mask[i][1] == 0:
                logits[i][0] += -1e9

        # print("LOGITS ", logits)

        action_dist = self.action_dist.proba_distribution(logits)
        actions = action_dist.get_actions(deterministic=deterministic)
        # print("state: ", obs_tensor)
        # print("actions: ", actions)
        values = self.value_net(latent_vf)
        log_prob = action_dist.log_prob(actions)
        # print("log_prob: ", log_prob)
        # print("prob: ", [np.exp(i) for i in log_prob.cpu()])
        # print(actions)
        # input()
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        actions, _, _= self.forward(observation, deterministic=deterministic)
        return actions