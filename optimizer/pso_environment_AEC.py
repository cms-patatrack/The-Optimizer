from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from .pso_environment_base import pso_environment_base as _env
from pettingzoo.utils import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

import copy


def env(**kwargs):
    env = raw_env(**kwargs)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["None", "human"],
        "name": "pso_environment_v0",
        "is_parallelizable" : True
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        AECEnv.__init__(self)
        self.env = _env(*args, **kwargs)

        self.agents = ["particle_" + str(p) for p in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        self.pso_iterations = kwargs['pso_iterations']

        # spaces
        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))

        self.render_mode = self.env.render_mode

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agents, list_of_list))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env._seed(seed=seed)

        obs = self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [(0) for _ in self.agents]))
        self.cumultaive_rewards = []
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{"cumulative_rewards" : []} for _ in self.agents]))
        return obs

    def close(self):
        pass

    def render(self):
        self.env.render()

    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)

        if is_last:
            for k in self.rewards.keys():
                self.rewards[k] = self.env.last_rewards[self.agent_name_mapping[k]]

            self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
            if self.env.pso.iteration == self.pso_iterations:
                self.terminations = dict(zip(self.agents, [True for _ in self.agents]))

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode == "human" and is_last:
            self.render()

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def action_masks(self) :
        return self.env.action_masks()
    
