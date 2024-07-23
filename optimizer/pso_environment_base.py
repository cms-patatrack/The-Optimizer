import numpy as np
from gymnasium.spaces import Discrete, Box
import copy
from optimizer import Randomizer
from optimizer.reinforcement_learning_utils import observe_list, find_new_bad_points
from pymoo.indicators.hv import HV
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import optimizer


class pso_environment_base:
    def __init__(self, pso, pso_iterations, metric_reward, evaluation_penalty, not_dominated_reward, radius_scaler = 0.03, render_mode = None):
        
        self.possible_pso = pso
        self.pso_iterations = pso_iterations
        self.num_agents = self.possible_pso.num_particles
        self.metric_reward = metric_reward
        self.evaluation_penalty = evaluation_penalty
        self.not_dominated_reward = not_dominated_reward

        self.last_dones = [False for _ in range(self.num_agents)]
        self.last_obs = [None for _ in range(self.num_agents)]
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]

        self.render_mode = render_mode
        self._seed()

        # state stuff
        self.ref_point = [5, 5]
        self.ind = HV(ref_point=self.ref_point)
        upper_bounds = np.array(self.possible_pso.upper_bounds)
        lower_bounds = np.array(self.possible_pso.lower_bounds)
        self.num_parameters = len(lower_bounds)
        self.max_dist = np.linalg.norm(upper_bounds - lower_bounds)
        self.radius = radius_scaler * self.max_dist
        self.reset()
        
        self.get_spaces()
        print("Created environment")

    def get_spaces(self):
        # Define the action and observation spaces for all of the agents
        len_obs = 2
        low = np.array([0.] * len_obs)
        high = np.array([self.num_agents * self.pso_iterations] * 2)
        obs_space = Box(low = low, high = high, shape = (len_obs,), dtype=np.float32)
        act_space = Discrete(2)

        self.observation_space = [obs_space for i in range(self.num_agents)]
        self.action_space = [act_space for i in range(self.num_agents)]

    def _seed(self, seed=None):
        self.np_random = Randomizer.rng

    def reset(self):
        self.pso = copy.deepcopy(self.possible_pso)

        # Set up the reward
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]
        self.action_list = []
        self.good_points = []
        self.bad_points = []

        # Evaluate all particles to begin with
        self.pso.step()
        self.hv = round(self.ind(np.array([p.fitness for p in self.pso.pareto_front])), 2)
        self.prev_hv = self.hv
        
        # Get observation
        obs_list = self.observe_list()
        self.last_obs = obs_list
        self.last_dones = [False for _ in range(self.num_agents)]
        self.invalid_actions = [[] for _ in range(self.num_agents)]
        
        return obs_list

    def step(self, action, agent_id, is_last):
        # save the action to give the reward
        self.action_list.append(action)
        p = self.pso.particles[agent_id]
        
        # Execute actions
        # p.num_skips = 0 if action else p.num_skips + 1
        optimization_output = self.pso.objective.evaluate(np.array([p.position]))[0] if action else [np.inf] * len(p.fitness)
        p.set_fitness(optimization_output)

        # Give negative reward if evaluated
        agent_obs = self.last_obs[agent_id]
        # if agent_obs[0] != 0 or agent_obs[1] != 0:
        #     self.last_rewards[agent_id] = self.evaluation_penalty * action

        if is_last:
            # Update pareto
            dominated, crowding_distances = self.pso.update_pareto_front()

            # Assign reward if not dominated
            # for id in range(self.num_agents):
            #     self.last_rewards[id] += self.not_dominated_reward if not dominated[id] else 0

            # Assign reward if hv improves
            # hv = round(self.ind(np.array([p.fitness for p in self.pso.pareto_front])), 2)
            # # print(hv, self.prev_hv)
            # if hv > self.prev_hv:
            #     # print("YEAH")
            #     self.prev_hv = hv
            #     for id in range(self.num_agents):
            #         self.last_rewards[id] += self.metric_reward

            # print(self.last_rewards)


            # If a particle is dominated and was evaluated add it to bad points list.      
            self.bad_points += find_new_bad_points(self.pso.particles, dominated, self.action_list)

            # Update velocities and positions
            for particle in self.pso.particles:
                particle.update_velocity(self.pso.pareto_front,
                                            crowding_distances,
                                            self.pso.inertia_weight,
                                            self.pso.cognitive_coefficient,
                                            self.pso.social_coefficient)
                particle.update_position(self.pso.lower_bounds, self.pso.upper_bounds)

            # If is last iteration assign Hyper volume reward to all agents
            if self.pso.iteration == self.pso_iterations - 1:
                self.hv = self.ind(np.array([p.fitness for p in self.pso.pareto_front]))
                print("Hyper Volume: ", self.hv)
                for id in range(self.num_agents):
                    self.last_rewards[id] += self.metric_reward * self.hv
            #     print(f"metricccc {self.metric_reward * self.hv}")

            # End of pso iteration
            self.pso.iteration += 1
            self.action_list = []
            self.invalid_actions = [[] for _ in range(self.num_agents)]

            # Generate new observations
            obs_list = self.observe_list()
            self.last_obs = obs_list

            # plt.figure()
            # pareto_x = [particle.fitness[0] for particle in self.pso.pareto_front]
            # pareto_y = [particle.fitness[1] for particle in self.pso.pareto_front]
            # plt.scatter(pareto_x, pareto_y, s=5)
            # plt.savefig("Pareto_try")
            # input()

        return self.observe(agent_id)

    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        observations = observe_list(self.pso,
                            np.array([p.position for p in self.pso.pareto_front]),
                            np.array(self.bad_points),
                            self.radius,
                            self.max_dist,
                            self.pso_iterations
                            )
        for i, obs in enumerate(observations):
            if obs[0] > 0: self.invalid_actions[i].append(0)              
        return observations
    
    def action_masks(self):
        return [action not in self.invalid_actions for action in self.possible_actions]
                                     
    def render(self):
        # To be fixed
        if self.num_parameters == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            plt.scatter([p.position[0] for p in self.pso.particles],[p.position[1] for p in self.pso.particles], color = 'black', marker = '.', s = 100)
            plt.scatter([p[0] for p in self.bad_points], [p[1] for p in self.bad_points], color = 'blue', marker = 'x', s = 50)
            plt.scatter([p.position[0] for p in self.pso.pareto_front], [p.position[1] for p in self.pso.pareto_front], color = 'green', marker = '*', s = 50)

            rect = patches.Rectangle((-4 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            ax.add_patch(rect)

            rect = patches.Rectangle((-2 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            ax.add_patch(rect)

            rect = patches.Rectangle((np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            ax.add_patch(rect)

            rect = patches.Rectangle((2 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            ax.add_patch(rect)

            rect = patches.Rectangle((4 * np.pi + np.pi / 2, -10), np.pi / 2, 20, linewidth=1, edgecolor=None, facecolor='green', alpha=0.2)
            ax.add_patch(rect)

            for p in self.pso.particles:
                c = plt.Circle(p.position, self.radius, color = 'black', fill = False)
                ax.add_patch(c)
            plt.xlim(-10,10)
            plt.ylim(-10,10)
            plt.show()
            plt.close()
        else:
            print("No implementation found for render mode")        