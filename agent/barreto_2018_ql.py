# from https://gitlab.inria.fr/creinke/transfer_rl

import numpy as np
import utils as utils
import exputils as eu
import exputils.data.logging as log

class Barreto2018QL:

    @staticmethod
    def default_config():
        dc = eu.AttrDict(
            alpha = 0.05,
            alpha_step_counter='episodes_per_reward_function',  # total_episodes, episodes_per_reward_function
            epsilon = 0.15, # can also be a dict that defines the function to get epsilon
            epsilon_step_counter = 'episodes_per_reward_function',  # total_episodes, episodes_per_reward_function
            gamma = 0.95,
            z_init_std = 0.01
        )
        return dc
        

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        # handle alpha from config
        if self.config.alpha_step_counter is not None and self.config.alpha_step_counter not in ['total_episodes', 'episodes_per_reward_function', '']:
            raise ValueError('Invalid value {!r} for alpha_step_counter configuration! Accepted values: \'total_episodes\', \'episodes_per_reward_function\'.'. format(self.config.epsilon_step_counter))

        if not self.config.alpha_step_counter:
            self.alpha = self.config.alpha
        else:
            self.alpha = 0.0

        # handle epsilon from config
        if self.config.epsilon_step_counter is not None and self.config.epsilon_step_counter not in ['total_episodes', 'episodes_per_reward_function', '']:
            raise ValueError('Invalid value {!r} for epsilon_step_counter configuration! Accepted values: \'total_episodes\', \'episodes_per_reward_function\'.'. format(self.config.epsilon_step_counter))

        if not self.config.epsilon_step_counter:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = 0.0

        self.z_weights = []

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.episodes_per_reward_function_counter = []
        self.episodes_counter = 0


    def add_reward_function(self, reward_function, reward_func_descr=None):

        self.episodes_per_reward_function_counter.append(0)

        # set up initial q-function for the new reward function
        z = np.random.randn(self.action_space.n, self.observation_space.shape[0]) * self.config.z_init_std
        self.z_weights.append(z)

        # return new rfunc index
        return len(self.z_weights) - 1


    def set_active_reward_func_idx(self, idx):
        self.active_reward_func_idx = idx


    def new_episode(self, state, info):

        # set alpha according to current episode
        if self.config.alpha_step_counter == 'total_episodes':
            self.alpha = utils.set_value_by_function(
                self.config.alpha,
                self.episodes_counter)
        elif self.config.alpha_step_counter == 'episodes_per_reward_function':
            self.alpha = utils.set_value_by_function(
                self.config.alpha,
                self.episodes_per_reward_function_counter[self.active_reward_func_idx])
        log.add_value('agent_alpha_per_episode', self.alpha)

        # set epsilon according to current episode
        if self.config.epsilon_step_counter == 'total_episodes':
            self.epsilon = utils.set_value_by_function(
                self.config.epsilon,
                self.episodes_counter)

        elif self.config.epsilon_step_counter == 'episodes_per_reward_function':
            self.epsilon = utils.set_value_by_function(
                self.config.epsilon,
                self.episodes_per_reward_function_counter[self.active_reward_func_idx])

        log.add_value('agent_epsilon_per_episode', self.epsilon)

        # increase internal counters of episodes
        self.episodes_per_reward_function_counter[self.active_reward_func_idx] += 1
        self.episodes_counter += 1


    def step(self, state, info):
        '''Draws an epsilon-greedy policy'''

        if np.random.rand() < self.epsilon:
            action = np.random.choice(np.arange(self.action_space.n))
        else:
            action = self.calc_max_action(state, self.active_reward_func_idx)

        return action


    def update(self, transition):
        '''use the given transition to update the agent'''

        state, action, next_state, reward, done, info = transition

        # identify the optimal action for the policy in the next state
        next_action = self.calc_max_action(next_state, self.active_reward_func_idx)

        # get z weights for current reward function (i.e. MDP)
        z_weight = self.z_weights[self.active_reward_func_idx]

        # calculate the Q-values based on the weights and the current state
        cur_q = np.sum(state * z_weight[action])
        next_q = np.sum(next_state * z_weight[next_action])

        # update the weights
        gamma = self.config.gamma if not done else 0
        z_weight[action] += self.alpha * (reward + gamma * next_q - cur_q) * state


    def calc_max_action(self, state, reward_function_idx):
        '''Get the optimal action for a given reward function'''

        # calculate the Q-values for all actions
        q_values = np.sum(state * self.z_weights[reward_function_idx], axis=1)

        # select action with max value, if there are several actions with max value, then select one randomly
        max_value = np.max(q_values)
        max_value_idxs = np.where(q_values == max_value)[0]
        action = np.random.choice(max_value_idxs)

        return action


    def calc_expected_return(self, state, action, reward_function_idx=None):

        if reward_function_idx is None:
            reward_function_idx = self.active_reward_func_idx

        q_value = np.sum(state * self.z_weights[reward_function_idx][action])

        return q_value