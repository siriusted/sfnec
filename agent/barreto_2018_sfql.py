# from https://gitlab.inria.fr/creinke/transfer_rl

import numpy as np
import utils as utils
import exputils as eu
import exputils.data.logging as log
import warnings

class Barreto2018SFQL:
    """
    Successor Feature Q-learning agent for continual reward transfer scenarios that uses a single linear layer to
    approximate the Psi-function. Based on SF QL agent in paper [Barreto 2018].
    (https://arxiv.org/pdf/1606.05312.pdf)

    Configuration:
        policy_update_mode: Which policies are updated. (default='active_and_gpi_optimal')
            - 'all': All policies are updated.
            - 'active': Only the active policy is updated.
            - 'active_and_gpi_optimal': The active policy and the policy that provided the optimal action are updated.
                                        This is the strategy used in the Barreto2018 paper.
            - 'active_and_optimal': The active policy and the policies for which taken action is also optimal are
                                    updated.

        additional_policy_update_action_mode: Which action for the next state is used to update additional policies.
            Note: The active policy is always updated using GPI. (default='target_policy')
            - 'gpi': Optimal next action according to GPI.
            - 'target_policy': Optimal next action only according to the policy that gets updated.
                               This is the strategy used in the Barreto2018 paper.
    """

    @staticmethod
    def default_config():
        dc = eu.AttrDict(
            alpha = 0.05,
            alpha_step_counter='episodes_per_reward_function',  # total_episodes, episodes_per_reward_function
            alpha_w=0.05,
            alpha_w_step_counter='episodes_per_reward_function',  # total_episodes, episodes_per_reward_function
            epsilon = 0.15, # can also be a dict that defines the function to get epsilon
            epsilon_step_counter = 'episodes_per_reward_function',  # total_episodes, episodes_per_reward_function
            gamma = 0.95,
            w_init_std = 0.01,
            z_init_std = 0.01,
            phi_func = lambda transition: transition[5]['feature'], # [5] is the info component of transitions
            phi_size = 4,
            absolute_weight_maximum = 1000,  # maximum absolute value of weights used for w and z
            policy_update_mode = 'active_and_gpi_optimal',
            additional_policy_update_action_mode = 'target_policy',
            dogpi = True,
            z_init_mode = 'previous', # random
        )
        return dc
        

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        # handle alpha from config
        if self.config.alpha_step_counter is not None and self.config.alpha_step_counter not in ['total_episodes', 'episodes_per_reward_function', '']:
            raise ValueError('Invalid value {!r} for alpha_step_counter configuration! Accepted values: \'total_episodes\', \'episodes_per_reward_function\'.'. format(self.config.alpha_step_counter))

        if not self.config.alpha_step_counter:
            self.alpha = self.config.alpha
        else:
            self.alpha = 0.0

        # handle alpha from config
        if self.config.alpha_w_step_counter is not None and self.config.alpha_w_step_counter not in ['total_episodes', 'episodes_per_reward_function', '']:
            raise ValueError('Invalid value {!r} for alpha_w_step_counter configuration! Accepted values: \'total_episodes\', \'episodes_per_reward_function\'.'. format(self.config.alpha_w_step_counter))

        if not self.config.alpha_w_step_counter:
            self.alpha_w = self.config.alpha_w
        else:
            self.alpha_w = 0.0

        # handle epsilon from config
        if self.config.epsilon_step_counter is not None and self.config.epsilon_step_counter not in ['total_episodes', 'episodes_per_reward_function', '']:
            raise ValueError('Invalid value {!r} for epsilon_step_counter configuration! Accepted values: \'total_episodes\', \'episodes_per_reward_function\'.'. format(self.config.epsilon_step_counter))

        if not self.config.epsilon_step_counter:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = 0.0

        if self.config.policy_update_mode not in ['all', 'active', 'active_and_gpi_optimal', 'active_and_optimal']:
            raise ValueError('Invalid value {!r} for policy_update_mode configuration! Accepted values: \'all\', \'active\', \'active_and_gpi_optimal\', \'active_and_optimal\''. format(self.config.policy_update_mode))

        if self.config.additional_policy_update_action_mode not in ['gpi', 'target_policy']:
            raise ValueError('Invalid value {!r} for additional_policy_update_action_mode configuration! Accepted values: \'gpi\', \'target_policy\''. format(self.config.additional_policy_update_action_mode))

        self.w_per_reward_func = []
        self.learn_w = False
        self.z_per_reward_func = []

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.episodes_per_reward_func_counter = []
        self.episodes_counter = 0

        self.obj_visits_per_reward_func_counter = []

    def add_reward_function(self, reward_function, reward_func_descr=None, w=None):

        self.episodes_per_reward_func_counter.append(0)
        self.obj_visits_per_reward_func_counter.append([0, 0])

        if w is None:
            # set up intial weight vector for the new reward function
            w = np.random.randn(self.config.phi_size) * self.config.w_init_std
            self.learn_w = True
        else:
            # use given w
            self.learn_w = False
        
        self.w_per_reward_func.append(w)

        # set up initial q-function for the new reward function
        if not self.z_per_reward_func or self.config.z_init_mode != 'previous':
            z = np.random.randn(self.action_space.n, self.observation_space.shape[0], self.config.phi_size) * self.config.z_init_std
        else:
            ######################################################################
            # use random number generator, because other agents are using it also
            # and to keep the rng the same (for comparison to DNN SF QL agent)
            # TODO: remove this part for final agent
            # _ = np.random.randn(self.action_space.n,
            #                     self.observation_space.shape[0],
            #                     self.config.phi_size) * self.config.z_init_std
            ######################################################################
            z = self.z_per_reward_func[self.active_reward_func_idx].copy()
        self.z_per_reward_func.append(z)

        return len(self.z_per_reward_func) - 1


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
                self.episodes_per_reward_func_counter[self.active_reward_func_idx])
        log.add_value('agent_alpha_per_episode', self.alpha)

        # set alpha_w according to current episode
        if self.config.alpha_w_step_counter == 'total_episodes':
            self.alpha_w = utils.set_value_by_function(
                self.config.alpha_w,
                self.episodes_counter)
        elif self.config.alpha_w_step_counter == 'episodes_per_reward_function':
            self.alpha_w = utils.set_value_by_function(
                self.config.alpha_w,
                self.episodes_per_reward_func_counter[self.active_reward_func_idx])
        # log.add_value('agent_alpha_w_per_episode', self.alpha_w)

        # set epsilon according to current episode
        if self.config.epsilon_step_counter == 'total_episodes':
            self.epsilon = utils.set_value_by_function(
                self.config.epsilon,
                self.episodes_counter)
        elif self.config.epsilon_step_counter == 'episodes_per_reward_function':
            self.epsilon = utils.set_value_by_function(
                self.config.epsilon,
                self.episodes_per_reward_func_counter[self.active_reward_func_idx])
        log.add_value('agent_epsilon_per_episode', self.epsilon)

        # increase internal counters of episodes
        self.episodes_per_reward_func_counter[self.active_reward_func_idx] += 1
        self.episodes_counter += 1

    def calc_own_max_action(self, state):
        psis = state @ self.z_per_reward_func[self.active_reward_func_idx]
        q_values = psis @ self.w_per_reward_func[self.active_reward_func_idx]
        _, max_action = utils.select_max_value(q_values)
        return max_action

    def step(self, state, info):
        '''Draws an epsilon-greedy policy'''

        if np.random.rand() < self.epsilon:
            action = np.random.choice(np.arange(self.action_space.n))
        elif self.config.dogpi:
            action, _ = self.calc_max_action(state)
        else:
            action = self.calc_own_max_action(state)

        return action


    def update(self, transition):
        """Use the given transition to update the agent."""

        # extract transition information
        state, action, next_state, reward, done, info = transition
        phi = self.config.phi_func(transition)

        if phi[0]:
            self.obj_visits_per_reward_func_counter[self.active_reward_func_idx][0] += 1
        
        if phi[1]:
            self.obj_visits_per_reward_func_counter[self.active_reward_func_idx][1] += 1

        if self.learn_w:
            # update the reward weights
            w = self.w_per_reward_func[self.active_reward_func_idx]
            w = w + self.alpha_w * (reward - np.matmul(phi, w)) * phi
            self.enforce_weight_maximum(w)
            self.w_per_reward_func[self.active_reward_func_idx] = w

        # set gamma to 0 if goal state is reached
        gamma = self.config.gamma if not done else 0

        # select which policies are updated
        # the active policy is always updated
        update_policies_idxs = [self.active_reward_func_idx]

        if self.config.dogpi and len(self.z_per_reward_func) > 1:

            if self.config.policy_update_mode == 'active_and_gpi_optimal':
                # get task from which the current action was taken
                _, c = self.calc_max_action(state)
                if c != self.active_reward_func_idx:
                    update_policies_idxs.append(c)

            elif self.config.policy_update_mode == 'active_and_optimal':
                # identify the policies, for which the current action is also optimal
                for p_idx in range(len(self.z_per_reward_func)):
                    if p_idx != self.active_reward_func_idx:
                        a, _ = self.calc_max_action(state, policy_idx=p_idx, target_reward_func_idx=p_idx)
                        if a == action:
                            update_policies_idxs.append(p_idx)

            elif self.config.policy_update_mode == 'all':
                # update all policies (besides the current)
                additional_policies_idxs = np.arange(len(self.z_per_reward_func)).tolist()
                del additional_policies_idxs[self.active_reward_func_idx]

                update_policies_idxs.extend(additional_policies_idxs)


        for p_idx in update_policies_idxs:
            # which action should be used as optimal next action ?
            # if active policy: use GPI procedure
            # if other policy: depends on additional_policy_update_action_mode config
            if p_idx == self.active_reward_func_idx or self.config.additional_policy_update_action_mode == 'gpi':
                if self.config.dogpi:
                    next_action, _ = self.calc_max_action(next_state, target_reward_func_idx=p_idx)
                else:
                    next_action = self.calc_own_max_action(next_state)
            else:
                # self.config.additional_policy_update_action_mode == 'target_policy':
                next_action, _ = self.calc_max_action(next_state, policy_idx=p_idx, target_reward_func_idx=p_idx)

            z_weight = self.z_per_reward_func[p_idx]

            current_psi = np.matmul(state, z_weight[action, :])
            next_psi = np.matmul(next_state, z_weight[next_action, :])

            for k in range(self.config.phi_size):
                z_weight[action, :, k] += self.alpha * (phi[k] + gamma * next_psi[k] - current_psi[k]) * state

            self.enforce_weight_maximum(z_weight)


    def calc_max_action(self, state, policy_idx=None, target_reward_func_idx=None):
        '''Get the optimal action for a given reward function'''

        if policy_idx is None:
            policy_idx = list(range(len(self.z_per_reward_func)))
        elif not isinstance(policy_idx, list):
            policy_idx = [policy_idx]

        if target_reward_func_idx is None:
            target_reward_func_idx = self.active_reward_func_idx

        # calculate the Q-values for all actions
        #Q = phi * z * w
        q_values = np.zeros((len(policy_idx), self.action_space.n))
        for idx in range(q_values.shape[0]):
            psi = np.matmul(state, self.z_per_reward_func[policy_idx[idx]])
            q_values[idx, :] = np.matmul(psi, self.w_per_reward_func[target_reward_func_idx])

        # identify the optimal policy and action
        # select action with highest return over all policies
        max_value = np.max(q_values)
        where_max_value = np.where(q_values == max_value)
        n_max_values = len(where_max_value[0])
        if n_max_values == 1:
            selected_val_idx = 0
        else:
            selected_val_idx = np.random.randint(n_max_values)
        max_policy_idx = where_max_value[0][selected_val_idx]
        max_action = where_max_value[1][selected_val_idx]

        return max_action, policy_idx[max_policy_idx]


    def calc_psi_function(self, state, action=None, reward_function_idx=None):

        if reward_function_idx is None:
            reward_function_idx = self.active_reward_func_idx

        if action is None:
            return np.array([state @ self.z_per_reward_func[reward_function_idx][action,:,:] for action in range(self.action_space.n)])

        psi = np.matmul(state, self.z_per_reward_func[reward_function_idx][action,:,:])
        return psi


    def calc_expected_return(self, state, action, reward_function_idx=None):

        if reward_function_idx is None:
            reward_function_idx = self.active_reward_func_idx

        q_values = []
        for idx in range(len(self.z_per_reward_func)):
            psi = np.matmul(state, self.z_per_reward_func[idx][action, :, :])
            q_values.append(np.matmul(psi, self.w_per_reward_func[reward_function_idx]))

        return np.max(q_values)


    def enforce_weight_maximum(self, weights):

        weight_above_maximum_inds = weights > self.config.absolute_weight_maximum
        if np.any(weight_above_maximum_inds):
            weights[weight_above_maximum_inds] = self.config.absolute_weight_maximum
            warnings.warn('Some weights have reached the maximum value of {}.'.format(self.config.absolute_weight_maximum))

        weight_above_maximum_inds = weights < -1 * self.config.absolute_weight_maximum
        if np.any(weight_above_maximum_inds):
            weights[weight_above_maximum_inds] = -1 * self.config.absolute_weight_maximum
            warnings.warn('Some weights have reached the maximum value of {}.'.format(-1 * self.config.absolute_weight_maximum))