# from https://gitlab.inria.fr/creinke/transfer_rl

import numpy as np
# import transfer_rl
import utils
import exputils as eu
import exputils.data.logging as log
import torch
import gym


class DNNSFQLearning:
    """
    Successor Feature Q-learning agent for continual reward transfer scenarios that uses deep neural networks to
    approximate the Psi-function. Based on SF QL agent in paper [Barreto 2018], but uses DNN approximation for the
    psi function.

    The default configuration replicates the SF-QL algorithm as used in the 2D object collection task by [Barreto 2018].

    Config parameters:

        is_set_initial_reward_weights:  True if initial reward weights are set. In this case if the add_reward_function method
                                        is called, the reward_function_descr parameter must be a dictionary with a 'reward_weight_vector'
                                        field that contains the reward weight vector. If False, the weights are randomly initialized.
                                        (default=False)

        psi_func:
            init_mode: How to initialize a new psi function. (default='previous')
                - 'random': Use random initialization from Psi-function model.
                - 'previous': Reuse the weights from the Psi-function model that was used for the previous reward
                              function.

            train_mode: Different modes that define which psi functions are trained during a training iteration.
                        (default='active_and_gpi_optimal')
                - 'all': all Psi-functions are trained.
                - 'active': only the active Psi-function is trained.
                - 'active_and_gpi_optimal': The active policy and the policies that provide the optimal actions for the
                                            sampled batch of transitions.
                                            This is the strategy used in the Barreto2018 paper.
                - 'active_and_optimal': The active policy and the policies for which the actions in the sampled batch of
                                        transitions is the optimal one.

            additional_policy_action_mode: Which action for the next state is used to update additional policies.
                Note: The active policy is always updated using GPI. (default='target_policy')
                - 'gpi': Optimal next action according to GPI.
                - 'target_policy': Optimal next action only according to the policy that gets updated.
                                   This is the strategy used in the Barreto2018 paper.
    """

    @staticmethod
    def default_config():
        dc = eu.AttrDict(
            is_set_initial_reward_weights=False,
            alpha_w=0.05,
            alpha_w_step_counter='episodes_per_reward_function',  # total_episodes, episodes_per_reward_function
            w_init_std=0.01,

            epsilon=0.15,  # can also be a dict that defines the function to get epsilon
            epsilon_step_counter='total_episodes',  # total_episodes

            gamma=0.95,

            get_feature_func=None,
            feature_space=None,

            psi_func=eu.AttrDict(
                model=eu.AttrDict(
                    cls=transfer_rl.approximation.ActionSeparatedLinearNN,
                ),

                optimizer=eu.AttrDict(
                    cls=torch.optim.SGD,
                    lr=0.01,
                ),

                loss=eu.AttrDict(
                    cls=torch.nn.MSELoss,
                ),

                replay_buffer=eu.AttrDict(
                    cls=transfer_rl.approximation.ReplayBuffer,
                ),

                n_iterations_per_training=1,

                init_mode='previous',  # 'random' or 'previous': random crea

                train_mode='active_and_gpi_optimal',

                additional_policy_action_mode='target_policy',

                gradient_clipping_value=1.0,
            ),

            log_reward_model_error=False,

        )
        return dc

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        # handle alpha_w from config
        if self.config.alpha_w_step_counter is not None and self.config.alpha_w_step_counter not in ['total_episodes', 'episodes_per_reward_function',
                                                                                                     '']:
            raise ValueError(
                'Invalid value {!r} for alpha_w_step_counter configuration! Accepted values: \'total_episodes\', \'episodes_per_reward_function\'.'.format(
                    self.config.alpha_w_step_counter))

        if not self.config.alpha_w_step_counter:
            self.alpha_w = self.config.alpha_w
        else:
            self.alpha_w = 0.0

        # handle epsilon from config
        if self.config.epsilon_step_counter is not None and self.config.epsilon_step_counter not in ['total_episodes', 'episodes_per_reward_function',
                                                                                                     '']:
            raise ValueError(
                'Invalid value {!r} for epsilon_step_counter configuration! Accepted values: \'total_episodes\', \'episodes_per_reward_function\'.'.format(
                    self.config.epsilon_step_counter))

        if not self.config.epsilon_step_counter:
            self.epsilon = self.config.epsilon
        else:
            self.epsilon = 0.0

        if not self.config.psi_func.init_mode in ['random', 'previous']:
            raise ValueError('Invalid configuration parameter: psi_func.init_mode ({!r}) can only be \'random\', \'previous\'!'.format(
                self.config.psi_func.init_mode))

        if not self.config.psi_func.train_mode in ['all', 'active', 'active_and_gpi_optimal', 'active_and_optimal']:
            raise ValueError(
                'Invalid configuration parameter: psi_func.init_mode ({!r}) can only be \'all\', \'active\', \'active_and_gpi_optimal\', \'active_and_optimal\'!'.format(
                    self.config.psi_func.train_mode))

        if not self.config.psi_func.additional_policy_action_mode in ['gpi', 'target_policy']:
            raise ValueError(
                'Invalid configuration parameter: psi_func.additional_policy_action_mode ({!r}) can only be \'gpi\', \'target_policy\'!'.format(
                    self.config.psi_func.additional_policy_action_mode))

        if self.config.get_feature_func is not None:
            self.get_feature_func = self.config.get_feature_func
        elif hasattr(env, 'get_feature_func'):
            self.get_feature_func = env.get_feature_func
        else:
            raise ValueError('No get_feature_func defined either in configuration or by the environment!')

        if self.config.feature_space is not None:
            self.feature_space = self.config.feature_space
        elif hasattr(env, 'feature_space'):
            self.feature_space = env.feature_space
        else:
            raise ValueError('No feature_space defined either in configuration or by the environment!')

        if isinstance(self.feature_space, gym.spaces.MultiDiscrete):
            self.feature_len = len(self.feature_space.nvec)
        elif isinstance(self.feature_space, gym.spaces.MultiBinary):
            self.feature_len = self.feature_space.n
        else:
            raise ValueError('Nonsupported feature_space object!')

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.active_reward_func_idx = None
        self.w_per_reward_func = []
        self.psi_func_per_reward_func = []

        # replay buffer for psi function for every reward function
        self.psi_func_replay_buffer = eu.misc.call_function_from_config(
            self.config.psi_func.replay_buffer,
            func_attribute_name='cls'
        )

        # loss function for the psi functions (default is MSE loss)
        self.psi_func_loss_func = eu.misc.call_function_from_config(
            self.config.psi_func.loss,
            func_attribute_name='cls'
        )

        self.episodes_per_reward_func_counter = []
        self.episodes_counter = 0

    def add_reward_function(self, reward_function, reward_func_descr=None, w=None):

        self.episodes_per_reward_func_counter.append(0)

        # if self.config.is_set_initial_reward_weights:
        #     # use given reward weight vector
        #     w = torch.tensor(reward_func_descr['reward_weight_vector'], dtype=torch.float32)
        # else:

        #     # set up intial weight vector for the new reward function
        #     w = torch.tensor(np.random.randn(self.feature_len) * self.config.w_init_std, dtype=torch.float32)
        if w is None:
            # set up intial weight vector for the new reward function
            w = torch.tensor(np.random.randn(self.feature_len) * self.config.w_init_std, dtype=torch.float32)
            self.learn_w = True
        else:
            # use given w
            w = torch.from_numpy(np.float32(w))
            self.learn_w = False
        
        self.w_per_reward_func.append(w)

        # set up initial q-function for the new reward function
        new_psi_func = eu.AttrDict()

        new_psi_func.model = eu.misc.call_function_from_config(
            self.config.psi_func.model,
            self.observation_space.shape[0],
            self.action_space.n,
            self.feature_len,
            func_attribute_name='cls'
        )

        # copy the parameters of the previous Q function
        if self.config.psi_func.init_mode is 'previous' and self.psi_func_per_reward_func:
            new_psi_func.model.load_state_dict(self.psi_func_per_reward_func[self.active_reward_func_idx].model.state_dict())

        new_psi_func.optimizer = eu.misc.call_function_from_config(
            self.config.psi_func.optimizer,
            new_psi_func.model.parameters(),
            func_attribute_name='cls'
        )

        self.psi_func_per_reward_func.append(new_psi_func)

        # return new rfunc index
        return len(self.psi_func_per_reward_func) - 1

    def set_active_reward_func_idx(self, idx):
        self.active_reward_func_idx = idx

    def new_episode(self, state, info):

        # set alpha_w according to current episode
        if self.config.alpha_w_step_counter == 'total_episodes':
            self.alpha_w = utils.set_value_by_function(
                self.config.alpha_w,
                self.episodes_counter)
        elif self.config.alpha_w_step_counter == 'episodes_per_reward_function':
            self.alpha_w = utils.set_value_by_function(
                self.config.alpha_w,
                self.episodes_per_reward_func_counter[self.active_reward_func_idx])
        log.add_value('agent_alpha_w_per_episode', self.alpha_w)

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

    def step(self, state, info):
        """Draws an epsilon-greedy policy"""

        if np.random.rand() < self.epsilon:
            action = np.random.choice(np.arange(self.action_space.n))
        else:
            action = self.calc_max_action_gpi(state)

        return action

    def update(self, transition):
        """Use the given transition to update the agent"""

        active_psi_func = self.psi_func_per_reward_func[self.active_reward_func_idx]

        state, action, next_state, reward, done, info = transition
        phi = torch.tensor(self.get_feature_func(transition), dtype=active_psi_func.model.dtype)

        if self.learn_w:
            # update the reward weights
            w = self.w_per_reward_func[self.active_reward_func_idx]
            reward_error = reward - torch.matmul(phi, w)
            w = w + self.alpha_w * reward_error * phi
            self.w_per_reward_func[self.active_reward_func_idx] = w

        if self.config.log_reward_model_error:
            log.add_value('reward_model_error_per_step', reward_error)

        # add transition to replay buffer for active reward function
        data = (torch.tensor(state, dtype=active_psi_func.model.dtype),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(next_state, dtype=active_psi_func.model.dtype),
                phi,
                torch.tensor(not done, dtype=torch.bool))

        self.psi_func_replay_buffer.add(data)

        # train the model
        self.train()

    def train(self, n_iterations=None):
        """
        Trains the approximator for n_iterations (either specified as attribute or in config) from samples
        """

        if len(self.psi_func_replay_buffer) >= self.psi_func_replay_buffer.config.batch_size:

            if n_iterations is None:
                n_iterations = self.config.psi_func.n_iterations_per_training

            for cur_iter in range(n_iterations):

                # get batch
                sampled_data = self.psi_func_replay_buffer.sample_and_split_as_tensors()
                state_batch, action_batch, next_state_batch, phi_batch, not_done_batch = sampled_data

                # if the psi_function should only be trained from the active function and additionally the one that
                # is optimal then identify which policies are optimal for the current sample data
                if len(self.psi_func_per_reward_func) > 1 and self.config.psi_func.train_mode == 'active_and_gpi_optimal':
                    max_policy_idxs = []
                    for state in state_batch:
                        _, max_policy_idx = self.calc_max_action_gpi(state, is_return_max_policy_idx=True)
                        max_policy_idxs.append(max_policy_idx)
                else:
                    max_policy_idxs = None

                # train the psi function for active reward function
                self._train_psi_func(self.active_reward_func_idx, sampled_data)

                if self.config.psi_func.train_mode != 'active':
                    for reward_func_idx in range(len(self.psi_func_per_reward_func)):
                        if reward_func_idx != self.active_reward_func_idx:
                            self._train_psi_func(reward_func_idx, sampled_data, max_policy_idxs=max_policy_idxs)

    def _train_psi_func(self, policy_idx, data, max_policy_idxs=None):

        state_batch, action_batch, next_state_batch, phi_batch, not_done_batch = data
        psi_func = self.psi_func_per_reward_func[policy_idx]

        if policy_idx != self.active_reward_func_idx:
            # if the current policy is an additional policy to be trained,

            sample_selection_inds = torch.ones(state_batch.shape[0], dtype=torch.bool)

            if self.config.psi_func.train_mode == 'active_and_gpi_optimal':

                # make sure that this is a numpy array, otherwise ic an not compare it elementwise
                # using max_policy_idxs != policy_idx
                max_policy_idxs = np.array(max_policy_idxs)

                # and only the ones that generate the optimal policy should be used,
                # then check for which sampled transition the current policy produces the optimal action
                # only train it on these samples
                sample_selection_inds[max_policy_idxs != policy_idx] = False

            elif self.config.psi_func.train_mode == 'active_and_optimal':

                for s_idx, state in enumerate(state_batch):
                    # calc return for action of current policy and compare it to the returns of other actions
                    cur_max_action = self.calc_max_action_for_specific_policy(
                        state,
                        policy_idx=policy_idx,
                        reward_func_idx=policy_idx)

                    if cur_max_action != action_batch[s_idx]:
                        sample_selection_inds[s_idx] = False

            state_batch = state_batch[sample_selection_inds]
            action_batch = action_batch[sample_selection_inds]
            next_state_batch = next_state_batch[sample_selection_inds]
            phi_batch = phi_batch[sample_selection_inds]
            not_done_batch = not_done_batch[sample_selection_inds]

        if len(not_done_batch) > 0:

            # calculate current state q-values for all actions
            # then select the q-values of the used actions (action_batch)
            dim_1_idxs = torch.arange(state_batch.shape[0])
            cur_psi_batch = psi_func.model(state_batch)[dim_1_idxs, action_batch, :]

            next_psi_batch = torch.zeros(cur_psi_batch.shape[0], cur_psi_batch.shape[1], device=cur_psi_batch.device)

            if torch.any(not_done_batch):
                with torch.no_grad():
                    next_psi_all_actions = psi_func.model(next_state_batch[not_done_batch])

                    if policy_idx == self.active_reward_func_idx or self.config.psi_func.additional_policy_action_mode == 'gpi':
                        # gpi
                        # idenitify the optimal actions according to the gpi for this policy and reward function
                        q_values = self.calc_expected_return(next_state_batch[not_done_batch], reward_func_idx=policy_idx)
                    else:
                        # self.config.psi_func.additional_policy_action_mode == 'target_policy':
                        q_values = self.calc_expected_return_from_psi(next_psi_all_actions, policy_idx)

                    dim_1_idxs = torch.arange(next_psi_all_actions.shape[0])
                    next_psi_batch[not_done_batch] = next_psi_all_actions[dim_1_idxs, q_values.argmax(1), :].detach()

            # Compute the expected psi values
            expected_state_action_psi = phi_batch + (self.config.gamma * next_psi_batch)

            # calc loss
            loss = self.psi_func_loss_func(cur_psi_batch, expected_state_action_psi)

            # backward pass
            psi_func.optimizer.zero_grad()
            loss.backward()

            # avoid exploding weights and gradients using gradient clipping
            if self.config.psi_func.gradient_clipping_value is not None:
                torch.nn.utils.clip_grad_value_(psi_func.model.parameters(), clip_value=self.config.psi_func.gradient_clipping_value)

            # optimize the model
            psi_func.optimizer.step()

    def calc_psi(self, state, action=None, policy_idx=None):
        """
        Calculates the Psi values for a specific policy.
        """

        if policy_idx is None:
            policy_idx = self.active_reward_func_idx

        if np.ndim(state) == 1:
            state_tensor = torch.tensor([state], dtype=torch.float)
            if action is not None:
                action_tensor = torch.tensor([action], dtype=torch.long)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            if action is not None:
                action_tensor = torch.tensor(action, dtype=torch.long)

        with torch.no_grad():
            psi_values = self.psi_func_per_reward_func[policy_idx].model(state_tensor)
            if action is not None:
                dim_1_idxs = torch.arange(psi_values.shape[0])
                psi_values = psi_values[dim_1_idxs, action_tensor, :]

        psi_values = psi_values.numpy()

        if np.ndim(state) == 1:
            psi_values = psi_values[0]

        return psi_values

    def calc_expected_return_from_psi(self, psi, reward_function_idx=None):
        """
        Psi is a torch tensor
        """
        if reward_function_idx is None:
            reward_function_idx = self.active_reward_func_idx

        return torch.matmul(psi, self.w_per_reward_func[reward_function_idx])

    def calc_expected_return(self, state, action=None, policy_idx='all', reward_func_idx=None, is_return_max_policy_idxs=False):
        """
        Calculates the expected return of policies for a reward function.

        :param state: State vector or list of vectors (either numpy or tensor).
        :param action: Action or list of actions (one per sate in the state vector list.) Either numpy or tensor.
                       If none, then expected return for all actions are returned. (default=None)
        :param policy_idx: Index of the policy for which its expected return should be computed.
                           If 'all', then the GPI is used, i.e. maximum expected return of all policies is returned per
                           action. (default='all')
        :reward_func_idx: Index of the reward function for which the expected return should be computed.
                          If None is given, then the expected return is computed for the active reward function.
                          (default=None)
        :param is_return_max_policy_idxs: Determines if the indexes of the policies whos action results in the highest
                                          return should be returned. (default=False)

        :return: Expected return and if is_return_max_policy_idxs=True then indexes of the policies whos action results
                 in it.
        """
        if reward_func_idx is None:
            reward_func_idx = self.active_reward_func_idx

        # get one fo the psi_func_models to identify its data type
        tmp_psi_func_model = self.psi_func_per_reward_func[reward_func_idx].model

        if isinstance(state, torch.Tensor):
            state_tensor = state.clone().detach()
        else:
            state_tensor = torch.tensor(state, dtype=tmp_psi_func_model.dtype)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        if action is not None:
            if isinstance(action, torch.Tensor):
                action_tensor = action.clone().detach()
            else:
                action_tensor = torch.tensor(action, dtype=torch.long)
            if action_tensor.ndim == 0:
                action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():

            if policy_idx is 'all':
                # calculate the expected reward over all policies

                expected_return = None
                if is_return_max_policy_idxs:
                    max_policy_idxs = None

                for p_idx in range(len(self.psi_func_per_reward_func)):
                    cur_psi_values = self.psi_func_per_reward_func[p_idx].model(state_tensor)
                    cur_expected_return = self.calc_expected_return_from_psi(cur_psi_values, reward_func_idx)

                    if expected_return is None:
                        expected_return = cur_expected_return
                    else:
                        expected_return = torch.max(expected_return, cur_expected_return)

                    if is_return_max_policy_idxs:
                        if max_policy_idxs is None:
                            max_policy_idxs = torch.zeros_like(cur_expected_return, dtype=torch.long)
                            max_policy_idxs[:] = p_idx
                        else:
                            max_policy_idxs[expected_return == cur_expected_return] = p_idx
            else:
                # calculate the expected reward for a specific policy
                psi_values = self.psi_func_per_reward_func[policy_idx].model(state_tensor)
                expected_return = self.calc_expected_return_from_psi(psi_values, reward_func_idx)

                if is_return_max_policy_idxs:
                    max_policy_idxs = torch.zeros_like(expected_return, dtype=torch.long)
                    max_policy_idxs[:] = policy_idx

            if action is not None:
                if action_tensor.ndim == 1:
                    action_tensor = torch.unsqueeze(action_tensor, 1)
                expected_return = expected_return.gather(1, action_tensor)

                if is_return_max_policy_idxs:
                    max_policy_idxs = max_policy_idxs.gather(1, action_tensor)

        expected_return = expected_return.numpy()
        if np.ndim(state) == 1:
            expected_return = expected_return[0]

        if is_return_max_policy_idxs:
            max_policy_idxs = max_policy_idxs.numpy()
            if np.ndim(state) == 1:
                max_policy_idxs = max_policy_idxs[0]

        if not is_return_max_policy_idxs:
            return expected_return
        else:
            return expected_return, max_policy_idxs

    def calc_max_action_gpi(self, state, reward_func_idx=None, is_return_max_policy_idx=False):
        """
        Calculate the optimal action according to generalized policy improvement, i.e. over all policies.

        :param state: State vector
        :param reward_func_idx: Index for the reward function for which the optimal action should be computed.
        :param is_return_max_policy_idx: Determines if the index of the policy whos action results in the highest
                                         return should be returned. (default=False)
        """

        if reward_func_idx is None:
            reward_func_idx = self.active_reward_func_idx

        if is_return_max_policy_idx:
            expected_return, max_policy_idxs = self.calc_expected_return(
                state,
                reward_func_idx=reward_func_idx,
                is_return_max_policy_idxs=True)

            _, max_action_idx = transfer_rl.utils.select_max_value(expected_return)

            max_policy_idx = max_policy_idxs[max_action_idx]

            return max_action_idx, max_policy_idx

        else:
            expected_return = self.calc_expected_return(
                state,
                reward_func_idx=reward_func_idx,
                is_return_max_policy_idxs=False)

            _, max_action_idx = transfer_rl.utils.select_max_value(expected_return)

            return max_action_idx

    def calc_max_action_for_specific_policy(self, state, policy_idx=None, reward_func_idx=None):
        """Calculates the optimal action of a specific policy."""

        if policy_idx is None:
            policy_idx = self.active_reward_func_idx

        if reward_func_idx is None:
            reward_func_idx = self.active_reward_func_idx

        # calc return
        expected_return = self.calc_expected_return(state, policy_idx=policy_idx, reward_func_idx=reward_func_idx)
        _, max_action_idx = transfer_rl.utils.select_max_value(expected_return)

        return max_action_idx