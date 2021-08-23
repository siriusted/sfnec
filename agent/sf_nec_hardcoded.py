import numpy as np
import exputils as eu
import utils
import exputils.data.logging as log
import torch
from approximation import SFNEC, ExperienceReplay
import warnings

class SFNECHardCodedAgent:

    @staticmethod
    def default_config():
        return eu.AttrDict(
            initial_epsilon = 1,
            final_epsilon = 0.005,
            epsilon_anneal_start = 1,
            epsilon_anneal_end = 100,
            start_learning_step = 1,
            eval_epsilon = 0,
            alpha_w=0.05,
            key_size = 4,
            dnd_capacity = 2000,
            num_neighbours= 20,
            dnd_alpha = 0.2,
            batch_size = 1,
            gamma = 0.95,
            n_step_horizon = 1,
            replay_frequency = 1,
            replay_buffer_size = 1,
            weight_init_func=None,
            episodic_update=False,
            reexplore_tasks=False,
            update_gpi_optimal_policy=True,
            action_selection='gpi', # or 'single'
            bootstrap_psi_mode='gpi', # or ql
            w_init_std = 0.01,
            psi_init_std=0.01,
            psi_init_mode='zeros', # or 'random'
            phi_func = lambda transition: transition[5]['feature'], # [5] is the info component of transitions
            phi_size = 4,
            absolute_weight_maximum = 1000,  # maximum absolute value of weights used for w
            device = torch.device('cpu'),

            nec = eu.AttrDict(
                embedding_net = eu.AttrDict(
                    cls=torch.nn.Linear,
                ),

                optimizer = eu.AttrDict(
                    cls=torch.optim.SGD,
                    lr=0.01,
                ),

                loss_func = eu.AttrDict(
                    cls=torch.nn.MSELoss,
                ),

                replay_buffer = eu.AttrDict(
                    cls=ExperienceReplay,
                ),

                n_iterations_per_training = 1,

                init_mode='random',     # 'random' - clear everything, 'keep_embedding', 'keep_nec', 'keep_dnd'
            ),
        )

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.episodes = 0
        self.steps = 0
        self.train_epsilon = self.config.initial_epsilon
        self.active_reward_func_idx = None
        self.w_per_reward_func = []
        self.learn_w = False
        self.nec_per_reward_func = []
        self.episodes_per_reward_func_counter = []
        self.obj_visits_per_reward_func_counter = []

        self.gpi_optimal_policy_idx = None
        self.gpi_optimal_policy_optimizer = None

        self._update_func = self._update_after_episode if self.config.episodic_update else self._update_after_horizon


    def train(self):
        self.training = True
        self.nec_net.train()

    def eval(self):
        self.training = False
        self.nec_net.eval()

    def new_episode(self, obs, info):
        # trackers for computing N-step returns and updating replay and dnd memories at the end of episode
        self.observations, self.keys, self.actions, self.phis, self.gpi_psis = [], [], [], [], []
        self.episodes += 1
        self.episodes_per_reward_func_counter[self.active_reward_func_idx] += 1

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

        if self.config.reexplore_tasks:
            self.steps = 0 # reset steps to reexplore with epsilon-annealing

        # set up nec for the new reward function
        embedding_net = eu.misc.call_function_from_config(
            self.config.nec.embedding_net,
            self.observation_space.shape[0],
            self.config.key_size,
            func_attribute_name='cls'
        )

        # assumes embedding_net is a single linear layer
        if self.config.weight_init_func is not None:
            eu.misc.call_function_from_config(self.config.weight_init_func, embedding_net.weight)

        sfnec_config = eu.combine_dicts(dict(value_size=self.config.phi_size), self.config)
        nec_net = SFNEC(embedding_net, self.action_space.n, sfnec_config).to(self.config.device)

        # copy the parameters of the previous nec if required
        if self.nec_per_reward_func:
            if self.config.nec.init_mode == 'keep_embedding':
                # print('Keeping embedding')
                nec_net.embedding_net.load_state_dict(self.nec_per_reward_func[-1].embedding_net.state_dict())
            elif self.config.nec.init_mode == 'keep_dnd':
                # print('Keeping dnd')
                nec_net.dnds.load_state_dict(self.nec_per_reward_func[-1].dnds.state_dict())
            elif self.config.nec.init_mode == 'keep_nec':
                # print('Keeping nec')
                nec_net.load_state_dict(self.nec_per_reward_func[-1].state_dict())

        self.nec_per_reward_func.append(nec_net)
        self.nec_net = nec_net

        # set up loss function, replay buffer and optimizer
        self.loss_func = eu.misc.call_function_from_config(
            self.config.nec.loss_func,
            func_attribute_name='cls'
        )

        self.replay_buffer = eu.misc.call_function_from_config(
            self.config.nec.replay_buffer,
            self.observation_space.shape,
            self.config.replay_buffer_size,
            (self.config.phi_size, ), # returns_shape
            func_attribute_name='cls'
        )

        self.optimizer = eu.misc.call_function_from_config(
            self.config.nec.optimizer,
            self.nec_net.parameters(),
            func_attribute_name='cls'
        )

        # set network to training mode
        self.train()

        # return new rfunc index
        return len(self.nec_per_reward_func) - 1

    def set_active_reward_func_idx(self, idx):
        self.active_reward_func_idx = idx

    def step(self, obs, info = {}):
        self.steps += 1

        # do epsilon-annealing
        if self.config.epsilon_anneal_start < self.steps <= self.config.epsilon_anneal_end:
            self.train_epsilon -= (self.config.initial_epsilon - self.config.final_epsilon) / (self.config.epsilon_anneal_end - self.config.epsilon_anneal_start)

        obs = torch.from_numpy(np.float32(obs)).to(self.config.device)

        psi_values, key = self.nec_net.lookup(obs)

        eps = self.train_epsilon if self.training else self.config.eval_epsilon

        position = info['position']
        log.add_value(f'phase_{self.active_reward_func_idx}_position_x', position[0])
        log.add_value(f'phase_{self.active_reward_func_idx}_position_y', position[1])


        if self.active_reward_func_idx == 0:
            optimal_action = utils.pick_obj_0_and_go_to_goal_policy(position, obs)#utils.up_left_to_goal_policy(info['position'])
            self.gpi_optimal_policy_idx = 0
            max_psi_value = psi_values[optimal_action]
        elif self.config.action_selection == 'gpi':
            # do gpi procedure
            gpi_action, self.gpi_optimal_policy_idx = self.calc_gpi_action(obs)
            max_psi_value = psi_values[gpi_action]
        else:
            # calc max action for this policy only
            q_values = psi_values @ self.w_per_reward_func[self.active_reward_func_idx]
            _, own_greedy_action = utils.select_max_value(q_values)
            max_psi_value = psi_values[own_greedy_action]
        
        # epsilon-greedy
        if np.random.rand() < eps:
            action = np.random.choice(np.arange(self.action_space.n))
        elif self.active_reward_func_idx == 0:
            action = optimal_action
        elif self.config.action_selection == 'gpi':
            action = gpi_action
            # log.add_value(f'gpi_policy_idx_phase_{self.active_reward_func_idx}', self.gpi_optimal_policy_idx)
        else:
            action = own_greedy_action

        # update keys and values trackers
        if self.training:
            self.keys.append(key)
            self.gpi_psis.append(max_psi_value)

        return action


    def update(self, transition):
        obs, action, next_obs, reward, done, info = transition

        phi = self.config.phi_func(transition)

        if phi[0]:
            self.obj_visits_per_reward_func_counter[self.active_reward_func_idx][0] += 1
        
        if phi[1]:
            self.obj_visits_per_reward_func_counter[self.active_reward_func_idx][1] += 1

        if self.learn_w:
            # update the reward weights
            w = self.w_per_reward_func[self.active_reward_func_idx]
            w_loss = reward - (phi @ w)
            w += self.config.alpha_w * w_loss * phi
            self.enforce_weight_maximum(w)
            self.w_per_reward_func[self.active_reward_func_idx] = w # by reference semantics means this is unnecessary I believe

        # update observations, actions, and rewards trackers
        self.observations.append(obs)
        self.actions.append(action)
        self.phis.append(phi)

        self._update_func(done)

        if self.config.action_selection == 'gpi' and self.config.update_gpi_optimal_policy:
            self.update_gpi_optimal_policy(obs, action, phi, next_obs)

        # optimize network
        if self.steps >= self.config.start_learning_step and self.steps % self.config.replay_frequency == 0:
            self.optimize()

    def optimize(self, n_iterations=None):
        """
        Here, we sample from the replay buffer and train the NEC model end-to-end with backprop
        """
        if self.replay_buffer.size() >= self.config.batch_size:

            if n_iterations is None:
                n_iterations = self.config.nec.n_iterations_per_training

            for cur_iter in range(n_iterations):
                observations, actions, returns = self.replay_buffer.sample(self.config.batch_size)
                self.optimizer.zero_grad()
                psi_values = self.nec_net(observations.to(self.config.device))[range(self.config.batch_size), actions] # pick psi_values for chosen actions
                loss = self.loss_func(psi_values, returns.to(self.config.device))
                # log.add_value(f'gpi_policy_loss_phase_{self.active_reward_func_idx}', loss.item())
                loss.backward()
                self.optimizer.step()

    def _update_after_episode(self, done):
        if done:
            episode_length = len(self.phis)

            # compute N-step returns in reverse order
            returns, n_step_returns = [None] * (episode_length + 1), [None] * episode_length
            returns[episode_length] = 0

            for t in range(episode_length - 1, -1, -1):
                returns[t] = self.phis[t] + self.config.gamma * returns[t + 1]
                if episode_length - t > self.config.n_step_horizon:
                    n_step_returns[t] = returns[t] + self.config.gamma ** self.config.n_step_horizon * (self.gpi_psis[t + self.config.n_step_horizon] - returns[t + self.config.n_step_horizon])
                else: # use on-policy monte carlo returns when below horizon
                    n_step_returns[t] = returns[t]

            self.keys, n_step_returns = torch.stack(self.keys), np.array(n_step_returns) # for fancy indexing

            # batch update of replay memory
            self.replay_buffer.append_batch(np.stack(self.observations), np.asarray(self.actions, dtype = np.int64), n_step_returns)

            # batch update of episodic memories
            unique_actions = np.unique(self.actions)
            for action in unique_actions:
                action_idxs = np.nonzero(self.actions == action)[0]
                self.nec_net.update_memory(action, self.keys[action_idxs], n_step_returns[action_idxs])

    def _update_after_horizon(self, done):
        episode_length = len(self.phis)
        time_idx = episode_length - (self.config.n_step_horizon + 1) # time_idx to compute n-step return for
        isEpisodeBeyondHorizon = episode_length > self.config.n_step_horizon

        if isEpisodeBeyondHorizon:
            # compute N step return
            n_step_return = np.asarray(sum([self.config.gamma ** i * self.phis[episode_length - (self.config.n_step_horizon - i + 1)] for i in range(self.config.n_step_horizon)]) \
                            + self.config.gamma ** self.config.n_step_horizon * self.gpi_psis[episode_length - 1], dtype=np.float32)

            # update replay memory
            self.replay_buffer.append(self.observations[time_idx], self.actions[time_idx], n_step_return)

            # update episodic memory
            self.nec_net.update_memory(self.actions[time_idx], torch.stack([self.keys[time_idx]]), [n_step_return])

        if done:
            # compute on-policy monte carlo returns when below horizon for last episode_length - (horizon + 1) steps in reverse order
            returns = [None] * episode_length
            returns[-1] = self.phis[-1]

            stop_t = time_idx if isEpisodeBeyondHorizon else -1 # all n_step_returns up to this time step should have been computed

            t = -1 # default t in case there's nothing to be computed except last step

            for t in range(episode_length - 2, stop_t, -1):
                returns[t] = self.phis[t] + self.config.gamma * returns[t + 1]

            keys, actions, n_step_returns = torch.stack(self.keys[t:]), self.actions[t:], np.array(returns[t:], dtype = np.float32) # for fancy indexing

            # batch update of replay memory
            self.replay_buffer.append_batch(np.stack(self.observations[t:]), np.asarray(actions, dtype = np.int64), n_step_returns)

            # batch update of episodic memories
            unique_actions = np.unique(actions)
            for action in unique_actions:
                action_idxs = np.nonzero(actions == action)[0]
                self.nec_net.update_memory(action, keys[action_idxs], n_step_returns[action_idxs])

    def update_gpi_optimal_policy(self, obs, action, phi, next_obs):
        # print('Gpi opt policy update called..')
        if self.gpi_optimal_policy_idx != self.active_reward_func_idx:
            # print('Updating gpi opt policy')
            nec = self.nec_per_reward_func[self.gpi_optimal_policy_idx]
            w = self.w_per_reward_func[self.gpi_optimal_policy_idx]

            # ensure obs and next_obs are float32 and set up tensors with batch size 1
            obs = torch.from_numpy(np.float32(obs)).unsqueeze(0).to(self.config.device)
            next_obs = torch.from_numpy(np.float32(next_obs)).unsqueeze(0).to(self.config.device)

            self.gpi_optimal_policy_optimizer = eu.misc.call_function_from_config(
                self.config.nec.optimizer,
                nec.parameters(),
                func_attribute_name='cls'
            )

            self.gpi_optimal_policy_optimizer.zero_grad()
            psi = nec(obs.to(self.config.device))[0, action]

            # compute optimal next psi according to the policy
            next_psis = nec(next_obs.to(self.config.device))[0]
            qs = next_psis.detach().cpu().numpy() @ w
            _, max_idx = utils.select_max_value(qs)

            # compute td-target for psi-value
            target = torch.from_numpy(np.float32(phi)).to(self.config.device) + self.config.gamma * next_psis[max_idx]

            loss = self.loss_func(psi, target)
            log.add_value(f'target_gpiopt_policy_loss_phase_{self.active_reward_func_idx}', loss.item())
            loss.backward()
            self.gpi_optimal_policy_optimizer.step()

    def calc_gpi_action(self, obs):
        '''Get the gpi optimal action'''

        # print('Computing gpi procedure...')

        n_policies = len(self.nec_per_reward_func)

        # calculate the Q-values for all policies, for all actions
        #Q = psi @ w
        q_values = np.zeros((n_policies, self.action_space.n), dtype = np.float32)
        psis = np.zeros((n_policies, self.action_space.n, self.config.phi_size), dtype = np.float32)

        for idx in range(n_policies):
            psis[idx, :, :], _ = self.nec_per_reward_func[idx].lookup(obs)
            q_values[idx, :] = psis[idx, :, :] @ self.w_per_reward_func[self.active_reward_func_idx]

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

        return max_action, max_policy_idx # psis[max_policy_idx, max_action, :]

    def enforce_weight_maximum(self, weights):

        weight_above_maximum_inds = weights > self.config.absolute_weight_maximum
        if np.any(weight_above_maximum_inds):
            weights[weight_above_maximum_inds] = self.config.absolute_weight_maximum
            warnings.warn('Some weights have reached the maximum value of {}.'.format(self.config.absolute_weight_maximum))

        weight_above_maximum_inds = weights < -1 * self.config.absolute_weight_maximum
        if np.any(weight_above_maximum_inds):
            weights[weight_above_maximum_inds] = -1 * self.config.absolute_weight_maximum
            warnings.warn('Some weights have reached the maximum value of {}.'.format(-1 * self.config.absolute_weight_maximum))

    
    def get_psi_values(self, observations, actions=None):
        """
        Computes psi_values for observation, action pairs passed in.

        Used for testing
        """
        with torch.no_grad():
            self.eval()
            observations = torch.from_numpy(observations)

            psi_values = self.nec_net(observations) if actions is None else self.nec_net(observations)[range(len(actions)), actions]

            return psi_values.numpy()

    def calc_psi_function(self, obs, action, reward_function_idx=None):

        obs = torch.from_numpy(np.float32(obs)).to(self.config.device)

        if reward_function_idx is None:
            reward_function_idx = self.active_reward_func_idx

        psis, _ = self.nec_per_reward_func[reward_function_idx].lookup(obs)
        return psis[action]


    def calc_expected_return(self, obs, action, reward_function_idx=None):

        obs = torch.from_numpy(np.float32(obs)).to(self.config.device)

        if reward_function_idx is None:
            reward_function_idx = self.active_reward_func_idx

        q_values = []
        for nec in self.nec_per_reward_func:
            psis, _ = nec.lookup(obs)
            q_values.append(psis[action] @ self.w_per_reward_func[reward_function_idx])

        return np.max(q_values)