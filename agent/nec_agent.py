import numpy as np
import utils 
import exputils as eu
import exputils.data.logging as log
import torch
from approximation import NEC, ExperienceReplay
import copy

class NECAgent:

    @staticmethod
    def default_config():
        return eu.AttrDict(
            initial_epsilon = 1,
            final_epsilon = 0.005,
            epsilon_anneal_start = 1, #make constant 0.15
            epsilon_anneal_end = 500,
            start_learning_step = 1000,
            eval_epsilon = 0,
            key_size = 4,
            dnd_capacity = 2000,
            num_neighbours= 4,
            dnd_alpha = 0.5,
            batch_size = 1,
            gamma = 0.95,
            n_step_horizon = 1,
            replay_frequency = 1,
            replay_buffer_size = 1,
            weight_init_func=None,
            episodic_update=False,
            reexplore_tasks=False,
            device = torch.device('cpu'),

            nec = eu.AttrDict(
                embedding_net = eu.AttrDict(
                    cls=torch.nn.Linear,
                ),

                optimizer = eu.AttrDict(
                    cls=torch.optim.RMSprop,
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
        self.nec_per_reward_func = []

        self.episodes_per_reward_func_counter = []

        self._update_func = self._update_after_episode if self.config.episodic_update else self._update_after_horizon


    def train(self):
        self.training = True
        self.nec_net.train()

    def eval(self):
        self.training = False
        self.nec_net.eval()

    def new_episode(self, obs, info):
        if self.training:
            # trackers for computing N-step returns and updating replay and dnd memories at the end of episode
            self.observations, self.keys, self.actions, self.values, self.rewards = [], [], [], [], []
            self.episodes += 1
            self.episodes_per_reward_func_counter[self.active_reward_func_idx] += 1

    def add_reward_function(self, reward_function, reward_func_descr=None, w=None):
        self.episodes_per_reward_func_counter.append(0)

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

        nec_net = NEC(embedding_net, self.action_space.n, self.config).to(self.config.device)

        # copy the parameters of the previous nec if required
        if self.nec_per_reward_func:
            if self.config.nec.init_mode is 'keep_embedding':
                nec_net.embedding_net.load_state_dict(self.nec_per_reward_func[-1].embedding_net.state_dict())
            elif self.config.nec.init_mode is 'keep_dnd':
                nec_net.dnds.load_state_dict(self.nec_per_reward_func[-1].dnds.state_dict())
            elif self.config.nec.init_mode is 'keep_nec':
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

        obs = torch.from_numpy(np.float32(obs))

        q_values, key = self.nec_net.lookup(obs.to(self.config.device))

        eps = self.train_epsilon if self.training else self.config.eval_epsilon

        max_q_value, max_idx = utils.select_max_value(np.array(q_values))

        # epsilon-greedy
        action = np.random.choice(np.arange(self.action_space.n)) if np.random.rand() < eps else max_idx

        # update keys and values trackers
        if self.training:
            self.keys.append(key)
            self.values.append(max_q_value)

        return action


    def update(self, transition):
        obs, action, next_obs, reward, done, info = transition

        # update observations, actions, and rewards trackers
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

        self._update_func(done)

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
                q_values = self.nec_net(observations.to(self.config.device))[range(self.config.batch_size), actions] # pick q_values for chosen actions
                loss = self.loss_func(q_values, returns.to(self.config.device))
                loss.backward()
                self.optimizer.step()

    def get_q_values(self, observations, actions):
        """
        Computes q_values for observation, action pairs passed in.

        Used for testing
        """
        with torch.no_grad():
            self.eval()
            observations = torch.from_numpy(observations)
            q_values = self.nec_net(observations)[range(len(actions)), actions]

            return q_values.numpy()

    def _update_after_episode(self, done):
        if done:
            episode_length = len(self.rewards)

            # compute N-step returns in reverse order
            returns, n_step_returns = [None] * (episode_length + 1), [None] * episode_length
            returns[episode_length] = 0

            for t in range(episode_length - 1, -1, -1):
                returns[t] = self.rewards[t] + self.config.gamma * returns[t + 1]
                if episode_length - t > self.config.n_step_horizon:
                    n_step_returns[t] = returns[t] + self.config.gamma ** self.config.n_step_horizon * (self.values[t + self.config.n_step_horizon] - returns[t + self.config.n_step_horizon])
                else: # use on-policy monte carlo returns when below horizon
                    n_step_returns[t] = returns[t]

            self.keys, n_step_returns = torch.stack(self.keys), np.array(n_step_returns, dtype = np.float32) # for fancy indexing

            # batch update of replay memory
            self.replay_buffer.append_batch(np.stack(self.observations), np.asarray(self.actions, dtype = np.int64), n_step_returns)

            # batch update of episodic memories
            unique_actions = np.unique(self.actions)
            for action in unique_actions:
                action_idxs = np.nonzero(self.actions == action)[0]
                self.nec_net.update_memory(action, self.keys[action_idxs], n_step_returns[action_idxs])

    def _update_after_horizon(self, done):
        episode_length = len(self.rewards)
        time_idx = episode_length - (self.config.n_step_horizon + 1) # time_idx to compute n-step return for
        isEpisodeBeyondHorizon = episode_length > self.config.n_step_horizon

        if isEpisodeBeyondHorizon:
            # compute N step return
            n_step_return = np.float32(sum([self.config.gamma ** i * self.rewards[episode_length - (self.config.n_step_horizon - i + 1)] for i in range(self.config.n_step_horizon)]) \
                            + self.config.gamma ** self.config.n_step_horizon * self.values[episode_length - 1])

            # update replay memory
            self.replay_buffer.append(self.observations[time_idx], self.actions[time_idx], n_step_return)

            # update episodic memory
            self.nec_net.update_memory(self.actions[time_idx], torch.stack([self.keys[time_idx]]), [n_step_return])

        if done:
            # compute on-policy monte carlo returns when below horizon returns below horizon for last episode_length - (horizon + 1) steps in reverse order
            returns = [None] * episode_length
            returns[-1] = self.rewards[-1]

            stop_t = time_idx if isEpisodeBeyondHorizon else -1 # all n_step_returns up to this time step should have been computed

            t = -1 # default t in case there's nothing to be computed except last step

            for t in range(episode_length - 2, stop_t, -1):
                returns[t] = self.rewards[t] + self.config.gamma * returns[t + 1]

            keys, actions, n_step_returns = torch.stack(self.keys[t:]), self.actions[t:], np.array(returns[t:], dtype = np.float32) # for fancy indexing

            # batch update of replay memory
            self.replay_buffer.append_batch(np.stack(self.observations[t:]), np.asarray(actions, dtype = np.int64), n_step_returns)

            # batch update of episodic memories
            unique_actions = np.unique(actions)
            for action in unique_actions:
                action_idxs = np.nonzero(actions == action)[0]
                self.nec_net.update_memory(action, keys[action_idxs], n_step_returns[action_idxs])

    def calc_q_function(self, obs, action=None, reward_function_idx=None):

        obs = torch.from_numpy(np.float32(obs)).to(self.config.device)

        if reward_function_idx is None:
            reward_function_idx = self.active_reward_func_idx

        q_values, _ = self.nec_per_reward_func[reward_function_idx].lookup(obs)

        if action is None:
            return q_values
        
        return q_values[action]
    
    def prepare_for_storage(self):
        agent = copy.deepcopy(self)
        self.loss_func = self.replay_buffer = self.optimizer = None
        agent.observations = agent.keys = agent.actions = agent.rewards = agent.values = None
        agent.loss_func = agent.replay_buffer = agent.optimizer = None
        agent.episodes_per_reward_func_counter = None
        return agent