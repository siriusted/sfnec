# from https://gitlab.inria.fr/creinke/transfer_rl


import exp.core
import exputils as eu
import exputils.data.logging as log
import numpy as np
import gym

def run_experiment(config=None, **kwargs):
    '''
    Runs RL experiments in a continual transfer setting between tasks with changing reward function.

    Experiment is splitted in different phases. In each phase a different reward function is active for the environment.
    Agents are informed about a change of the phase and the corresponding reward function via:
        > new_rfunc_idx = agent.add_reward_function(rfunc)
        > agent.set_active_reward_func_idx(new_rfunc_idx)
    where rfunc is a handle to a reward function.
    '''

    default_config = eu.AttrDict(
        env_class = None,
        env_config = None,

        agent_class = None,
        agent_config = None,

        reward_functions = None,
        reward_weights = None,

        n_episodes_per_phase = 100,
        n_max_steps = 1000,
        n_max_steps_per_epsiode=None,

        log_functions = [],

        log_name_episode_per_phase = 'episode_per_phase',
        log_name_step_per_phase = 'step_per_phase',
        log_name_reward_per_phase = 'reward_per_phase',
        log_name_total_reward = 'total_reward',

        logger = None
    )
    config = eu.combine_dicts(kwargs, config, default_config)

    if config.logger is not None:
        original_log = log.get_log()
        log.set_log(config.logger)

    try:
        # generate env
        if type(config.env_class) == str: # gym builtin envs
            env = gym.make(config.env_class)
            env.seed(config.seed)
        else:
            env = config.env_class(**config.env_config)

        # generate agent
        agent = config.agent_class(env, **config.agent_config)


        # properly default reward_weights if not given
        if config.reward_weights is None:
            config.reward_weights = [None] * len(config.reward_functions)

        # loop over reward_functions (phases)
        for phase_idx, rfunc in enumerate(config.reward_functions):

            if isinstance(rfunc, tuple):
                rfunc_handle = rfunc[0]
                rfunc_descr = rfunc[1]
            else:
                rfunc_handle = rfunc
                rfunc_descr = None

            env.reward_function = rfunc_handle

            if rfunc_descr is None:
                agent_rfunc_idx = agent.add_reward_function(rfunc_handle, w=config.reward_weights[phase_idx])
            else:
                agent_rfunc_idx = agent.add_reward_function(rfunc_handle, reward_func_descr=rfunc_descr, w=config.reward_weights[phase_idx])
            agent.set_active_reward_func_idx(agent_rfunc_idx)

            transfer_rl.exp.core.run_rl_training(
                env,
                agent,
                n_episodes=config.n_episodes_per_phase,
                n_max_steps=config.n_max_steps,
                n_max_steps_per_epsiode=config.n_max_steps_per_epsiode,
                log_name_episode=config.log_name_episode_per_phase,  # rename some default log poperty names
                log_name_step=config.log_name_step_per_phase,
                log_functions=config.log_functions)

            # if hasattr(agent, 'prepare_for_storage'):
            #     agent_to_store = agent.prepare_for_storage()
            #     log.add_single_object('agent_phase_{}'.format(phase_idx), agent_to_store)
            # else:
                # log.add_single_object('agent_phase_{}'.format(phase_idx), agent)

            log_phase_counters(config, phase_idx)

        log_reward_per_phase(config, phase_idx)
        log_total_reward(config)
        log_obj_visit_counters(agent)

    finally:
        # reset to original log
        this_log = log.get_log()
        if config.logger is not None:
            log.set_log(original_log)

    return agent, this_log


def log_phase_counters(config, phase_idx):

    # add extra counters
    if log.contains('step_per_phase'):

        # identify how many new steps were made during the phase
        if log.contains('phase_per_step'):
            n_new_steps = len(log.get_values('step_per_phase')) - len(log.get_values('phase_per_step'))
        else:
            n_new_steps = len(log.get_values('step_per_phase'))

        # what was the total final step over all phases
        final_last_step = log.get_values('step')[-1] if log.contains('step') else -1

        # add counters regarding phase and steps
        for step_idx in range(n_new_steps):
            log.add_value('phase_per_step', phase_idx)
            log.add_value('step', final_last_step + 1 + step_idx)

    if log.contains('episode_per_phase'):

        # identify how many new episodes were made during the phase
        if log.contains('phase_per_episode'):
            n_new_episodes = len(log.get_values('episode_per_phase')) - len(log.get_values('phase_per_episode'))
        else:
            n_new_episodes = len(log.get_values('episode_per_phase'))

        # what was the total final episode over all phases
        final_last_episode = log.get_values('episode')[-1] if log.contains('episode') else -1

        # add counters regarding phase and episodes
        for episode_idx in range(n_new_episodes):
            log.add_value('phase_per_episode', phase_idx)
            log.add_value('episode', final_last_episode + 1 + episode_idx)


def log_reward_per_phase(config, phase_idx):

    # convert lists to numpy arrays
    phase_per_episode = np.array(log.get_values('phase_per_episode'))
    reward_per_episode = np.array(log.get_values('reward_per_episode'))

    phase_idxs = np.unique(phase_per_episode)
    for phase_idx in phase_idxs:
        # compute reward sum over episodes
        log.add_value(
            config.log_name_reward_per_phase,
            np.sum(reward_per_episode[phase_per_episode == phase_idx]))


def log_total_reward(config):
    log.add_value(
        config.log_name_total_reward,
        np.sum(log.get_values(config.log_name_reward_per_phase)))


def log_obj_visit_counters(agent):
    '''Log visitation counts and frequencies for the two objects in the environment per phase'''

    if hasattr(agent, 'obj_visits_per_reward_func_counter'):

        for phase in range(len(agent.obj_visits_per_reward_func_counter)):

            for obj in range(2):
                log_prefix = f'phase_{phase}_obj{obj}_'
                n_visits = agent.obj_visits_per_reward_func_counter[phase][obj]
                log.add_value(f'{log_prefix}visits', n_visits)
                log.add_value(f'{log_prefix}visit_frequency', n_visits / agent.episodes_per_reward_func_counter[phase])