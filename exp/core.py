# from https://gitlab.inria.fr/creinke/transfer_rl

import numpy as np
import exputils as eu
import exputils.data.logging as log
import copy

def run_rl_training(env, agent, config=None, **kwargs):

    default_config = eu.AttrDict(
        n_max_steps=np.inf,
        n_episodes = np.inf,
        n_max_steps_per_epsiode=np.inf,
        logger = None,
        log_name_episode = 'episode',
        log_name_step = 'step',
        log_name_step_per_episode = 'step_per_episode',
        log_name_episode_per_step = 'episode_per_step',
        log_name_reward_per_episode = 'reward_per_episode',
        log_name_reward_per_step = 'reward_per_step',
        log_functions = [],
    )
    config = eu.combine_dicts(kwargs, config, default_config)

    if config.n_episodes == np.inf and config.n_max_steps == np.inf:
        raise ValueError('n_episodes or n_max_steps_per_epsiode must not be inf.')

    # internally log-functions are a list
    log_functions = config.log_functions
    if not isinstance(log_functions, list):
        log_functions = [log_functions]

    # set custom logger if given
    if config.logger is not None:
        original_log = log.get_log()
        log.set_log(config.logger)

    try:
        step = 0
        episode = 0
        while episode < config.n_episodes and step < config.n_max_steps:
            # step += 1
            episode += 1

            obs, info = env.reset()
            agent.new_episode(obs, info)

            done = False
            step_per_episode = 0

            # logging
            log.add_value(config.log_name_episode, episode)
            # log.add_value(config.log_name_step, step)
            # log.add_value(config.log_name_step_per_episode, step_per_episode)
            # log.add_value(config.log_name_reward_per_step, np.nan)
            # log.add_value(config.log_name_episode_per_step, episode)

            reward_per_episode = 0

            while not done and step_per_episode < config.n_max_steps_per_epsiode  and step < config.n_max_steps:

                step += 1
                step_per_episode += 1
                
                action = agent.step(obs, info)

                prev_obs = obs
                obs, reward, done, info = env.step(action)

                transition = (prev_obs, action, obs, reward, done, info)
                agent.update(transition)

                reward_per_episode += reward

                # logging
                log.add_value(config.log_name_step, step)
                log.add_value(config.log_name_step_per_episode, step_per_episode)
                log.add_value(config.log_name_reward_per_step, reward)
                log.add_value(config.log_name_episode_per_step, episode)

                # additional logging if exists
                for log_func in log_functions:
                    log_func(
                        log,
                        env=env,
                        agent=agent,
                        step=step,
                        episode=episode,
                        step_per_episode=step_per_episode,
                        transition=transition,
                    )

            log.add_value(config.log_name_reward_per_episode, reward_per_episode)

    finally:
        # reset to original logger
        this_log = log.get_log()
        if config.logger is not None:
            log.set_log(original_log)

    return this_log



def record_agent_behavior(env, agent, config=None, **kwargs):

    default_config = eu.AttrDict(
        n_episodes = 100,
        n_max_steps = 1000,
        logger = None,
        env_copy_mode = 'deepcopy',
        agent_copy_mode = 'deepcopy',
        is_update_agent = False,
        log_name_episode = 'episode',
        log_name_step = 'step',
        log_name_step_per_episode = 'step_per_episode',
        log_name_episode_per_step = 'episode_per_step',
        log_name_reward_per_episode = 'reward_per_episode',
        log_name_reward_per_step = 'reward_per_step'
    )
    config = eu.combine_dicts(kwargs, config, default_config)

    if config.env_copy_mode == 'deepcopy':
        env = copy.deepcopy(env)

    if config.agent_copy_mode == 'deepcopy':
        agent = copy.deepcopy(agent)

    # set custom logger if given
    if config.logger is not None:
        original_log = log.get_log()
        log.set_log(config.logger)

    try:
        step = -1
        for episode_idx in range(config.n_episodes):
            step += 1

            obs, info = env.reset()
            state = env.get_state()
            agent.new_episode(obs, info)

            done = False
            step_per_episode = 0

            # logging
            log.add_value(config.log_name_episode, episode_idx)
            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_step_per_episode, step_per_episode)
            log.add_value(config.log_name_reward_per_step, np.nan)
            log.add_value(config.log_name_episode_per_step, episode_idx)

            reward_per_episode = 0

            while not done and step_per_episode < config.n_max_steps:

                action = agent.step(obs, info)

                prev_obs = obs
                prev_done = done
                prev_info = info
                prev_state = state
                obs, reward, done, info = env.step(action)
                state = env.get_state()

                log.add_object('state', prev_state)
                log.add_object('obs', prev_obs)
                log.add_object('done', prev_done)
                log.add_object('info', prev_info)
                log.add_object('action', action)
                log.add_object('reward', reward)

                transition = (prev_obs, action, obs, reward, done, info)
                # default: no agent update during a test
                if config.is_update_agent:
                    agent.update(transition)

                reward_per_episode += reward

                step += 1
                step_per_episode += 1

                # logging
                log.add_value(config.log_name_step, step)
                log.add_value(config.log_name_step_per_episode, step_per_episode)
                log.add_value(config.log_name_reward_per_step, reward)
                log.add_value(config.log_name_episode_per_step, episode_idx)

            # log final state
            log.add_object('state', state)
            log.add_object('obs', obs)
            log.add_object('done', done)
            log.add_object('info', info)

            log.add_value(config.log_name_reward_per_episode, reward_per_episode)

    finally:
        # reset to original logger
        this_log = log.get_log()
        if config.logger is not None:
            log.set_log(original_log)

    return this_log