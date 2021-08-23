import exputils as eu
import numpy as np

def set_value_by_function(value_config, *args, **kwargs):

    # default: return the given config if it is not a dict and has a function argument
    retval = value_config

    if isinstance(value_config, dict) and 'function' in value_config:

        # all items in the value_config dict are the parameters to the function, besides the function itself
        value_config_function_params = value_config.copy()
        del value_config_function_params['function']

        function_paramters = eu.combine_dicts(kwargs, value_config_function_params)

        retval = value_config['function'](*args, **function_paramters)

    return retval


def linear_value_adaptation(step, init_value=1.0, final_value=0.0, start_step=0, end_step=None, delta=None):

    if end_step is not None and delta is not None:
        raise ValueError('Only the end_step or the delta parameter can be set for linear_discounting, not both!')

    if end_step is not None:
        delta = (final_value - init_value) / (end_step - start_step)

    value = init_value

    if step > start_step:
        n_steps = step - start_step

        value = init_value + delta * n_steps

        if init_value >= final_value:
            value = max(final_value, value)
        else:
            value = min(final_value, value)

    return value






