# from https://gitlab.inria.fr/creinke/transfer_rl

from utils.spaces import num_flat_space_elements
from utils.spaces import nd_to_flat_space_index
from utils.spaces import flat_to_nd_space_index
from utils.set_values import set_value_by_function
from utils.set_values import linear_value_adaptation
from utils.similarity_search import inverse_distance_kernel
from utils.similarity_search import knn_search
from utils.similarity_search import combine_by_key

import numpy as np
import torch
import random

# correct solution:
def softmax(x):
    # """Compute softmax values for each sets of scores in x."""
    # e_x = np.exp(x - np.max(x))
    # return (e_x.transpose() / e_x.transpose().sum(axis=0)).transpose()
    x = np.array(x)

    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))

    elif x.ndim == 2:
        e_x = np.exp(x - np.tile(np.max(x, axis=1), (x.shape[1], 1)).transpose())

    else:
        raise NotImplementedError('Softmax function supports only vectors or matrices with 2 dimensions!')

    return (e_x.transpose() / e_x.transpose().sum(axis=0)).transpose()


def seed(seed=None, is_set_random=True, is_set_numpy=True, is_set_torch=True):
    '''
    Sets the random seed for random, numpy and pytorch.

    :param seed: Seed (integer) or configuration dictionary which contains a 'seed' property.
                 If None is given, a seed is chosen via torch.seed().
    :param is_set_random: Should random seed of random be set. (default=True)
    :param is_set_numpy: Should random seed of numpy.random be set. (default=True)
    :param is_set_torch: Should random seed of torch be set. (default=True)
    :return: Seed that was set.
    '''

    if seed is None:
        seed = torch.seed()
    elif isinstance(seed, dict):
        seed = seed.get('seed', None)

    if is_set_numpy:
        np.random.seed(seed)

    if is_set_random:
        random.seed(seed)

    if is_set_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


def select_max_value(values):
    '''
    Returns the maximum value and its idx in the given numpy array. Selects randomly one of the values if there exist
    several maxima (uses np.random.choice).
    '''

    max_value = np.max(values)
    max_value_idxs = np.where(values == max_value)[0]
    if len(max_value_idxs) == 1:
        idx = max_value_idxs[0]
    else:
        idx = np.random.choice(max_value_idxs)

    return max_value, idx
    

def up_left_to_goal_policy(position):
    '''
    Hard-coded policy to go to object 0 and then to goal.

    Used for testing how agents learn SFs if given perfect behavioural policy
    '''

    x, y = position

    if y < 0.7:
        return 0 # go up
    elif y > 0.9:
        return 2 # go down
    elif x < 0.7:
        return 1 # go right
    else:
        return 3 # go left


def pick_obj_0_and_go_to_goal_policy(position, obs):
    '''
    Hard-coded policy to go to object 0 and then to goal.

    Expects agent to be in different parts of the space sometimes

    Used for testing how agents learn SFs if given eps-greedy perfect behavioural policy
    '''

    x, y = position

    has_picked_object_0 = bool(obs[-3]) # contains information on if the object has been picked

    if y < 0.7:
        if x >= 0.7 and not has_picked_object_0: # close to goal but has not picked object 0
            return 3 # go left to avoid goal until you pick object
        return 0 # go up
    elif y > 0.9:
        if x >= 0.7 and not has_picked_object_0: # close to goal but has not picked object 0
            return 3 # go left to avoid goal until you pick object
        return 2 # go down
    elif x < 0.2:
        return 1 # go right
    elif not has_picked_object_0:
        if x > 0.7: # conservative condition, 0.9 should be correct; avoid going to goal before picking object 0
            return 2 # go down 
        return 3 # go left
    elif x < 0.7:
        return 1 # go right
    else:
        return 3 # go left

