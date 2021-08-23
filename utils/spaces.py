# from https://gitlab.inria.fr/creinke/transfer_rl

from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
import numpy as np

def flat_to_nd_space_index(index, space):
    nd_index = None
    if isinstance(space, Discrete):
        nd_index = index
    elif isinstance(space, MultiDiscrete):
        nd_index = np.unravel_index(index, space.nvec)

    return nd_index

def nd_to_flat_space_index(index, space):
    flat_index = None
    if isinstance(space, Discrete):
        flat_index = index
    elif isinstance(space, MultiDiscrete):
        flat_index = np.ravel_multi_index(index, space.nvec)

    return flat_index

def num_flat_space_elements(space):
    n = None
    if isinstance(space, Discrete):
        n = space.n
    elif isinstance(space, MultiDiscrete):
        n = np.prod(space.nvec)
    return n