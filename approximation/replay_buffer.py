# from https://gitlab.inria.fr/creinke/transfer_rl

import exputils as eu
import random
import torch
import numpy as np

class ReplayBuffer:
    '''General purpose replay buffer.'''

    @staticmethod
    def default_config():
        dc = eu.AttrDict(
            capacity=1,
            batch_size=1
        )
        return dc


    def __init__(self, config=None, **argv):
        self.config = eu.combine_dicts(argv, config, self.default_config())

        self.memory = []
        self.position = 0


    def add(self, data):
        """Adds data to the replay buffer."""
        if len(self.memory) < self.config.capacity:
            self.memory.append(None)
        self.memory[self.position] = data
        self.position = (self.position + 1) % self.config.capacity


    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.config.batch_size
        return random.sample(self.memory, batch_size)


    def sample_and_split(self, batch_size=None):
        '''
        Samples data points and if they consist of a list, array, or tuple splits the different parts and returns
        them as individual tensor arrays
        '''

        if batch_size is None:
            batch_size = self.config.batch_size

        data = self.sample(batch_size)
        buf = [*zip(*data)]
        return [list(x) for x in buf]


    def sample_and_split_as_tensors(self, batch_size=None):
        '''
        Samples data points and if they consist of a list, array, or tuple splits the different parts and returns
        them as individual tensor arrays
        '''

        if batch_size is None:
            batch_size = self.config.batch_size

        data = self.sample(batch_size)
        buf = [*zip(*data)]

        out = []
        for x in buf:
            if np.ndim(x[0]) == 0:
                # scalar
                out.append(torch.tensor(x))
            else:
                # array
                out.append(torch.stack(x))

        return out


    def sample_and_split_as_nparray(self, batch_size=None):
        '''
        Samples data points and if they consist of a list, array, or tuple splits the different parts and returns
        them as individual tensor arrays
        '''

        if batch_size is None:
            batch_size = self.config.batch_size

        data = self.sample(batch_size)
        buf = [*zip(*data)]
        return [np.array(x) for x in buf]


    def __len__(self):
        return len(self.memory)