import exputils as eu
import numpy as np
import utils
import torch
from torch import nn
from torch.nn import Parameter


class DND(nn.Module):
    """
    Implementation of a differentiable neural dictionary as introduced in
    Neural Episodic Control (Pritzel et. al, 2017)
    """

    @staticmethod
    def default_config():
        dc = eu.AttrDict(
            dnd_capacity=100,
            num_neighbours=5,
            key_size=8,
            dnd_alpha=0.1,
            device=torch.device('cpu'),
            update_combine_mode='ql', # or max
            key_requires_grad=True, # used to set if the key should be updated or not, particularly useful when using Identity module
        )
        return dc

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        # opposed to paper description, this list is not growing but pre-initialised and gradually replaced
        self.keys = Parameter(torch.ones(self.config.dnd_capacity, self.config.key_size, device=self.config.device) * 1e6, requires_grad=self.config.key_requires_grad) # use large values to allow for low similarity with keys while warming up
        self.values = Parameter(torch.zeros(self.config.dnd_capacity, device=self.config.device))

        self.keys_hash = {} # key to index
        self.last_used = np.linspace(self.config.dnd_capacity, 1, self.config.dnd_capacity, dtype=np.uint32) # used to manage lru replacement
        # initialised in descending order to foster the replacement of earlier indexes before later ones

    def lookup(self, key):
        """
        Single-mode lookup.

        To be used when going through DND without plans to keep gradients

        Params:
            - key: observation embedding to be looked up
        """
        with torch.no_grad():
            sq_distances, neighbour_idxs = utils.knn_search(key, self.keys, self.config.num_neighbours)

            # maintain lru here
            self.last_used += 1 # increment time last used for all keys
            self.last_used[neighbour_idxs.reshape(-1)] = 0 # reset time last used for neighbouring keys

            # compute the Q
            w_i = utils.inverse_distance_kernel(torch.tensor(sq_distances, device=self.config.device))
            w_i /= torch.sum(w_i)
            v_i = self.values[neighbour_idxs]

            return torch.sum(w_i * v_i).item()

    def forward(self, keys):
        """
        Batch mode lookup.

        Gradients are kept track of here in order to be used during gradient based
        optimization of self.keys and self.values involved in computing Q-values

        Params:
            - keys: tensor of observation embeddings to be looked up
        """

        sq_distances, neighbour_idxs = utils.knn_search(keys, self.keys, self.config.num_neighbours)

        neighbour_idxs = neighbour_idxs.reshape(-1) # flattened list view useful below

        # maintain lru here
        self.last_used += 1
        self.last_used[neighbour_idxs] = 0

        # re-compute distances for backprop
        neighbours = self.keys[neighbour_idxs].view(-1, self.config.num_neighbours, self.config.key_size)
        sq_distances = ((keys.unsqueeze(dim = 1) - neighbours) ** 2).sum(dim = 2)
        weights = utils.inverse_distance_kernel(sq_distances)
        weights /= weights.sum(dim = 1, keepdim = True)

        values = self.values[neighbour_idxs].view(-1, self.config.num_neighbours)

        return torch.sum(weights * values, dim = 1)


    def update_batch(self, keys, values):
        """
        Update the DND with keys and values experienced from an episode

        Params:
            - keys: tensor of keys to be inserted in dnd
            - values: tensor of values to be inserted in dnd
        """
        # first handle duplicates inside the batch of data by either taking the max or averaging
        alpha = None if self.config.update_combine_mode == 'max' else self.config.dnd_alpha
        # if len(values) > 1:
        #     print('in dnd before merge values', flush=True)
        #     print(values.dtype, flush=True)
        #     print(values, flush=True)
        keys, values = utils.combine_by_key(keys, values, op=self.config.update_combine_mode, alpha=alpha) # returns keys as a list of tuples that can be used to index self.keys_hash
        match_idxs, match_dnd_idxs, new_idxs = [], [], []
        

        # limit to make sure keys and values are not larger than capacity
        keys, values = keys[-self.config.dnd_capacity:], values[-self.config.dnd_capacity:]
        # if len(values) > 1:
        #     print('in dnd after merge values', flush=True)
        #     print(values.dtype, flush=True)
        #     print(values, flush=True)

        # then group indices of exact matches for existing keys and new keys
        for i, key in enumerate(keys):
            if key in self.keys_hash:
                match_dnd_idxs.append(self.keys_hash[key])
                match_idxs.append(i)
            else:
                new_idxs.append(i)

        num_matches, num_new = len(match_idxs), len(new_idxs)

        self.last_used += 1 # maintain time since keys used

        with torch.no_grad():
            # make tensors for fancy indexing and easy interoperability with self.keys and self.values
            keys, values = torch.tensor(keys, device=self.config.device), torch.tensor(values, device=self.config.device)

            # update exact matches using dnd learning rate
            if num_matches:
                self.values[match_dnd_idxs] += self.config.dnd_alpha * (values[match_idxs] - self.values[match_dnd_idxs])
                self.last_used[match_dnd_idxs] = 0

            # replace least recently used keys with new keys
            if num_new:
                lru_idxs = np.argsort(self.last_used)[-num_new:] # get lru indices using the self.last_used
                self.keys[lru_idxs] = keys[new_idxs]
                # if len(values) > 1:
                #     print('in replace least recently used', flush=True)
                #     print(values.dtype, flush=True)
                #     print(values, flush=True)
                #     print('self.values', flush=True)
                #     print(self.values.dtype, flush=True)
                self.values[lru_idxs] = values[new_idxs]
                self.last_used[lru_idxs] = 0

                # update self.keys_hash
                inv_hash = {v: k for k, v in self.keys_hash.items()}

                for idx in lru_idxs:
                    if idx in inv_hash:
                        del self.keys_hash[inv_hash[idx]]
                    self.keys_hash[tuple(self.keys[idx].detach().cpu().numpy())] = idx
