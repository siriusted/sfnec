import exputils as eu
import numpy as np
import utils
import torch
from torch import nn
from torch.nn import Parameter


class DND_SF(nn.Module):
    """
    Implementation of a differentiable neural dictionary as introduced in
    Neural Episodic Control (Pritzel et. al, 2017)

    Here the values are the successor features (SF)
    """

    @staticmethod
    def default_config():
        dc = eu.AttrDict(
            dnd_capacity=100,
            num_neighbours=5,
            key_size=8,
            value_size=8, # phi_dim
            dnd_alpha=0.1,
            device=torch.device('cpu'),
            psi_init_std=0.01,
            psi_init_mode='zeros', # or 'random'
            update_combine_mode='ql', # or max
            key_requires_grad=True, # used to set if the key should be updated or not, particularly useful when using Identity module
        )
        return dc

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.keys = Parameter(torch.ones(self.config.dnd_capacity, self.config.key_size, device=self.config.device) * 1e6, requires_grad=self.config.key_requires_grad) # use large values to allow for low similarity with keys while warming up
        
        if self.config.psi_init_mode == 'zeros':
            self.values = Parameter(torch.zeros(self.config.dnd_capacity, self.config.value_size, device=self.config.device))
        else:
            self.values = Parameter(torch.randn(self.config.dnd_capacity, self.config.value_size, device=self.config.device) * self.config.psi_init_std) 
        
        self.keys_hash = {} # key to index
        self.last_used = np.linspace(self.config.dnd_capacity, 1, self.config.dnd_capacity, dtype=np.uint32) # used to manage lru replacement
        # initialised in descending order to foster the replacement of earlier indexes before later ones

    def _lookup(self, keys, verbose=False):
        if verbose:
            print(f'Shape of keys: {keys.shape}')
        sq_distances, neighbour_idxs = utils.knn_search(keys, self.keys, self.config.num_neighbours)

        if verbose:
            print(f'Shape of sq_dists: {sq_distances.shape}')
            print(f'Shape of neighbour_idxs: {neighbour_idxs.shape}')

        neighbour_idxs = neighbour_idxs.reshape(-1) # flattened list view useful below

        if verbose:
            print(f'Shape of neighbour_idxs after flatten: {neighbour_idxs.shape}')

        # maintain lru here
        self.last_used += 1
        self.last_used[neighbour_idxs] = 0

        # re-compute distances for backprop
        neighbours = self.keys[neighbour_idxs].view(-1, self.config.num_neighbours, self.config.key_size)
        if verbose:
            print(f'Shape of neighbours for recomputing sq dists: {neighbours.shape}')
        sq_distances = ((keys.unsqueeze(dim = 1) - neighbours) ** 2).sum(dim = 2)
        weights = utils.inverse_distance_kernel(sq_distances)
        weights /= weights.sum(dim = 1, keepdim = True)

        weights = weights.reshape(-1, self.config.num_neighbours, 1)
        values = self.values[neighbour_idxs].view(-1, self.config.num_neighbours, self.config.value_size)

        return torch.sum(weights * values, dim = 1)

    def forward(self, keys, training=False):
        """
        Batch mode lookup.

        If training is set to True, Gradients are kept track of here in order to be used during gradient based
        optimization of self.keys and self.values involved in computing SF-values

        Params:
            - keys: tensor of observation embeddings to be looked up
        """

        if training:
            with torch.no_grad():
                return self._lookup(keys)[0]

        return self._lookup(keys)


    def update_batch(self, keys, values):
        """
        Update the DND with keys and values experienced from an episode

        Params:
            - keys: tensor of keys to be inserted in dnd
            - values: tensor of values to be inserted in dnd
        """
        # first handle duplicates inside the batch of data by either taking the max or averaging
        alpha = None if self.config.update_combine_mode == 'max' else self.config.dnd_alpha
        keys, values = utils.combine_by_key(keys, values, op=self.config.update_combine_mode, alpha=alpha) # returns keys as a list of tuples that can be used to index self.keys_hash
        match_idxs, match_dnd_idxs, new_idxs = [], [], []

        # limit to make sure keys and values are not larger than capacity
        keys, values = keys[-self.config.dnd_capacity:], values[-self.config.dnd_capacity:]

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
                self.values[lru_idxs] = values[new_idxs]
                self.last_used[lru_idxs] = 0

                # update self.keys_hash
                inv_hash = {v: k for k, v in self.keys_hash.items()}

                for idx in lru_idxs:
                    if idx in inv_hash:
                        del self.keys_hash[inv_hash[idx]]
                    self.keys_hash[tuple(self.keys[idx].detach().cpu().numpy())] = idx


if __name__ == "__main__":
    dnd = DND_SF(
        dict(
            dnd_capacity=4,
            num_neighbours=2,
            key_size=2,
            value_size=2
        )
    )

    dnd.keys = Parameter(torch.tensor([[5., 5.], [7., 7.], [10., 10.], [8., 8.]]))
    dnd.values = Parameter(torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]))

    query1 = torch.tensor([4., 4.])
    query2 = torch.tensor([5., 5.])

    queries = torch.tensor([[4., 4.], [5., 5.]])

    print(dnd.forward(query1.unsqueeze(0), training=True))
    print(dnd.forward(query2.unsqueeze(0), training=True))
    print(dnd.forward(queries))


