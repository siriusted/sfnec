import exputils as eu
import torch
from torch import nn
from approximation import DND

class NEC(nn.Module):
    """
    Implementation of neural episodic control architecture as introduced in
    Neural Episodic Control (Pritzel et. al, 2017)
    """

    def __init__(self, embedding_net, n_actions, config=None,  **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config)
        self.embedding_net = embedding_net
        self.dnds = nn.ModuleList([DND(config) for _ in range(n_actions)])

    def forward(self, observations):
        """
        Batch mode forward pass through entire NEC network
        """
        keys = self.embedding_net(observations)
        qs = torch.stack([dnd(keys) for dnd in self.dnds]).T # to get q_values in shape [batch_size x actions]

        return qs

    def lookup(self, obs):
        """
        Single mode forward pass through entire NEC network without gradient tracking
        """
        with torch.no_grad():
            key = self.embedding_net(obs.unsqueeze(0))
            qs = [dnd.lookup(key) for dnd in self.dnds]

            return qs, key.squeeze(0) # remove batch dimension

    def update_memory(self, action, keys, values):
        """
        Used to batch update an action's DND
        """
        self.dnds[action].update_batch(keys, values)