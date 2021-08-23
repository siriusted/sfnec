import numpy as np
import torch
from torch import nn
from approximation import DND_SF

class SFNEC(nn.Module):
    """
    Implementation of neural episodic control architecture as introduced in
    Neural Episodic Control (Pritzel et. al, 2017)

    Modified architecture has a DND that stores Successor Features as its values
    """

    def __init__(self, embedding_net, n_actions, config=None):
        super().__init__()
        self.embedding_net = embedding_net
        self.dnds = nn.ModuleList([DND_SF(config) for _ in range(n_actions)])

    def forward(self, observations):
        """
        Batch mode forward pass through entire NEC network
        """
        keys = self.embedding_net(observations)
        psis = torch.stack([dnd(keys) for dnd in self.dnds])
        # to get psi_values in shape [batch_size x actions x SF_shape] SF_shape := value_size
        psis = torch.transpose(psis, 0, 1)
        return psis

    def lookup(self, obs):
        """
        Single mode forward pass through entire NEC network without gradient tracking
        """
        with torch.no_grad():
            key = self.embedding_net(obs.unsqueeze(0))
            psis = np.array([dnd(key, training=True).detach().cpu().numpy() for dnd in self.dnds])

            return psis, key.squeeze(0) # remove batch dimension;

    def update_memory(self, action, keys, values):
        """
        Used to batch update an action's DND
        """
        self.dnds[action].update_batch(keys, values)