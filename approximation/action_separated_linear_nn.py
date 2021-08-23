# from https://gitlab.inria.fr/creinke/transfer_rl

import exputils as eu
import torch

class ActionSeparatedLinearNN(torch.nn.Module):
    """
    Deep network model consisting of linear layers. Has separate layers/modules for each action.

    Configuration:
        - hidden_layers: Number and dimension of hidden layers. List with number of neurons per hidden loyer.
                         No hidden lauers if None. (default=None)

        - activation_func: Activation function of neurons in hidden layers. (default=None)
                           Please note, the output layer has no activation function.

        - is_bias: Should bias weights be used. (default=True)

        - weight_init_func: Initialization function of weights (not biases!).
                            Configuration follows the exputils function configuration definition.
                            Example:
                                 weight_init_std = dict(
                                     func=torch.nn.init.normal_,     # handle to the function
                                     mean=0.0,                       # function parameters
                                     std=1.0
                                 )
                            The function must take as first argument the weight parameters from a linear layer.
                            If None, then the pytorch standard random initialization of linear layers is used.
                            (default=None)

    Properties:
        - dtype: Get torch data type if the input and output tensors which is torch.float32.
    """

    @staticmethod
    def default_config():
        dc = eu.AttrDict(
            hidden_layers=None,
            activation_func=None,
            is_bias=True,
            weight_init_func=None,
        )
        return dc


    @property
    def dtype(self):
        # torch.nn.Linear uses torch.float32 as dtype for parameters
        return torch.float32


    def __init__(self, state_dim, n_actions, n_out=1, config=None, **argv):
        super().__init__()
        self.config = eu.combine_dicts(argv, config, self.default_config())

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_out = n_out

        self.layers_per_action = torch.nn.ModuleList()

        # create for each action separate layers
        for action_idx in range(n_actions):

            # create list of layers
            layers = []

            # no hidden layer of nothing is defined
            if self.config.hidden_layers is None or len(self.config.hidden_layers) == 0:
                layers.append(self._create_linear_layer(state_dim, self.n_out))
            else:
                next_n_in = state_dim
                for n_hidden_neurons in self.config.hidden_layers:
                    layers.append(self._create_linear_layer(next_n_in, n_hidden_neurons))
                    if self.config.activation_func is not None:
                        layers.append(self.config.activation_func())
                    next_n_in = n_hidden_neurons

                # output
                layers.append(self._create_linear_layer(next_n_in, self.n_out))

            self.layers_per_action.append(torch.nn.Sequential(*layers))


    def _create_linear_layer(self, n_in, n_out):

        linear_layer = torch.nn.Linear(n_in, n_out, bias=self.config.is_bias)

        if self.config.weight_init_func is not None:
            eu.misc.call_function_from_config(self.config.weight_init_func, linear_layer.weight)

        return linear_layer


    def forward(self, state, actions=None):

        # outputs are for [state_idx, action_idx, feature_idx]
        output = torch.empty([state.shape[0], self.n_actions, self.n_out])
        for action_idx, action_layers in enumerate(self.layers_per_action):
            output[:, action_idx, :] = action_layers(state)

        # if actions are specified for each state, then only return the feature_comb probabilities for these actions
        if actions is not None:

            # TODO: check in DQN code if this can be doen with a mask instead of a loop
            buf = torch.empty([output.shape[0], output.shape[2]])
            for state_idx in range(output.shape[0]):
                buf[state_idx, :] = output[state_idx, actions[state_idx], :]

            output = buf

        return output