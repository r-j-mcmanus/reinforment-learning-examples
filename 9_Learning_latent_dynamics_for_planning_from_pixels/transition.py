import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch import Tensor

from state import State


class Transition(nn.Module):
    def __init__(self, input_size: int, state_size: int,
                 mean_only=False, activation=F.elu, min_stddev=1e-5):
        """
        https://arxiv.org/pdf/1802.03006

        This model defines the transitions between states, where the distributions is
        modelled as a Gaussian distribution. 

        Args:
            input_size (int): The input size for the transition model, e.g the state and 
                action for SSM, the hidden RNN vector in RSSM.
            state_size (int): Dimensionality of the latent state space.
            activation (callable): Activation function used in hidden layers (default: ELU).
            min_stddev (float): Minimum standard deviation to ensure numerical stability.

        Attributes:
            transition_fc1 (nn.Linear): First layer of the transition network.
            transition_mean (nn.Linear): Layer to compute the mean of the transition distribution.
            transition_stddev (nn.Linear): Layer to compute the stddev of the transition distribution.
        """
        super().__init__()
        self.state_size = state_size
        self.mean_only = mean_only
        self.min_stddev = min_stddev
        self.activation = activation

        self._hidden_size = 32
        self.input_size = input_size

        self.transition_fc1 = nn.Linear(self.input_size, self._hidden_size)
        self.transition_fc2 = nn.Linear(self._hidden_size, self._hidden_size)
        self.transition_mean = nn.Linear(self._hidden_size, state_size)
        self.transition_stddev = nn.Linear(self._hidden_size, state_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, input: Tensor) -> State:
        """
        Given the RNN hidden activation vectors, computes the prior distribution over the next state.
        
        h_t: RNN hidden activation vectors

        theta_1 = NN parameters for mean of p(alpha)
        theta_2 = NN parameters for stddev of p(alpha)
        
        p(z_t| h_t) ~ N(mean(theta_1), stddev(theta_2))
        """
        hidden = self.activation(self.transition_fc1(input))
        hidden = self.activation(self.transition_fc2(hidden))
        mean = self.transition_mean(hidden)
        stddev = F.softplus(self.transition_stddev(hidden)) + self.min_stddev
        sample = mean if self.mean_only else MultivariateNormal(mean, torch.diag_embed(stddev)).rsample()
        return State(mean, stddev, sample)
