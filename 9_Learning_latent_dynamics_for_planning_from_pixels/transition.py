import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch import Tensor

from state import State
from constants import *


class Transition(nn.Module):
    def __init__(self, *, mean_only=False, activation=F.elu):
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
        self.mean_only = mean_only
        self.LOG_STD_MIN = -26
        self.LOG_STD_MAX = 2
        self.activation = activation

        state_size = Constants.World.latent_state_dimension
        hidden_size = Constants.Common.MLP_width
        input_size = Constants.World.hidden_state_dimension

        self.transition_fc1 = nn.Linear(input_size, hidden_size)
        self.transition_fc2 = nn.Linear(hidden_size, hidden_size)
        self.transition_mean = nn.Linear(hidden_size, state_size)
        self.transition_log_std = nn.Linear(hidden_size, state_size)

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
        x = self.activation(self.transition_fc1(input))
        x = self.activation(self.transition_fc2(x))
        mean: Tensor = self.transition_mean(x)
        log_std: Tensor = self.transition_log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        stddev = log_std.exp()
        sample: Tensor = mean if self.mean_only else torch.distributions.Normal(mean, stddev).rsample()
        return State(mean, stddev, sample)
    