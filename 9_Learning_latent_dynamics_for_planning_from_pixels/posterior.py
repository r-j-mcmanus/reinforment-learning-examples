import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch import Tensor

from state import State
from constants import *

class Posterior(nn.Module):
    """
        Representation model that computes the approximate posterior distribution over the current latent state given the previous 
        state, action, and current observation.

        x_t: current observation
        h_t: RNN hidden activation vectors
        z_t: current latent state

        phi_1 = NN parameters for mean of p(alpha|x)
        phi_2 = NN parameters for stddev of p(alpha|x)
        
        p(z_t| h_t, x_t) = p(x_t | z_t, h_t) p(z_t| h_t) / p(x_t) ~ N(mean(phi_1), stddev(phi_2))

        Approximate the Bayesian posterior using a NN that takes the the prior and current observation.
    """

    def __init__(self, obs_size: int, *,
                 mean_only=False, activation=F.elu, min_stddev=1e-5):
        """
        https://arxiv.org/pdf/1802.03006

        Gaussian State Space Model (SSM) implemented in PyTorch.

        This model defines the posterior for transitions between states given observations,
        modelled as Gaussian distributions, parameterized by feedforward neural networks.

        Args:
            input_size (int): e.g. obs_size for SSM, or rnn_hidden_size + obs_size for RSSM
            state_size (int): Dimensionality of the latent state space.
            activation (callable): Activation function used in hidden layers (default: ELU).
            min_stddev (float): Minimum standard deviation to ensure numerical stability.

        Attributes:
            posterior_fc1 (nn.Linear): First layer of the posterior network.
            posterior_mean (nn.Linear): Layer to compute the mean of the posterior distribution.
            posterior_stddev (nn.Linear): Layer to compute the stddev of the posterior distribution.
        """
        super().__init__()
        stoch_state_dim = Constants.World.latent_state_dimension
        hidden_state_dimension = Constants.World.hidden_state_dimension
        input_size = hidden_state_dimension + obs_size
        self.mean_only = mean_only
        self.min_stddev = min_stddev
        self.activation = activation

        hidden_size = Constants.Common.MLP_width

        # Posterior network
        self.posterior_fc1 = nn.Linear(input_size, hidden_size)
        self.posterior_fc2 = nn.Linear(hidden_size, hidden_size)
        self.posterior_mean = nn.Linear(hidden_size, stoch_state_dim)
        self.posterior_stddev = nn.Linear(hidden_size, stoch_state_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: Tensor, h: Tensor) -> State:
        """
        Computes the approximate posterior distribution over the current latent state given the previous 
        state, action, and current observation.

        Example input for for RSSM (see fig2c where the posterior is conditioned on h_t as well as the 
        latent space predicted by the prior)

        x_t: current observation
        h_t: RNN hidden activation vectors
        z_t: current latent state

        phi_1 = NN parameters for mean of p(alpha|x)
        phi_2 = NN parameters for stddev of p(alpha|x)
        
        p(z_t| h_t, x_t) = p(x_t | z_t, h_t) p(z_t| h_t) / p(x_t) ~ N(mean(phi_1), stddev(phi_2))

        Approximate the Basian posterior using a NN that takes the the prior and current observation.
        """
        hidden = self.activation(self.posterior_fc1(torch.concat([obs, h], dim=-1)))
        hidden = self.activation(self.posterior_fc2(hidden))
        mean = self.posterior_mean(hidden)
        stddev = F.softplus(self.posterior_stddev(hidden)) + self.min_stddev
        sample = mean if self.mean_only else MultivariateNormal(mean, torch.diag_embed(stddev)).rsample()
        return State(mean, stddev, sample)
    