import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Distribution
from torch import Tensor

from state import State


class SSM(nn.Module):
    def __init__(self, state_size: int, obs_size: int, action_size: int, mean_only=False, activation=F.elu, min_stddev=1e-5):
        """
        https://arxiv.org/pdf/1802.03006

        Gaussian State Space Model (SSM) implemented in PyTorch.

        This model defines a latent dynamics system where transitions between states
        are modeled as Gaussian distributions. It includes both a transition model
        (prior) and a posterior model, each parameterized by feedforward neural networks.

        Args:
            state_size (int): Dimensionality of the latent state space.
            embed_size (int): Size of the hidden layers used in the transition and posterior networks.
            mean_only (bool): If True, sampling is deterministic using the mean only.
            activation (callable): Activation function used in hidden layers (default: ELU).
            min_stddev (float): Minimum standard deviation to ensure numerical stability.

        Attributes:
            transition_fc1 (nn.Linear): First layer of the transition network.
            transition_mean (nn.Linear): Layer to compute the mean of the transition distribution.
            transition_stddev (nn.Linear): Layer to compute the stddev of the transition distribution.
            posterior_fc1 (nn.Linear): First layer of the posterior network.
            posterior_mean (nn.Linear): Layer to compute the mean of the posterior distribution.
            posterior_stddev (nn.Linear): Layer to compute the stddev of the posterior distribution.
        """

        super(SSM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.obs_size = obs_size
        self.mean_only = mean_only
        self.min_stddev = min_stddev
        self.activation = activation

        self._hidden_size = 32

        # Transition network
        self.transition_fc1 = nn.Linear(state_size + action_size, self._hidden_size)
        self.transition_mean = nn.Linear(self._hidden_size, state_size)
        self.transition_stddev = nn.Linear(self._hidden_size, state_size)

        # Posterior network
        self.posterior_fc1 = nn.Linear(2 * state_size + obs_size, self._hidden_size)
        self.posterior_mean = nn.Linear(self._hidden_size, state_size)
        self.posterior_stddev = nn.Linear(self._hidden_size, state_size)

    def distribution_from_state(self, state: State) -> Distribution:
        """Given a state dict, returns the associated MultivariateNormal distribution."""
        stddev = torch.clamp(state.stddev, min=self.min_stddev)
        return MultivariateNormal(state.mean, torch.diag_embed(stddev))

    def divergence_from_states(self, lhs: State, rhs: State) -> Tensor:
        """Given two states, finds the associated distribtions and returns the KL divergence."""
        lhs_dist = self.distribution_from_state(lhs)
        rhs_dist = self.distribution_from_state(rhs)
        return torch.distributions.kl_divergence(lhs_dist, rhs_dist)

    def transition(self, prev_state: State, prev_action: Tensor) -> State:
        """
        Given the previous state and action, computes the prior distribution over the next state.
        
        z_{t-1}: previous latent state (sample)
        a_{t-1}: previous action

        theta_1 = NN parameters for mean of p(alpha)
        theta_2 = NN parameters for stddev of p(alpha)
        
        p(z_t| z_{t-1}, a_{t-1}) ~ N(mean(theta_1), stddev(theta_2))
        """
        inputs = torch.cat([prev_state.sample, prev_action], dim=1)
        
        assert len(inputs[0]) == self.state_size + self.action_size, \
            f"Expected input size {self.state_size + self.action_size}, got {len(inputs[0])}"

        hidden = self.activation(self.transition_fc1(inputs))
        mean = self.transition_mean(hidden)
        stddev = F.softplus(self.transition_stddev(hidden)) + self.min_stddev
        sample = mean if self.mean_only else MultivariateNormal(mean, torch.diag_embed(stddev)).rsample()
        return State(mean, stddev, sample)

    def posterior(self, prev_state: State, prev_action: Tensor, obs: Tensor) -> State:
        """
        Computes the approximate posterior distribution over the current latent state given the previous 
        state, action, and current observation.

        z_{t-1}: previous latent state
        a_{t-1}: previous action
        x_t: current observation

        phi_1 = NN parameters for mean of p(alpha|x)
        phi_2 = NN parameters for stddev of p(alpha|x)
        
        p(z_t| z_{t-1}, a_{t-1}, x_t) = p(x_t | z_t, z_{t-1}, a_{t-1}) p(z_t| z_{t-1}, a_{t-1}) / p(x_t) ~ N(mean(phi_1), stddev(phi_2))

        Approximate the Basian posterior using a NN that takes the the prior and current observation.
        """
        prior = self.transition(prev_state, prev_action)
        inputs = torch.cat([prior.mean, prior.stddev, obs], dim=1)
        hidden = self.activation(self.posterior_fc1(inputs))
        mean = self.posterior_mean(hidden)
        stddev = F.softplus(self.posterior_stddev(hidden)) + self.min_stddev
        sample = mean if self.mean_only else MultivariateNormal(mean, torch.diag_embed(stddev)).rsample()
        return State(mean, stddev, sample)