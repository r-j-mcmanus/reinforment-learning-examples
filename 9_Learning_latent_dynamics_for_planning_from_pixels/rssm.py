import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Distribution
from torch import Tensor

from state import State
from transition import Transition
from posterior import Posterior
from decoder import Decoder

class RSSM(nn.Module):
    def __init__(self, state_size: int, obs_size: int, action_size: int, 
                 rnn_hidden_size=32, rnn_num_layers=1,
                 mean_only=False, activation=F.elu, min_stddev=1e-5):
        """
        https://arxiv.org/pdf/1802.03006

        Gaussian State Space Model (SSM) implemented in PyTorch.

        This model defines a latent dynamics system where transitions between states
        are modeled as Gaussian distributions. It includes both a transition model
        (prior) and a posterior model, each parameterized by feedforward neural networks.

        Args:
            state_size (int): Dimensionality of the latent state space.
            obs_size (int): Dimensionality of the observation state space.
            action_size (int): Dimensionality of the latent state space.
            activation (callable): Activation function used in hidden layers (default: ELU).
            min_stddev (float): Minimum standard deviation to ensure numerical stability.

        Attributes:
            transition_fc1 (nn.Linear): First layer of the transition network.
            transition_mean (nn.Linear): Layer to compute the mean of the transition distribution.
            transition_stddev (nn.Linear): Layer to compute the stddev of the transition distribution.
            posterior_fc1 (nn.Linear): First layer of the posterior network.
            posterior_mean (nn.Linear): Layer to compute the mean of the posterior distribution.
            posterior_stddev (nn.Linear): Layer to compute the stddev of the posterior distribution.
            _rnn (nn.RNN): The RNN module for deterministic state transitions.
            _hn (Tensor): The hidden state of the RNN.
        """
        super(RSSM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.obs_size = obs_size
        self.mean_only = mean_only
        self.min_stddev = min_stddev
        self.activation = activation

        self._hidden_size = 32
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_num_layers = rnn_num_layers
        self.decoder_hidden_size = 32

        # deterministic network
        self._rnn = nn.RNN(state_size + action_size, rnn_hidden_size, rnn_num_layers)
        self._hn = None

        # Transition network
        self.transition = Transition(self._rnn_hidden_size, state_size)

        # Posterior network
        post_input_size = self._rnn_hidden_size + self.obs_size
        self.posterior = Posterior(post_input_size, state_size)

        # State Decoder
        decoder_input_size = self._rnn_hidden_size + state_size
        self.observation = Decoder(decoder_input_size, self.obs_size)
        self.reward = Decoder(decoder_input_size, 1)

    def distribution_from_state(self, state: State) -> Distribution:
        """Given a state dict, returns the associated MultivariateNormal distribution."""
        stddev = torch.clamp(state.stddev, min=self.min_stddev)
        return MultivariateNormal(state.mean, torch.diag_embed(stddev))

    def divergence_from_states(self, lhs: State, rhs: State) -> Tensor:
        """Given two states, finds the associated distribtions and returns the KL divergence."""
        lhs_dist = self.distribution_from_state(lhs)
        rhs_dist = self.distribution_from_state(rhs)
        return torch.distributions.kl_divergence(lhs_dist, rhs_dist)

    def deterministic_step(self, prev_state: Tensor, prev_action: Tensor, h0: Tensor | None = None) -> None:
        """
        Forward pass through the deterministic RNN state space model and store the hidden state.

        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})

        Args:
            prev_state (Tensor): Previous state tensor of shape (batch_size, state_size).
            prev_action (Tensor): Previous action tensor of shape (batch_size, action_size).
        """
        inputs = torch.cat([prev_state, prev_action], dim=1).unsqueeze(0)  # (1, batch_size, state_size + action_size)
        batch_size = inputs.size(1)
        if self._hn is None:
            if h0 is None:
                self._hn = torch.zeros(self._rnn_num_layers, batch_size, self._rnn_hidden_size)
            else:
                assert h0.size() == (self._rnn_num_layers, batch_size, self._rnn_hidden_size)
                self._hn = h0
        _, self._hn = self._rnn(inputs, self._hn)

    def reset_hidden(self) -> None:
        """Resets the RNN hidden state."""
        self._hn = None

    def posterior(self, obs: Tensor) -> State:
        """
        Computes the approximate posterior distribution over the current latent state given the previous 
        state, action, and current observation.

        x_t: current observation
        h_t: RNN hidden activation vectors
        z_t: current latent state

        phi_1 = NN parameters for mean of p(alpha|x)
        phi_2 = NN parameters for stddev of p(alpha|x)
        
        p(z_t| h_t, x_t) = p(x_t | z_t, h_t) p(z_t| h_t) / p(x_t) ~ N(mean(phi_1), stddev(phi_2))

        Approximate the Bayesian posterior using a NN that takes the the prior and current observation.
        """
        prior = self.transition(self._hn)
        inputs = torch.cat([prior.mean, prior.stddev, self._hn, obs], dim=1)
        return self.posterior(inputs)
    
    def prior(self) -> State:
        return self.transition(self._hn)
    
    def observation_error(self, posterior_state: State, obs: Tensor):
        inputs = torch.cat([self._hn, posterior_state.sample])
        return nn.functional.mse_loss(self.observation(inputs), obs)
    
    def reward_error(self, posterior_state: State, reward: Tensor):
        inputs = torch.cat([self._hn, posterior_state.sample])
        return nn.functional.mse_loss(self.reward(inputs), reward)
    