import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Distribution
from torch import Tensor

from state import State
from transition import Transition
from posterior import Posterior
from decoder import Decoder

from collections import deque 

class RSSM(nn.Module):
    def __init__(self, state_size: int, obs_size: int, action_size: int, 
                 rnn_hidden_size=32, rnn_num_layers=1,
                 mean_only=False, activation=F.elu, min_stddev=1e-5):
        """
        https://arxiv.org/pdf/1802.03006
        https://arxiv.org/pdf/1811.04551

        Gaussian State Space Model (SSM) implemented in PyTorch.

        This model defines a latent dynamics system where transitions between states
        are modeled as Gaussian distributions. It includes both a transition model
        (prior) and a posterior model, each parameterized by feedforward neural networks.

        Importantly, the model is stateless. The deterministic state vector must be passed
        in the relevant functions.

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

        # Transition network
        self.transition = Transition(self._rnn_hidden_size, state_size)

        # Posterior network
        post_input_size = self._rnn_hidden_size + self.obs_size
        self.posterior_model = Posterior(post_input_size, state_size)

        # State Decoder
        decoder_input_size = self._rnn_hidden_size + state_size
        self.observation = Decoder(decoder_input_size, self.obs_size)
        self.reward = Decoder(decoder_input_size, 1)
    
    def rollout(self, initial_state: State, initial_hidden_state: Tensor, action: Tensor, horizon_length: int = 3) -> list[State]:
        """predicts the next state from both the deterministic and stochastic space model given the 
        initial observation and actions up to the learning horizon_length. 
        
        Returns
        -------
        stoch_steps: list[Tensor]
        """
        deterministic_steps = []
        stochastic_steps = []

        h = initial_hidden_state
        s = initial_state # predict what the state should be given the observation and the hidden state

        sequence_len = action.shape[1]    
        for i in range(min(horizon_length, sequence_len)):
            h = self.deterministic_step(s.sample.squeeze(), action[:,i], h)
            s = self.transition(h) #  stochastic step
            deterministic_steps.append(h)
            stochastic_steps.append(s)
        return stochastic_steps

    def distribution_from_state(self, state: State) -> Distribution:
        """Given a state dict, returns the associated MultivariateNormal distribution."""
        stddev = torch.clamp(state.stddev, min=self.min_stddev)
        return MultivariateNormal(state.mean, torch.diag_embed(stddev))

    def divergence_from_states(self, post: State, prior: State) -> Tensor:
        """Given two states, finds the associated distribtions and returns the KL divergence."""
        lhs_dist = self.distribution_from_state(post)
        rhs_dist = self.distribution_from_state(prior)
        return torch.distributions.kl_divergence(lhs_dist, rhs_dist).mean()

    def latent_overshooting_kl(self, posteriors: deque[State], priors: deque[State], actions, hs: deque[Tensor], d_max: int):
        """
        posteriors: list of posterior states q(s_t)
        proprs: list of prior states p(s_t|s_{t-1}, a_{t-1})
        actions: list of actions a_t
        hs: list of deterministic hidden state h_t
        d_max: maximum overshooting distance
        """
        kl_total = 0.0
        count = 0

        for d in range(1, d_max + 1):
            for t in range(d, len(posteriors)):
                # Given the inital deterministic hidden state h_t, the prior at time t and the d actions subsequent actions,
                # predict the priot at time t+d
                pred_prior = self.rollout_prior(priors[t], actions[t:t+d], hs[t])

                # Compute KL divergence between posterior and computed prior
                kl = self.divergence_from_states(posteriors[t+d], pred_prior)
                kl_total += kl
                count += 1

        return kl_total / count

    def rollout_prior(self, start_state: State, actions: list[Tensor], h: Tensor) -> State:
        for action in actions:
            h = self.deterministic_step(start_state.sample, action, h)
            start_state = self.transition(h)
        return start_state

    def deterministic_step(self, prev_state: Tensor, prev_action: Tensor, h: Tensor ) -> Tensor:
        """
        Forward pass through the deterministic RNN state space model and store the hidden state.

        h_t = f(h_{t-1}, z_{t-1}, a_{t-1})

        Args:
            prev_state (Tensor): Previous state tensor of shape (batch_size, state_size).
            prev_action (Tensor): Previous action tensor of shape (batch_size, action_size).
            h (Tensor): deterministic hidden state h_t.

        Returns:
            Tensor: next deterministic hidden state (batch_size, rnn_hidden_size).
        """
        inputs = torch.cat([prev_state.squeeze(), prev_action], dim=1).unsqueeze(0)  # (1, batch_size, state_size + action_size)
        _, h = self._rnn(inputs, h)
        assert isinstance(h, torch.Tensor)
        return h

    def posterior(self, obs: Tensor, h: Tensor) -> State:
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
        assert isinstance(h, torch.Tensor)
        prior = self.transition.forward(h)
        inputs = torch.cat([prior.mean, prior.stddev, h, obs.unsqueeze(0)], dim=2)
        return self.posterior_model(inputs)
    
    def prior(self, h: Tensor) -> State:
        return self.transition(h)
    
    def observation_reconstruction_error(self, posterior_state: State, obs: Tensor, h: Tensor):
        """
        Taking a sample of the posterior states for MC intergration of the expectation over the posterior and
        using the observation model to move from the state space to observation space. I then find the mse 
        loss for my reconstruction vs the real observation
        """
        assert isinstance(h, torch.Tensor)
        inputs = torch.cat([h, posterior_state.sample], dim=2)
        return nn.functional.mse_loss(self.observation(inputs), obs).mean() # use mse not log likelihood as the observation model is assumed to be Gaussian
    
    def reward_reconstruction_error(self, posterior_state: State, reward: Tensor, h: Tensor):
        """
        Taking a sample of the posterior states for MC intergration of the expectation over the posterior and
        using the observation model to move from the state space to observation space. I then find the mse 
        loss for my reconstruction vs the real observation
        """
        inputs = torch.cat([h, posterior_state.sample], dim=2)
        return nn.functional.mse_loss(self.reward(inputs), reward).mean() # use mse not log likelihood as the observation model is assumed to be Gaussian
    