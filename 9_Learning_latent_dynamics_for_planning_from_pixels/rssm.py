import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Distribution
from torch import Tensor
import torch.optim as optim

from state import State
from transition import Transition
from posterior import Posterior
from decoder import Decoder

from collections import deque 
from constants import *

class RSSM(nn.Module):
    def __init__(self, obs_size: int, action_size: int, *, mean_only=False, activation=F.elu):
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
        state_size = Constants.World.latent_state_dimension

        self.obs_size = obs_size
        self.mean_only = mean_only
        self.activation = activation

        hidden_state_dimension = Constants.World.hidden_state_dimension

        # deterministic network
        # 1811.04551 uses rnn, but later dreamer models (2010.02193) use GRU RNNs (1406.1078)
        # Outline:
        #   RNN
        #       h = torch.tanh(W_x(x) + W_h(h))
        #   GRU
        #       z = sigmoid(W_z(x) + U_z(h))  -- how much of the old memory to keep
        #       r = sigmoid(W_r(x) + U_r(h))  -- how much past information to forget when computing new content
        #       h_candidate = torch.tanh(W_xh(x) + U_hh(r * h))
        #       h = (1 - z) * h_candidate + z * h  -- mix old with new directly
        # we use a single layer determanistic model and allow for complications in the state model.
        # TODO test more layers?
        self._rnn = nn.GRU(state_size + action_size, 
                           hidden_state_dimension, 
                           device=DEVICE) 
        
        # Transition network
        self.transition = Transition()

        # Posterior network
        self.posterior_model = Posterior(self.obs_size)

        # State Decoder
        decoder_input_size = hidden_state_dimension + state_size
        self.observation = Decoder(decoder_input_size, self.obs_size)
        self.reward = Decoder(decoder_input_size, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=1e-3)


    def rollout(self, initial_state: State, initial_hidden_state: Tensor, action: Tensor) -> list[State]:
        """predicts the next state from both the deterministic and stochastic space model given the 
        initial observation and actions up to the learning horizon_length. 
        
        Returns
        -------
        stochastic_steps: list[Tensor]
        """
        deterministic_steps = []
        stochastic_steps = []

        horizon_length = Constants.Behaviors.imagination_horizon

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
        stddev = torch.clamp(state.stddev, min=Constants.Common.min_stddev)
        return MultivariateNormal(state.mean, torch.diag_embed(stddev))

    def divergence_from_states(self, post: State, prior: State) -> Tensor:
        """Given two states, finds the associated distribtions and returns the KL divergence. 
        Uses KL balancing as learning the transition function is difficult and we want to 
        avoid regularizing the representations toward a poorly trained prior"""
        post_dist = self.distribution_from_state(post)
        sg_post_dist = self.distribution_from_state(post.detach())
        prior_dist = self.distribution_from_state(prior)
        sg_prior_dist = self.distribution_from_state(prior.detach())
        a = Constants.World.kl_balancing
        kl_loss = (
            torch.distributions.kl_divergence(sg_post_dist, prior_dist).mean() * a +
            torch.distributions.kl_divergence(post_dist, sg_prior_dist).mean() * (1-a)
        )
        return kl_loss

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
        inputs = torch.cat([prev_state, prev_action], dim=-1)  # (batch_size, state_size + action_size)
        _, h = self._rnn(inputs.unsqueeze(dim=0), h.unsqueeze(dim=0)) 
        return h.squeeze(0)

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
        # we can remove the transition from this, probably causing a longer training time?
        # this is as p = f(h) and q = f_2(h, p, o) = f_2(h, f(h), o)
        # so if q = f_3(h, o) it can match f_2(h, f(h), o)
        # would probably make for a more robust method as p could change distribution and you wouldnt need to change q
        prior = self.transition.forward(h)
        inputs = torch.cat([prior.mean, prior.stddev, h, obs], dim=-1)
        return self.posterior_model(inputs)
    
    def prior(self, h: Tensor) -> State:
        return self.transition(h)

    def observation_reconstruction_error(self, posterior_state: State, obs: Tensor, h: Tensor):
        """
        Taking a sample of the posterior states for MC intergration of the expectation over the posterior and
        using the observation model to move from the state space to observation space. I then find the mse 
        loss for my reconstruction vs the real observation
        """
        return nn.functional.mse_loss(self.observation(h, posterior_state.mean), obs).mean() # use mse not log likelihood as the observation model is assumed to be Gaussian

    def reward_reconstruction_error(self, posterior_state: State, reward: Tensor, h: Tensor):
        """
        Taking a sample of the posterior states for MC intergration of the expectation over the posterior and
        using the observation model to move from the state space to observation space. I then find the mse 
        loss for my reconstruction vs the real observation
        """
        return nn.functional.mse_loss(self.reward(h, posterior_state.mean), reward).mean() # use mse not log likelihood as the observation model is assumed to be Gaussian
