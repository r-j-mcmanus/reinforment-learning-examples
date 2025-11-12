"""Trains on sequences of observations and actions to learn latent dynamics for one step"""

from collections import deque

import torch
import torch.optim as optim

from rssm import RSSM
from state import State
from replay_memory import ReplayMemory



def train_rssm(model: RSSM, replay_memory: ReplayMemory, overshooting_distance: int = 3,
               *, 
               sequence_length: int = 10, batch_size: int = 16):
    """Trains an RSSM on sequences of observations and actions."""
    # Instantiate model
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(100):  # number of epochs
        total_loss = 0

        loss = None

        h = torch.zeros(model._rnn_num_layers, batch_size, model._rnn_hidden_size)
        hs: deque[torch.Tensor] = deque()
        posterior_states: deque[State] = deque()
        prior_states: deque[State] = deque()

        # see `state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = _get_batch(memory)`
        # in optimise.py
        state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = replay_memory.sample_sequential(batch_size, sequence_length)

        # to do vectorise this then sum along sequence dimension
        for t in range(1, sequence_length):
            obs = non_final_next_states[:, t]
            reward = reward_batch[:, t]
            prev_obs = state_batch[:, t]
            action = action_batch[:, t]

            # deterministic, transition and posterior
            h = model.deterministic_step(prev_obs, action, h) # updates hidden vector
            prior_state = model.prior(h) # uses hidden vector 
            posterior_state = model.posterior(obs, h)

            # KL divergence
            beta = min(1.0, epoch / 50) # prevents early collapse of latent space
            #kl_loss = beta * model.divergence_from_states(posterior_state, prior_state)
            kl_loss = beta * model.latent_overshooting_kl(posterior_states, prior_states, action[-overshooting_distance:], hs, overshooting_distance)

            # Reconstruction loss, predicting obs from latent state
            recon_loss = model.observation_reconstruction_error(posterior_state, obs, h)

            # reward loss
            reward_loss = model.reward_reconstruction_error(posterior_state, reward, h)

            # Total loss
            loss = recon_loss + kl_loss + reward_loss
            total_loss += loss.item()

            h = h.detach()
            hs.append(h.detach())
            posterior_states.append(posterior_state.detach()) # Not sure this is a good idea
            prior_states.append(prior_state.detach()) # Not sure this is a good idea

        assert isinstance(loss, torch.Tensor)
        # Back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")