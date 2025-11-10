"""Trains on sequences of observations and actions to learn latent dynamics for one step"""

# setup
import torch
import torch.nn as nn
import torch.optim as optim

from rssm import RSSM
from state import State

# Example dimensions
state_size = 32
embed_size = 64
obs_size = 48
action_size = 8
sequence_length = 10
batch_size = 16

# Instantiate model
model = RSSM(state_size, obs_size, action_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Random dummy data for demonstration
observations = torch.randn(batch_size, sequence_length, obs_size)
actions = torch.randn(batch_size, sequence_length, action_size)


for epoch in range(100):  # number of epochs
    total_loss = 0

    prev_state = State(
        mean=torch.zeros(batch_size, state_size),
        stddev=torch.ones(batch_size, state_size),
        sample=torch.zeros(batch_size, state_size)
    )

    for t in range(1, sequence_length):
        obs = observations[:, t]
        prev_obs = observations[:, t - 1]
        action = actions[:, t - 1]

        # deterministic, transition and posterior
        model.deterministic_step(prev_obs, action)
        prior_state = model.transition()
        posterior_state = model.posterior(obs)

        # KL divergence
        kl_loss = model.divergence_from_states(posterior_state, prior_state).mean()

        # Reconstruction loss (example: predicting obs from latent)
        # You would need a decoder here; for now, assume identity
        # recon_loss = nn.functional.huber_loss(decode(posterior_state.sample), obs)
        recon_loss = 0

        # reward loss
        # reward_loss = nn.functional.huber_loss(predicted_action, target_action)
        reward_loss = 0

        # Total loss
        loss = recon_loss + kl_loss + reward_loss
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update state
        prev_state = posterior_state
        prev_state.sample = prev_state.sample.detach()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")