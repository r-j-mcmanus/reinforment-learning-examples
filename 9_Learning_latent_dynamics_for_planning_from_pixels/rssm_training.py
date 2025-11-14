"""Trains on sequences of observations and actions to learn latent dynamics for one step"""

from collections import deque

import torch
import torch.optim as optim

from rssm import RSSM
from state import State
from replay_memory import ReplayMemory


def train_rssm(rssm: RSSM, replay_memory: ReplayMemory, horizon_length: int = 3,
               *, 
               sequence_length: int = 5, batch_size: int = 3, epoch_count: int = 100) -> RSSM:
    """Trains an RSSM on sequences of observations and actions."""
    # Instantiate model
    optimizer = optim.AdamW(rssm.parameters(), lr=1e-3)

    for epoch in range(epoch_count):  # number of epochs
        loss = 0

        sequence_transitions = replay_memory.sample_sequential(batch_size, sequence_length)
        h_1: torch.Tensor = torch.zeros(rssm._rnn_num_layers, batch_size, rssm._rnn_hidden_size)
        h_t = h_1 # relabeled for ease in for loop

        past_rollouts = deque([], maxlen=horizon_length)

        for t in range(0, sequence_length):
            # transition shape (batch_size, sequence_length, _)
            obs = sequence_transitions.state[:,t] # used in reconstruction and posterior
            action = sequence_transitions.action[:,t] # use to find the next deterministic step
            reward = sequence_transitions.reward[:,t] # used in reward reconstruction loss
            next_actions = sequence_transitions.action[:, t:t+horizon_length] # used in rollout

            posterior_state = rssm.posterior(obs, h_t) # predict what the state should be given the observation and the hidden state
            transition_state: State = rssm.transition(h_t) # predict what the state should be given the hidden state

            # getting the diagonal lines in fig 3 c of 1811.04551, does not take the left most node
            s_it = rssm.rollout(posterior_state, h_t, next_actions)
            # this finds s_{i|j} where:
            #  i in [sequence_start, sequence_length]
            #  j = sequence_start is the last observation the state is conditioned on
            past_rollouts.append(s_it)

            kl_loss = 0
            beta = min(1.0, (epoch+1) / 50) # prevents early collapse of latent space
            for i, state_rollout in enumerate(past_rollouts):
                # take the ith so we test against the ith step of the rollout
                if i < len(state_rollout): # the end of the sequence will not have the full horizon
                    kl_loss += beta * rssm.divergence_from_states(transition_state, state_rollout[i]) # KL divergence
            
            # Reconstruction loss, predicting next_obs from latent state
            reconstruction_loss = rssm.observation_reconstruction_error(posterior_state, obs, h_t)

            # reward loss
            reward_loss = rssm.reward_reconstruction_error(posterior_state, reward, h_t)

            loss += reconstruction_loss + reward_loss + kl_loss
            
            # h for the next loop
            h_t = rssm.deterministic_step(posterior_state.sample, action, h_t)

        assert isinstance(loss, torch.Tensor)
        # Back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    return rssm