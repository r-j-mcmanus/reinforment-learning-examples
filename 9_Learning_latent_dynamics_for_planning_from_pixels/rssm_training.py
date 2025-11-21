"""Trains on sequences of observations and actions to learn latent dynamics for one step"""

from collections import deque

import torch
import torch.optim as optim

from rssm import RSSM
from state import State
from replay_memory import ReplayMemory
from latent_memory import LatentMemory

import pandas as pd

from constants import *


def train_rssm(rssm: RSSM, replay_memory: ReplayMemory, latent_memory: LatentMemory, beta_growth: bool = False) -> RSSM:
    """Trains an RSSM on sequences of observations and actions.
    
    Arguments:
    rssm: RSSM
    replay_memory: ReplayMemory
    latent_memory: LatentMemory
    beta_growth: bool - should beta grow from 0 to 1, set to True for initial training
    """
    # Instantiate model
    optimizer = optim.AdamW(rssm.parameters(), lr=1e-3)

    horizon_length = Constants.Behavior.imagination_horizon
    sequence_length = Constants.World.sequence_length
    batch_size = Constants.World.batch_size
    epoch_count = Constants.World.epoch_count
    beta_growth_rate = Constants.World.beta_growth_rate
    hidden_state_dimension = Constants.World.hidden_state_dimension

    # to track loss
    df = pd.DataFrame(columns=['epoch', 'reward_loss', 'reconstruction_loss'] + [f'KL_({t},{i})' for t in range(sequence_length) for i in range(horizon_length) if i < t])

    for epoch in range(epoch_count):  # number of epochs
        loss = 0

        sequence_transitions = replay_memory.sample_sequential()
        h_1 = torch.zeros(batch_size, hidden_state_dimension, device=DEVICE)
        h_t = h_1 # relabeled for ease in for loop

        past_rollouts = deque([], maxlen=horizon_length)
        
        row = {}
        row['epoch'] = epoch

        for t in range(0, sequence_length):
            # transition shape (batch_size, sequence_length, _)
            obs = sequence_transitions.state[:,t] # used in reconstruction and posterior
            action = sequence_transitions.action[:,t] # use to find the next deterministic step
            reward = sequence_transitions.reward[:,t] # used in reward reconstruction loss
            next_actions = sequence_transitions.action[:, t:t+horizon_length] # used in rollout

            posterior_state = rssm.posterior(obs, h_t) # predict what the state should be given the observation and the hidden state
            transition_state: State = rssm.transition(h_t) # predict what the state should be given the hidden state

            # maybe add lag here to store more uncorrilated states improving sampling?
            latent_memory.add(posterior_state.sample, h_t) # maybe take mean instead?

            # getting the diagonal lines in fig 3 c of 1811.04551, does not take the left most node
            s_it = rssm.rollout(posterior_state, h_t, next_actions)
            # this finds s_{i|j} where:
            #  i in [sequence_start, sequence_length]
            #  j = sequence_start is the last observation the state is conditioned on
            past_rollouts.append(s_it)

            kl_loss = 0
            # prevents early collapse of latent space
            beta = min(1.0, (epoch+1) / beta_growth_rate) if beta_growth else 1 
            for i, state_rollout in enumerate(past_rollouts):
                # take the ith so we test against the ith step of the rollout
                if i < len(state_rollout): # the end of the sequence will not have the full horizon
                    kl_latent = rssm.divergence_from_states(transition_state, state_rollout[i]) # KL divergence
                    row[f'kl_({t},{i})'] = float(kl_latent.item())
                    kl_loss += beta * kl_latent

            assert isinstance(kl_loss, torch.Tensor)

            # Reconstruction loss, predicting next_obs from latent state
            reconstruction_loss = rssm.observation_reconstruction_error(posterior_state, obs, h_t)

            # reward loss
            reward_loss = rssm.reward_reconstruction_error(posterior_state, reward, h_t)

            loss += reconstruction_loss + reward_loss - kl_loss
            
            # h for the next loop
            h_t = rssm.deterministic_step(posterior_state.sample, action, h_t)

            row['reconstruction_loss'] = float(reconstruction_loss.item())
            row['reward_loss'] = float(reward_loss.item())

        assert isinstance(loss, torch.Tensor)
        # Back-propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _df = pd.DataFrame([row])
        df = pd.concat([df, _df], ignore_index=True)

        print(_df) 
        print(f"Epoch {epoch}, Total Loss: {loss.item():.4f}")

    df.to_csv('rssm_losses.csv')

    return rssm
