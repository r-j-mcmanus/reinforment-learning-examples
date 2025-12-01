"""Trains on sequences of observations and actions to learn latent dynamics for one step"""

from collections import deque

import torch
import torch.optim as optim
import pandas as pd

from rssm import RSSM
from state import State
from episode_memory import EpisodeMemory
from latent_memory import LatentMemory
from plots import plot_rssm_data

from constants import *


def train_rssm(rssm: RSSM, episode_memory: EpisodeMemory, episode: int, df: pd.DataFrame | None = None, beta_growth: bool = False) -> tuple[RSSM, pd.DataFrame]:
    """Trains an RSSM on sequences of observations and actions.
    
    Arguments:
    rssm: RSSM
    replay_memory: ReplayMemory
    episode: int - for logging
    beta_growth: bool - should beta grow from 0 to 1, set to True for initial training
    """
    # Instantiate model

    horizon_length = Constants.Behaviors.imagination_horizon
    sequence_length = Constants.World.sequence_length
    batch_size = Constants.World.batch_size
    epoch_count = Constants.World.epoch_count
    beta_growth_rate = Constants.World.beta_growth_rate
    hidden_state_dimension = Constants.World.hidden_state_dimension

    if df is None:
        # to track loss
        df = pd.DataFrame()
    assert isinstance(df, pd.DataFrame)

    for epoch in range(epoch_count):  # number of epochs
        loss = 0
        beta_norm = min(1.0, (epoch+1) / epoch_count) if beta_growth else 1 
        beta = 0.1 * beta_norm
        sequence_transitions = episode_memory.sample()
        h_1 = torch.zeros(batch_size, hidden_state_dimension, device=DEVICE)
        h_t = h_1 # relabeled for ease in for loop

        past_rollouts = deque([], maxlen=horizon_length)
        
        row = {}
        row['epoch'] = epoch
        row['episode'] = episode

        for t in range(0, sequence_length):
            # transition shape (batch_size, obs/action/reward dim)
            obs = sequence_transitions.state[:,t].detach() # used in reconstruction and posterior
            action = sequence_transitions.action[:,t].detach() # use to find the next deterministic step
            reward = sequence_transitions.reward[:,t].detach() # used in reward reconstruction loss
            next_actions = sequence_transitions.action[:, t:t+horizon_length].detach() # used in rollout

            # print('obs.shape', obs.shape)

            posterior_state = rssm.posterior(obs, h_t) # predict what the state should be given the observation and the hidden state
            transition_state: State = rssm.transition(h_t) # predict what the state should be given the hidden state

            # getting the diagonal lines in fig 3 c of 1811.04551, does not take the left most node
            s_it = rssm.rollout(posterior_state, h_t, next_actions)
            # this finds s_{i|j} where:
            #  i in [sequence_start, sequence_length]
            #  j = sequence_start is the last observation the state is conditioned on
            past_rollouts.append(s_it)

            # TODO maybe try re-doing this to make it clearer?
            kl_loss = 0
            kl_n = 0
            # prevents early collapse of latent space
            for i, state_rollout in enumerate(past_rollouts):
                # take the ith so we test against the ith step of the rollout
                if i < len(state_rollout): # the end of the sequence will not have the full horizon
                    kl_latent = rssm.divergence_from_states(transition_state, state_rollout[i]) # KL divergence
                    row[f'kl_({t},{i})'] = float(kl_latent.item())
                    kl_loss += kl_latent
                    kl_n += 1

            assert isinstance(kl_loss, torch.Tensor)
            row['kl_loss'] = float(kl_loss.item())
            kl_loss = beta * kl_loss / kl_n

            # Reconstruction loss, predicting next_obs from latent state
            reconstruction_loss = rssm.observation_reconstruction_error(posterior_state, obs, h_t)

            # reward loss
            reward_loss = rssm.reward_reconstruction_error(posterior_state, reward, h_t)

            loss += reconstruction_loss + reward_loss + kl_loss
            
            # h for the next loop
            h_t = rssm.deterministic_step(posterior_state.mean, action, h_t)

            row['reconstruction_loss'] = float(reconstruction_loss.item())
            row['reward_loss'] = float(reward_loss.item())
            row['beta'] = beta

        assert isinstance(loss, torch.Tensor)
        # Back-propagation
        rssm.optimizer.zero_grad()
        loss.backward()
        max_grad_norm = 100
        torch.nn.utils.clip_grad_norm_(rssm.parameters(), max_grad_norm)
        rssm.optimizer.step()
        
        _df = pd.DataFrame([row])
        df = pd.concat([df, _df], ignore_index=True)

        #print(_df) 
        print(f"Episode {episode}, Epoch {epoch}, Total Loss: {loss.item():.4f}")

    #df.to_csv('rssm_losses.csv')
    #plot_rssm_data(df, episode)

    return rssm, df
