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


def train_rssm(rssm: RSSM, episode_memory: EpisodeMemory, episode: int, epoch: int, df: pd.DataFrame, 
               *,
               latent_overshooting: bool = False) -> tuple[RSSM, pd.DataFrame]:
    """Trains an RSSM on sequences of observations and actions.
    
    Arguments:
    rssm: RSSM
    replay_memory: ReplayMemory
    episode: int - for logging
    """
    # Instantiate model

    horizon_length = Constants.Behaviors.imagination_horizon
    sequence_length = Constants.World.sequence_length
    batch_size = Constants.World.batch_size
    epoch_count = Constants.World.epoch_count
    beta_growth_rate = Constants.World.beta_growth_rate
    hidden_state_dimension = Constants.World.hidden_state_dimension

    # each epoch picks a collection of sequences we Back-propagate the loss from
    # we do this for multiple epochs
    loss = torch.zeros([1])
    # beta varying prevents early collapse of latent space
    beta = 0.01 * min(1.0, len(df) / Constants.World.beta_growth_rate) 
    episode_transitions = episode_memory.sample()
    h_1 = torch.zeros(batch_size, hidden_state_dimension, device=DEVICE)
    h_t = h_1 # relabeled for ease in for loop

    past_rollouts = deque([], maxlen=horizon_length)
    
    row = {}
    row['epoch'] = epoch
    row['episode'] = episode
    row['beta'] = beta

    kl_loss_total = 0
    reconstruction_loss_total = 0
    reward_loss_total = 0

    for t in range(0, sequence_length):
        # transition shape (batch_size, obs/action/reward dim)
        obs = episode_transitions.state[:,t].detach() # used in reconstruction and posterior
        action = episode_transitions.action[:,t].detach() # use to find the next deterministic step
        reward = episode_transitions.reward[:,t].detach() # used in reward reconstruction loss
        next_actions = episode_transitions.action[:, t:t+horizon_length].detach() # used in rollout

        # print('obs.shape', obs.shape)
        # s_{t|t} - state at time t conditioned on observations up to time t
        representation_state: State = rssm.representation(obs, h_t) # predict what the state should be given the observation and the hidden state

        if latent_overshooting:
            kl_loss = get_kl_latent_overshooting_loss(past_rollouts, rssm, representation_state)
            kl_loss = beta * kl_loss / Constants.Behaviors.imagination_horizon
        else:
            kl_loss = beta * get_kl_loss(rssm, representation_state, h_t)

        # Reconstruction loss
        reconstruction_loss = rssm.observation.loss(h_t, representation_state.mean, obs)

        # reward loss
        reward_loss = rssm.reward.loss(h_t, representation_state.mean, reward)

        loss += reconstruction_loss + reward_loss + kl_loss
        
        # getting the diagonal lines in fig 3 c of 1811.04551, does not take the left most node
        # s_{i|t} - state at time i conditioned on observations up to time t where j in [t+1, ..., t+horizon length]
        s_it = rssm.transition_rollout(representation_state, h_t, next_actions)
        
        if latent_overshooting:
            past_rollouts.append(s_it)

        # h for the next loop
        h_t = rssm.deterministic_step(representation_state.sample, action, h_t)

        #row[f'reconstruction_loss_{t}'] = float(reconstruction_loss.item())
        #row[f'reward_loss_{t}'] = float(reward_loss.item())

        kl_loss_total += kl_loss.item()
        reconstruction_loss_total += reconstruction_loss.item()
        reward_loss_total += reward_loss.item()
    
    row[f'total_kl_loss'] = kl_loss_total
    row[f'total_reconstruction_loss'] = reconstruction_loss_total
    row[f'total_reward_loss'] = reward_loss_total
    row[f'total_loss'] = loss.item()

    # Back-propagation
    rssm.optimizer.zero_grad()
    loss.backward()
    max_grad_norm = 100
    torch.nn.utils.clip_grad_norm_(rssm.parameters(), max_grad_norm)
    rssm.optimizer.step()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    print(f"Episode {episode}, Epoch {epoch}, Total Loss: {loss.item():.4f}")

    return rssm, df


def get_kl_loss(rssm: RSSM, representation_state: State, h: torch.Tensor) -> torch.Tensor:
    return rssm.divergence_from_states(representation_state, rssm.transition.forward(h))


def get_kl_latent_overshooting_loss(past_rollouts, rssm, representation_state) -> torch.Tensor:
    kl_loss = torch.zeros([1])
    # Latent overshooting:
    # find kl for same step but where the obs conditioned on ends at different times
    for j in range(len(past_rollouts)):
        rollout = past_rollouts[j]
        latent_step = len(past_rollouts) - 1 - j
        if latent_step >= len(rollout):
            # at the end of a sequence we can roll out no more as there are no actions or obs to condition with
            continue
        rolled_out_state = rollout[latent_step]
        # take the ith so we test against the ith step of the rollout
        kl_latent = rssm.divergence_from_states(representation_state, rolled_out_state) # KL divergence
        #row[f'kl_({t},{t-len(past_rollouts)+j})'] = float(kl_latent.item())
        kl_loss += kl_latent
    return kl_loss