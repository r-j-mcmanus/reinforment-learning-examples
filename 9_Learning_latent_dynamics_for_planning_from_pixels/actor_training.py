import pandas as pd
import torch
from torch import Tensor

from latent_memory import LatentMemory
from episode_memory import EpisodeMemory
from constants import *
from critic import Critic
from policy import Policy
from rssm import RSSM
from plots import plot_ll_data


def latent_learning(rssm: RSSM, memory: EpisodeMemory, critic: Critic, actor: Policy, episode: int, df: pd.DataFrame):    
    """
    Sample N latent trajectories, these will be in M different episodes. 
    Pick the M sequences, find the values for h and q for all elements in that episode with the RSSM.
    For the actual index in that episode we latently learn from, roll out the transitions for H steps
    """
    for epoch in range(Constants.Behaviors.latent_epoch_count):
        latent_starts, obs, actions = memory.sample_for_latent()

        # detach world model results
        with torch.no_grad():
            z_latent_start, h_latent_start = rollout_transitions(rssm, latent_starts, obs, actions)
        dreamt_z, dreamt_h = dream_sequence(rssm, actor, z_latent_start, h_latent_start)
        with torch.no_grad():
            rewards: Tensor = rssm.reward(dreamt_h, dreamt_z)
        
        dreamt_z, dreamt_h, rewards = dreamt_z.detach(), dreamt_h.detach(), rewards.detach()

        target_values = critic.target(dreamt_z)

        # we use the gradients of L_targets when finding the critic loss function
        L_targets = compute_lambda_targets(rewards, target_values)

        # find the predicted state value for the stabilising critic from the current state 
        predicted_state_values = critic.predicted(dreamt_z)

        # optimise the stabilising net based on the minibatch
        actor_total_loss, reinforce_loss, dynamic_backprop_loss, entropy = actor.optimise(critic, L_targets, dreamt_z.detach()) 
        critic_loss = critic.optimise(predicted_state_values, L_targets)
        print(f'Episode {episode}, Epoch {epoch}, Critic loss {critic_loss}, Actor loss {actor_total_loss}')

        # soft update the target networks
        # larger UPDATE_DELAY reduces variance
        # allowing for critic to represent policy by allowing multiple changes based on policy
        #if step % Constants.Behaviour.slow_critic_update_interval == 0:
        actor.soft_update()
        critic.soft_update()

        row = {}
        row['epoch'] = epoch
        row['episode'] = episode
        row['actor_loss'] = actor_total_loss
        row['reinforce_loss'] = reinforce_loss
        row['dynamic_backprop_loss'] = dynamic_backprop_loss
        row['entropy'] = entropy
        row['critic_loss'] = critic_loss
        _df = pd.DataFrame([row])
        df = pd.concat([df, _df], ignore_index=True)
    plot_ll_data(df, 0)
    return df

def rollout_transitions(rssm: RSSM, latent_starts: dict[int, list[int]], obs: Tensor, actions: Tensor)->tuple[Tensor, Tensor]:
    """
    Get the start hidden and representation states for the dreaming sequences
     
    Returns
    -------
    tuple[Tensor, Tensor] - z_latent_start with shape (trajectory count, latent sate dimension) , h_latent_start with shape (trajectory count, hidden sate dimension)
    """

    # we want to simulate the episodes we sample from
    episode_count = len(latent_starts)
    hidden_state_size = Constants.World.hidden_state_dimension

    sequence_len = obs.shape[1]

    # find the values for the hidden state and the posterior / representation model
    h_0 = torch.zeros([episode_count, hidden_state_size], device=DEVICE) # dim to be fixed later
    h = h_0 # relabeled for ease in for loop
    hs = []
    zs = []
    for idx in range(sequence_len):
        o = obs[:, idx, :]
        a = actions[:, idx, :]
        hs.append(h)
        z = rssm.representation(o, h)
        zs.append(z.mean)
        h = rssm.deterministic_step(z.mean, a, h)

    # stack the states in the episodes we want to make dream sequences from
    z_latent_start = []
    h_latent_start = []
    for ep_idx, ep_key in enumerate(latent_starts.keys()):
        steps = latent_starts[ep_key]
        for step in steps:
            z_latent_start.append(zs[step-1][ep_idx, :])
            h_latent_start.append(hs[step-1][ep_idx, :])

    # shape (trajectory count, sate dimension) 
    z_latent_start = torch.stack(z_latent_start)
    h_latent_start = torch.stack(h_latent_start)

    return z_latent_start, h_latent_start


def compute_lambda_targets(rewards: Tensor, target_values: Tensor, 
                           gamma: float = Constants.Behaviors.discount_factor,
                           lam: float = Constants.Behaviors.lambda_target_parameter):
    """
    Compute lambda-returns G_t^lambda over an imagination horizon.

    Args
    ----
    rewards: (T, B, 1)    - predicted rewards for imagined steps
    target_values: (T, B) - predicted state-values for each imagined state (V_t)
    gamma: discount factor
    lam:   lambda parameter (0,1)

    Returns
    -------
    lambda_returns: (T, B)
    """
    # sanity / shape handling
    T, B = rewards.shape[0], rewards.shape[1]
    # squeeze reward last dim -> (T, B)
    rewards = rewards.squeeze(-1)
    assert rewards.shape == (T, B)
    assert target_values.shape == (T, B)

    next_return: Tensor = target_values[-1]
    lambda_returns = torch.empty_like(target_values)  # (T, B)

    # backward recursion
    for t in range(T - 1, -1, -1):
        # tp1 = time plus 1
        v_tp1 = target_values[t + 1] if t + 1 < T else next_return  # V_{t+1}; for t==T-1 we use v[-1] (bootstrap)
        # V_t = r_t + gamma * ((1 - lam) * v_{t+1} + lam * V_{t+1})
        lambda_ret_t = rewards[t] + gamma * ((1.0 - lam) * v_tp1 + lam * next_return)
        lambda_returns[t] = lambda_ret_t
        next_return = lambda_ret_t

    return lambda_returns



def dream_sequence(rssm: RSSM, actor: Policy, z_latent_start: Tensor, h_latent_start: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Imagination MDP: 
    Sample from the states used during the rssm training for z_0 and h_0. 
    Returns a sequence of states z_{1:H} outputed by the transition predictor ~ p(z_t | z_{t-1}, a_{t-1})
    where actions are drawn from the policy ~ p(a_t | z_t)

    Arguments
    ---------
    rssm: RSSM
    actor: Policy
    z_latent_start: Tensor with shape (trajectory count, latent sate dimension)
    h_latent_start: Tensor with shape (trajectory count, hidden sate dimension)

    Returns
    -------
    tuple(Tensor, Tensor) - dreamt z and h, each with shape = (horizon_length, trajectory count, hidden sate dimension)
    """
    dreamt_h = []
    dreamt_z = []

    # let the actor explore the latent space
    z = z_latent_start
    h = h_latent_start
    for _ in range(Constants.Behaviors.imagination_horizon):
        action = actor.predict(z)
        with torch.no_grad():
            h = rssm.deterministic_step(z, action, h)
            z = rssm.transition(h).sample
        dreamt_z.append(z)
        dreamt_h.append(h)

    dreamt_z = torch.stack(dreamt_z)
    dreamt_h = torch.stack(dreamt_h)

    return dreamt_z, dreamt_h
