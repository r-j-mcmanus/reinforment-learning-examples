import pandas as pd
import torch
from torch import Tensor

from episode_memory import EpisodeMemory
from constants import *
from rssm import RSSM
from ddgp import DDPG


def train_critic(rssm: RSSM, ddpg:DDPG, latent_starts: Tensor, obs: Tensor, actions: Tensor) -> float:
    # see eq 5, and fig 3 which shows t=1 is the latent start conditioned on observation data
    
    # detach world model results as we dont use their trees
    with torch.no_grad():
        # where we start dreaming for the traj count of dreams
        z_latent_start, h_latent_start = rollout_transitions(rssm, latent_starts, obs, actions)
        # the im horizon of future states following the target actor policy
        dreamt_z, dreamt_h = dream_sequence(rssm, ddpg, z_latent_start, h_latent_start)
        rewards, _ = rssm.reward.forward(dreamt_h, dreamt_z)
        target_values = ddpg.critic_target.forward(dreamt_z)

    # we use the gradients of L_targets when finding the critic loss function
    with torch.no_grad():
        # the target values for the dreamt rollout the target network given dreamt rewards the actor gives rise to
        L_targets = compute_lambda_targets(rewards, target_values)

    # the critic aims to predict future returns for the actor's policy given the current state
    critic_value = torch.concat([
        ddpg.critic.forward(z_latent_start).unsqueeze(dim=0), 
        ddpg.critic.forward(dreamt_z[:-1])
    ])

    assert critic_value.shape == L_targets.shape
    critic_loss = ddpg.critic_loss_foo(critic_value, L_targets)

    ddpg.critic_optimizer.zero_grad()
    critic_loss.backward()
    ddpg.critic_optimizer.step()

    return critic_loss.item()


def train_actor(rssm: RSSM, ddpg:DDPG, latent_starts: Tensor, obs: Tensor, actions: Tensor) -> tuple[float,float,float,float]:
    # Actor loss eq.6 2010.02193
    rho = Constants.Behaviors.actor_gradient_mixing
    
    # where we start dreaming for the traj count of dreams
    z_latent_start, h_latent_start = rollout_transitions(rssm, latent_starts, obs, actions)
    
    # the im horizon of future states following the target actor policy
    dreamt_z, dreamt_h = dream_sequence(rssm, ddpg, z_latent_start, h_latent_start)
    dreamt_rewards, _ = rssm.reward.forward(dreamt_h, dreamt_z)
    

    # we use the last target when making the lambda target, but we use the start target when using loss
    target_values: Tensor = ddpg.critic_target(dreamt_z)
    # see eq.4 2010.02193 for lambda def, basically for each t in horizon, take the weighted mean of the critic reward and the state's reward 
    L_targets = compute_lambda_targets(dreamt_rewards, target_values)
    target_values = torch.concat([ddpg.critic_target.forward(z_latent_start).unsqueeze(dim=0), target_values[:-1]])
    action_weight = (L_targets - target_values).detach()

    # reinforce_loss
    # the associated gradient is from the distribution not the sample, so we do not use the reparametrisation trick
    action_sf, log_prob_sf, dist_sf = ddpg.actor(dreamt_z, reparameterize=False)
    reinforce_loss = -rho * (log_prob_sf.sum(-1) * action_weight).mean()
    
    # reparam_loss
    reparam_loss = -(1-rho) * (L_targets.mean())

    # entropy loss
    entropy_loss = -Constants.Behaviors.actor_entropy_loss_scale * dist_sf.entropy().sum(dim=-1).mean()

    actor_loss = reinforce_loss + reparam_loss + entropy_loss

    ddpg.actor_optimizer.zero_grad()
    actor_loss.backward()
    ddpg.actor_optimizer.step()

    return actor_loss.item(), reinforce_loss.item(), reparam_loss.item(), entropy_loss.item()


def train_ddpg(rssm: RSSM, memory: EpisodeMemory, ddpg: DDPG, episode: int, epoch: int, df: pd.DataFrame):    
    """
    Sample N latent trajectories, these will be in M different episodes. 
    Pick the M sequences, find the values for h and q for all elements in that episode with the RSSM.
    For the actual index in that episode we latently learn from, roll out the transitions for H steps
    """
    latent_starts, obs, actions = memory.sample_for_latent()

    critic_loss = train_critic(rssm, ddpg, latent_starts, obs, actions)
    actor_total_loss, reinforce_loss, dynamic_backprop_loss, entropy  = train_actor(rssm, ddpg, latent_starts, obs, actions)

    # soft update the target networks
    # larger UPDATE_DELAY reduces variance
    # allowing for critic to represent policy by allowing multiple changes based on policy
    #if step % Constants.Behaviour.slow_critic_update_interval == 0:
    ddpg.soft_update()
    
    row = {}
    row['epoch'] = epoch
    row['episode'] = episode
    row['actor_loss'] = actor_total_loss
    row['reinforce_loss'] = reinforce_loss
    row['dynamic_backprop_loss'] = dynamic_backprop_loss
    row['entropy'] = entropy
    row['critic_loss'] = critic_loss
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def rollout_transitions(rssm: RSSM, latent_starts: dict[int, list[int]], obs: Tensor, actions: Tensor)->tuple[Tensor, Tensor]:
    """
    Get the start hidden and representation states for the dreaming sequences
     
    Returns
    -------
    tuple[Tensor, Tensor] - z_latent_start with shape (trajectory count, latent sate dimension) , h_latent_start with shape (trajectory count, hidden sate dimension)
    """

    with torch.no_grad():
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
            zs.append(z.sample)
            h = rssm.deterministic_step(z.sample, a, h)

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
    rewards: (T, B, 1)    - predicted rewards for imagined steps, 1<t<=H
    target_values: (T, B) - predicted state-values for each imagined state (V_t), 1<t<=H
    gamma: discount factor
    lam:   lambda parameter in (0,1)

    Returns
    -------
    lambda_returns: (T, B)
    """
    # sanity / shape handling
    T, B = rewards.shape[0], rewards.shape[1]
    # squeeze reward last dim -> (T, B)
    rewards = rewards.squeeze(-1)

    # handel the case t = H
    v_lambda_tH = rewards[-1] + gamma * target_values[-1]

    lambda_returns = torch.zeros_like(target_values)  # (T, B)
    #inset at the end as its G^lambda_{h:h}
    lambda_returns[-1] = v_lambda_tH

    # backward recursion for 1<t<H
    # important note: target_values[i] = v_{t=i+1}
    next_lambda_target = v_lambda_tH
    for t in range(T - 2, -1, -1):
        # V_t = r_t + gamma * ((1 - lam) * v_{t+1} + lam * V_{t+1}) 
        v_lambda_t = rewards[t] + gamma * (
            (1.0 - lam) * target_values[t + 1] +
            lam * next_lambda_target
        )
        lambda_returns[t] = v_lambda_t
        next_lambda_target = v_lambda_t

    return lambda_returns


def dream_sequence(rssm: RSSM, ddgp: DDPG, z_latent_start: Tensor, h_latent_start: Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    target: in the dreamt sequence, do we use the target network for picking actions

    Returns
    -------
    tuple(Tensor, Tensor) - dreamt z and h, each with shape = (horizon_length, trajectory count, hidden sate dimension)
    """
    dreamt_h = []
    dreamt_z = []

    with rssm.freeze_module():
        # let the actor explore the latent space
        z = z_latent_start.detach()
        h = h_latent_start.detach()
        for _ in range(Constants.Behaviors.imagination_horizon):
            action = ddgp.select_action(z)
            h = rssm.deterministic_step(z, action, h)
            z = rssm.transition.forward(h).sample # transition returns a State where the attribute sample is rsample 
            dreamt_z.append(z)
            dreamt_h.append(h)

    dreamt_z = torch.stack(dreamt_z)
    dreamt_h = torch.stack(dreamt_h)

    return dreamt_z, dreamt_h
