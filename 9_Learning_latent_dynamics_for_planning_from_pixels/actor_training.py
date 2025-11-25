import torch
from torch import Tensor

from latent_memory import LatentMemory
from constants import *
from critic import Critic
from policy import Policy
from rssm import RSSM

def latent_learning(rssm: RSSM, latent_memory: LatentMemory, critic: Critic, actor: Policy):    
    dreamt_s, dreamt_h = dream_sequence(rssm, actor, latent_memory)

    unflatten_shape = dreamt_s.shape[:-1]

    # dreamt_states is shape (,,state_dim) and we need it to be (,state_dim) for the networks
    # so we flatten
    dreamt_s_flatten = dreamt_s.view(-1, dreamt_s.size(-1))  # shape [batch*seq_len, state_dim]
    dreamt_h_flatten = dreamt_h.view(-1, dreamt_h.size(-1))  # shape [batch*seq_len, state_dim]

    target_values = critic.target(dreamt_s_flatten)
    rewards = rssm.reward(dreamt_h_flatten, dreamt_s_flatten)
    rewards = rewards.view(*unflatten_shape, -1)
    
    # the target values 
    target_values = target_values.view(*unflatten_shape, -1) 

    # we use the gradients of L_targets when finding the critic loss function
    L_targets = compute_lambda_targets(rewards, target_values)

    # find the predicted state value for the stabilising critic from the current state 
    predicted_state_values = critic.predicted(dreamt_s_flatten.detach())
    predicted_state_values = predicted_state_values.view(*unflatten_shape, -1) 

    # optimise the stabilising net based on the minibatch
    actor.optimise(critic, L_targets, dreamt_s.detach()) 
    critic.optimise(predicted_state_values, L_targets.detach())


    # soft update the target networks
    # larger UPDATE_DELAY reduces variance
    # allowing for critic to represent policy by allowing multiple changes based on policy
    #if step % Constants.Behaviour.slow_critic_update_interval == 0:
    actor.soft_update()
    critic.soft_update()


def compute_lambda_targets(rewards: Tensor, target_values: Tensor, gamma: float = 0.995):
    """
    rewards: [T] tensor
    target_values: [T] tensor
    """
    H = Constants.Behaviors.imagination_horizon
    L = Constants.Behaviors.lambda_target_parameter

    # TODO I need to make gamma from p(gamma | h, s)

    V_targets = []
    V_next = rewards[-1] + gamma * target_values[-1] # the last step isnt iterative
    # V_targets.append(V_next) we dont use the final step in the summations eq.5 or eq.6 

    # go backwards as the early are defined in terms of the latter
    for t in reversed(range(H)):
        V_next = rewards[t] + gamma * ( (1 - L) * target_values[t] + L * V_next )
        V_targets.append(V_next)
    V_targets.reverse()
    return torch.stack(V_targets)


def dream_sequence(rssm: RSSM, actor: Policy, latent_memory: LatentMemory) -> tuple[torch.Tensor, torch.Tensor]:
    """Imagination MDP: 
    Sample from the states used during the rssm training for z_0 and h_0. 
    Returns a sequence of states z_{1:H} outputed by the transition predictor ~ p(z_t | z_{t-1}, a_{t-1})
    where actions are drawn from the policy ~ p(a_t | z_t)

    Arguments
    ---------
    rssm: RSSM
    actor: Policy
    latent_memory: LatentMemory

    Returns
    -------
    torch.Tensor - shape = (horizon_length, batch_size, hidden_dimension + state_dimension)
    """
    # this comes directly from env interaction, and so isnt a 'dream'
    s, h = latent_memory.sample() # initial states, shape - (trajectory_count, (hidden|stoch)_state_dim)

    dreamt_h = []
    dreamt_s = []

    # let the actor explore the latent space
    for _ in range(Constants.Behaviors.imagination_horizon):
        action = actor.predict(s)
        h = rssm.deterministic_step(s, action, h)
        s = rssm.transition(h).sample
        dreamt_s.append(s)
        dreamt_h.append(h)

    dreamt_s = torch.stack(dreamt_s)
    dreamt_h = torch.stack(dreamt_h)

    return dreamt_s, dreamt_h
