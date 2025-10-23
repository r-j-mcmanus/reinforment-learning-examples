import torch
from torch import Tensor

import numpy as np
from replay_memory import Memory, Transition
from constants import *
from CriticNet import StabilisingCriticNet, TargetCriticNet
from policy import StabilisingPolicyNet, TargetPolicyNet

def apply_learning_step(memory: Memory, step: int, action_low: Tensor | None, action_high: Tensor | None, 
                        stabilising_critic_net_1: StabilisingCriticNet, target_critic_net_1: TargetCriticNet, 
                        stabilising_critic_net_2: StabilisingCriticNet, target_critic_net_2: TargetCriticNet, 
                        stabilising_policy: StabilisingPolicyNet,  target_policy: TargetPolicyNet):
    """Performs a single TD3 learning step using a sampled batch from memory"""

    if len(memory) < BATCH_SIZE:
        raise ValueError(f'memory {len(memory)} less than batch size {BATCH_SIZE}')
    
    transitions = memory.sample(BATCH_SIZE)

    # will turn 
    #   [('state', 'action', 'next_state', 'reward')]
    # to 
    #   (['state'], ['action'], ['next_state'], ['reward'])]
    batch = Transition(*zip(*transitions))
    
    # change the list of transition data (states, actions and rewards) into tensors
    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action) # index of the action used
    reward_batch = torch.cat(batch.reward)

    with torch.no_grad():
        # Predict next actions using the target polic
        next_actions = target_policy.noisy_forward(next_state_batch, action_low, action_high)

        # find the state predicted state value for the stabilising critice from the current state 
        # and the predicted state value for the next state using the target network 
        predicted_state_action_values_1 = stabilising_critic_net_1(torch.cat([state_batch, action_batch], dim=1))
        predicted_state_action_values_2 = stabilising_critic_net_2(torch.cat([state_batch, action_batch], dim=1))

        # key td3 step: reduce overestimation bias.
        # compute target Q-values using the minimum of the two target critics
        bootstrap_val = torch.min(
            target_critic_net_1(torch.cat([next_state_batch, next_actions], dim=1)).detach(), 
            target_critic_net_2(torch.cat([next_state_batch, next_actions], dim=1)).detach(), 
        )

        expected_state_action_values = reward_batch + GAMMA * bootstrap_val
    
    # optimise the stabilising net based on the minibatch
    stabilising_critic_net_1.optimise(predicted_state_action_values_1, expected_state_action_values)
    stabilising_critic_net_2.optimise(predicted_state_action_values_2, expected_state_action_values)

    # allows for critic to represent policy by allowing multiple changes based on policy
    # larger UPDATE_DELAY reduces variance
    if step % UPDATE_DELAY == 0:
        # update the actor policy using samples policy gradient 
        stabilising_policy.deterministic_update(stabilising_critic_net_1, state_batch) # picking 1 seems arbitery

        # soft update the target networks
        target_policy.soft_update(stabilising_policy)
        target_critic_net_1.soft_update(stabilising_critic_net_1)
        target_critic_net_2.soft_update(stabilising_critic_net_2)
