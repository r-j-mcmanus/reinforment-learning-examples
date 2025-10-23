import torch
import torch.nn as nn
from torch.optim import Optimizer

from replay_memory import Memory, Transition
from constants import *
from DNQ.dqn import DQN


def apply_q_learning_step(memory: Memory, policy_net: nn.Module, target_net: nn.Module, optimizer: Optimizer):
    """"""

    if len(memory) < BATCH_SIZE:
        raise ValueError(f'memory {len(memory)} less than batch size {BATCH_SIZE}')
    
    transitions = memory.sample(BATCH_SIZE)

    # will turn 
    #   [('state', 'action', 'next_state', 'reward')]
    # to 
    #   (['state'], ['action'], ['next_state'], ['reward'])]
    batch = Transition(*zip(*transitions))
    
    # find out which state are not final by mapping the lambda across the transition's next state 
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE, dtype=torch.bool)

    # find out which state do not go to a final state mapping the lambda across the transition's next state 
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # change the list of transition data (states, actions and rewards) into tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action) # index of the action used
    reward_batch = torch.cat(batch.reward)

    # Get the state action values for the policy net by doing a forward pass.
    # The (state index, action index) are how the function is accessed on the
    # sampled states.
    # We get reward from the policy_net for the action as gather will pick the 
    # accoiated value for each row given the action.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # get the value from the target network
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE) # if the state is final then there is no expected reward
    with torch.no_grad(): # calculation costs less as we tell pytorch we will not be optimising
        # get the expected state value from the target_net using the memory rewards
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values from the target network
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss which we aim to minimize
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad() # removes previously found gradients
    loss.backward() # computes the gradients of the loss with respect to all model parameters

    # In-place gradient clipping at max abs value of 100
    # prevents any gradient from becoming too large
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # apply gradient decent using the optimizer
    optimizer.step()