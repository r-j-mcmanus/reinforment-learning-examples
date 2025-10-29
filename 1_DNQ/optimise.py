import torch
import torch.nn as nn

from dqn import StabilisingCriticNet, TargetCriticNet
from replay_memory import Memory, Transition
from constants import *


def apply_learning_step(memory: Memory, policy_net: StabilisingCriticNet, target_net: TargetCriticNet):
    """Performs a single DQN learning step using a sampled batch from memory"""

    if len(memory) < BATCH_SIZE:
        return
    
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

    # Current Q-values from policy network
    predicted_state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
    # predicted_state_action_values = state_batch[torch.arange(state_batch.size(0)), action_batch] # this brakes the computation graph and optimisation will not work

    # get the Q-values from the target network
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE) # if the state is final then there is no expected reward
    with torch.no_grad(): # calculation costs less as we tell pytorch we will not be optimising
        # get the expected state value from the target_net using the memory rewards
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    # Compute the expected Q values from the target network
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # -> predicted_state_action_values
    # | | | | ...
    # r -> next_state_values
    # if the nets were the real state_value function then
    # predicted_state_action_values = r + next_state_values

    policy_net.optimise(predicted_state_action_values, expected_state_action_values)
    target_net.soft_update(policy_net)