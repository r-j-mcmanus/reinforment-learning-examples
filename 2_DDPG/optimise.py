import torch

from replay_memory import Memory, Transition
from constants import *
from CriticNet import StabilisingCriticNet, TargetCriticNet
from policy import StabilisingPolicyNet, TargetPolicyNet

def apply_learning_step(memory: Memory, 
                        stabalising_critic_net: StabilisingCriticNet, target_critic_net: TargetCriticNet, 
                        stabalising_policy: StabilisingPolicyNet,  target_policy: TargetPolicyNet):
    """Performs a single DDPG learning step using a sampled batch from memory"""

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
        # using the target policy to predict the next step
        next_actions = target_policy(next_state_batch)
        # find the state predicted state value for the stabalising critice from the current state 
        # and the predicted state value for the next state using the target network 
        predicted_state_action_values = stabalising_critic_net(torch.cat([state_batch, action_batch], dim=1))
        expected_state_action_values = reward_batch + GAMMA * target_critic_net(torch.cat([next_state_batch, next_actions], dim=1)).detach()
    
    # optimise the stabalising net based on the minibatch
    stabalising_critic_net.optimise(predicted_state_action_values, expected_state_action_values)

    # update the actor policy using samples policy gradient 
    stabalising_policy.optimise(stabalising_critic_net, state_batch)

    # soft update the target networks
    target_policy.soft_update(stabalising_policy)
    target_critic_net.soft_update(stabalising_critic_net)