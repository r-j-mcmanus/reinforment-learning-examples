import torch

from replay_memory import Memory, Transition
from constants import *
from CriticNet import StabilisingCriticNet, TargetCriticNet
from policy import StabilisingPolicyNet, TargetPolicyNet

def apply_learning_step(memory: Memory, step: int,
                        stabalising_critic_net: StabilisingCriticNet, target_critic_net: TargetCriticNet, 
                        stabalising_policy: StabilisingPolicyNet,  target_policy: TargetPolicyNet):
    """Performs a single DDPG learning step using a sampled batch from memory"""

    if len(memory) < BATCH_SIZE:
        return 

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).detach()
    
    state_batch = torch.cat(batch.state).detach()
    action_batch = torch.cat(batch.action).detach() # value of action used
    reward_batch = torch.cat(batch.reward).detach()

    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE) # if the state is final then there is no expected reward
    with torch.no_grad():
        next_actions = target_policy(non_final_next_states)
        next_state_values[non_final_mask] = target_critic_net(torch.cat([non_final_next_states, next_actions], dim=1)).squeeze()
    expected_state_action_values = (GAMMA * next_state_values) + reward_batch

    predicted_state_action_values = stabalising_critic_net(torch.cat([state_batch, action_batch], dim=1).detach()).squeeze()
    stabalising_critic_net.optimise(predicted_state_action_values, expected_state_action_values, step)
    stabalising_policy.optimise(stabalising_critic_net, state_batch, step)

    target_policy.soft_update(stabalising_policy)
    target_critic_net.soft_update(stabalising_critic_net)
    