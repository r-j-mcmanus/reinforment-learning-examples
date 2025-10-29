import torch
import torch.nn.functional as F
from torch import Tensor

from replay_memory import Memory, Transition
from constants import *
from dqn import StabilisingDQN, TargetDQN


def apply_learning_step(memory: Memory, step: int, stabilising_net: StabilisingDQN, target_net: TargetDQN):
    """"""
    if len(memory) < BATCH_SIZE:
        return

    state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = _get_batch(memory)

    bootstrap_state_action_values = _get_bootstrap_state_action_value(target_net, state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask)
    predicted_state_action_values = _get_predicted_state_action_value(stabilising_net, state_batch, action_batch)
    
    stabilising_net.optimise(predicted_state_action_values, bootstrap_state_action_values, step)

    if step % UPDATE_DELAY == 0:
        target_net.soft_update(stabilising_net)


def _get_batch(memory: Memory) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # change the list of transition data (states, actions and rewards) into tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action) # index of the action used
    reward_batch = torch.cat(batch.reward)

    return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask


def _get_bootstrap_state_action_value(target_net: TargetDQN, state_batch: Tensor, action_batch: Tensor, reward_batch: Tensor, non_final_next_states: Tensor, 
                                      non_final_mask: Tensor) -> Tensor:
    next_expected_softmax = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad(): # calculation costs less as we tell pytorch we will not be optimising
        expected_softmax = F.log_softmax(target_net(state_batch), dim=1).gather(1, action_batch.unsqueeze(1))
        expected_softmax = torch.clamp(expected_softmax, min=MUNCHAUSEN_LOWER_BOUND)

        # get the value from the target network
        bootstrap_next_state_action_values = torch.zeros(BATCH_SIZE, device=DEVICE) # if the state is final then there is no expected reward
        
        next_policy = target_net(non_final_next_states)
        
        # get the greedy next action 
        next_greedy_action = next_policy.argmax(1).unsqueeze(1)
        bootstrap_next_state_action_values[non_final_mask] = next_policy.gather(1, next_greedy_action).squeeze(1)
        next_expected_softmax[non_final_mask] = F.log_softmax(next_policy, dim=1).gather(1, next_greedy_action).squeeze(1)

    # Compute the expected Q values from the target network using the Bellman equation
    bootstrap_state_action_values = (bootstrap_next_state_action_values - M_TAU * next_expected_softmax) * GAMMA + reward_batch + ALPHA * M_TAU *  expected_softmax
    return bootstrap_state_action_values


def _get_predicted_state_action_value(stabilising_net: StabilisingDQN, state_batch: Tensor, action_batch: Tensor):
    return stabilising_net(state_batch.detach()).gather(1, action_batch.unsqueeze(1))
