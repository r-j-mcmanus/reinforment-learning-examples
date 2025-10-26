import torch

from replay_memory import NStepReplayMemory, Transition
from constants import *
from dqn import StabilisingDQN, TargetDQN


def apply_learning_step(memory: NStepReplayMemory, step: int, stabilising_net: StabilisingDQN, target_net: TargetDQN):
    """"""

    if len(memory) < BATCH_SIZE:
        raise ValueError(f'memory {len(memory)} less than batch size {BATCH_SIZE}')
    
    transitions = memory.sample(BATCH_SIZE)

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

    predicted_state_action_values = stabilising_net.state_action_value(state_batch, action_batch)

    # Todo I dont think this will work, dims are prob wrong
    with torch.no_grad(): # calculation costs less as we tell pytorch we will not be optimising        
        # get the value from the target network
        bootstrap_next_state_action_values = torch.zeros(BATCH_SIZE, device=DEVICE) # if the state is final then there is no expected reward
        # get the greedy next action 
        bootstrap_next_state_action_values[non_final_mask] = target_net.greedy_predict(non_final_next_states)
        # Compute the expected Q values from the target network using the Bellman equation
        bootstrap_state_action_values = bootstrap_next_state_action_values * GAMMA ** memory.n + reward_batch

    stabilising_net.optimise(predicted_state_action_values, bootstrap_state_action_values)

    if step % UPDATE_DELAY == 0:
        target_net.soft_update(stabilising_net)
