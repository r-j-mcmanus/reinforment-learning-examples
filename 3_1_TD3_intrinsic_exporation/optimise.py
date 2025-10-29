import torch
from gymnasium import Env

from replay_memory import ReplayMemory, Transition
from constants import *
from CriticNet import StabilisingCriticNet, TargetCriticNet
from policy import StabilisingPolicyNet, TargetPolicyNet, IntrinsicRewardNet


def apply_learning_step(memory: ReplayMemory, step: int, env: Env, 
                        stabilising_critic_net_1: StabilisingCriticNet, target_critic_net_1: TargetCriticNet, 
                        stabilising_critic_net_2: StabilisingCriticNet, target_critic_net_2: TargetCriticNet, 
                        stabilising_policy: StabilisingPolicyNet,  target_policy: TargetPolicyNet,
                        stabalising_intrinsic_net: IntrinsicRewardNet, target_intrinsic_net: TargetPolicyNet):
    """Performs a single TD3 learning step using a sampled batch from memory"""
    if len(memory) < BATCH_SIZE:
        return        

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    
    # next_state_batch has to be treated diferently
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).detach()

    # change the list of transition data (states, actions and rewards) into tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action) # index of the action used
    reward_batch = torch.cat(batch.reward)
    
    bootstrap_val = torch.zeros(BATCH_SIZE, device=DEVICE) # if the state is final then there is no expected reward
    with torch.no_grad():
        # Predict next actions using the target policy
        next_actions = target_policy.noisy_actions(non_final_next_states, env)


        # key td3 step: reduce overestimation bias.
        # compute target Q-values using the minimum of the two target critics
        bootstrap_val[non_final_mask] = torch.min(
            target_critic_net_1(torch.cat([non_final_next_states, next_actions], dim=1)).squeeze(), 
            target_critic_net_2(torch.cat([non_final_next_states, next_actions], dim=1)).squeeze(), 
        )

        # the reward for trying a new state action pair indicated by how well the 2 targets agree on the state-value
        q_1 = target_critic_net_1(torch.cat([state_batch, action_batch], dim=1)).squeeze()
        q_2 = target_critic_net_2(torch.cat([state_batch, action_batch], dim=1)).squeeze()
        intrinsic_reward_val = torch.abs(q_1 - q_2)

        # intrinsic_reward_vals = target_intrinsic_net(state_batch)

        # outside of the no_grad as it's graph is used by the critic net optimiser
        expected_state_action_values = reward_batch + GAMMA * bootstrap_val + intrinsic_reward_val

        if step % 100 == 0:
            print(intrinsic_reward_val[:3])
            a=1

    # find the predicted state value for the stabilising critic from the current state 
    predicted_state_action_values_1 = stabilising_critic_net_1(torch.cat([state_batch, action_batch], dim=1).detach()).squeeze()
    predicted_state_action_values_2 = stabilising_critic_net_2(torch.cat([state_batch, action_batch], dim=1).detach()).squeeze()

    # optimise the stabilising net based on the minibatch
    stabilising_critic_net_1.optimise(predicted_state_action_values_1, expected_state_action_values, step)
    stabilising_critic_net_2.optimise(predicted_state_action_values_2, expected_state_action_values, step)


    # allows for critic to represent policy by allowing multiple changes based on policy
    # larger UPDATE_DELAY reduces variance
    if step % UPDATE_DELAY == 0:
        # update the actor policy using samples policy gradient 
        stabilising_policy.optimise(stabilising_critic_net_1, state_batch, step) # picking 1 seems arbitery
        stabalising_intrinsic_net.optimise(stabilising_critic_net_1, stabilising_critic_net_2, state_batch, step)

        # soft update the target networks
        target_policy.soft_update(stabilising_policy)
        target_intrinsic_net.soft_update(stabalising_intrinsic_net)
        target_critic_net_1.soft_update(stabilising_critic_net_1)
        target_critic_net_2.soft_update(stabilising_critic_net_2)
