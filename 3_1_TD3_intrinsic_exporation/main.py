import random
import gymnasium as gym
from gymnasium import Env
from itertools import count
from torch import Tensor

import torch

from CriticNet import StabilisingCriticNet, TargetCriticNet
from replay_memory import ReplayMemory
from policy import StabilisingPolicyNet, TargetPolicyNet, IntrinsicRewardNet
from optimise import apply_learning_step
from constants import *

# ensure reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# following 1802.09477
# Addressing Function Approximation Error in Actor-Critic Methods

# This tries to incurage exploring by picking a policy that aims to pick actions that maximize the difference between the two Q values
# this doesnt seem to do much
# try making a world model, then you can track steps into the future, picking states that down the road are unexplored? 


def run(env: Env):
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    state, _ = env.reset()
    n_observations = len(state)
    shape = env.action_space.shape
    if not isinstance(shape, tuple):
        raise Exception
    n_actions = shape[0]

    memory = ReplayMemory(10_000)

    stabalising_critic_net_1 = StabilisingCriticNet(n_observations, n_actions)
    target_critic_net_1 = TargetCriticNet(n_observations, n_actions)
    target_critic_net_1.load_state_dict(stabalising_critic_net_1.state_dict()) # insure initially equal

    stabalising_critic_net_2 = StabilisingCriticNet(n_observations, n_actions)
    target_critic_net_2 = TargetCriticNet(n_observations, n_actions)
    target_critic_net_2.load_state_dict(stabalising_critic_net_2.state_dict()) # insure initially equal

    stabalising_policy_net = StabilisingPolicyNet(n_observations, n_actions) 
    target_policy_net = TargetPolicyNet(n_observations, n_actions)
    target_policy_net.load_state_dict(stabalising_policy_net.state_dict()) # insure initially equal

    stabalising_intrinsic_net = IntrinsicRewardNet(n_observations, n_actions) 
    target_intrinsic_net = TargetPolicyNet(n_observations, n_actions)
    target_intrinsic_net.load_state_dict(stabalising_intrinsic_net.state_dict()) # insure initially equal


    # tracking how long the episode lasted
    episode_durations = []
                
    action_low, action_high = None, None
    if isinstance(env.action_space, gym.spaces.Box):
        action_low = torch.tensor(env.action_space.low, device=DEVICE)
        action_high = torch.tensor(env.action_space.high, device=DEVICE)


    for i_episode in range(num_episodes):

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        observations = []
        rewards = []

        for step in count():
            assert isinstance(state, Tensor)

            #action = stabalising_policy_net.noisy_actions(state, env)
            # action = stabalising_intrinsic_net.noisy_actions(state, env)
            action = stabalising_intrinsic_net(state)
            
            if step % 100 == 0:
                print(f'step {step}')
                print(f'action {action.item()}')
                q_1 = stabalising_critic_net_1(torch.cat([state, action], dim=1))
                q_2 = stabalising_critic_net_2(torch.cat([state, action], dim=1))
                d_q = torch.abs(q_1 - q_2).mean().item()
                print(f'Delta Q {d_q}')

            # Following penalty is not from a paper but to address an issue i saw, check if it does anything!
            # penalize invalid actions
            penalty = torch.tensor(0.0, device=DEVICE)
            valid_action = action.clone()
            if isinstance(action_low, Tensor) and isinstance(action_high, Tensor):
                if torch.any(action < action_low) or torch.any(action > action_high):
                    valid_action = torch.clamp(action, action_low, action_high)
                    # add a continouse penalty that grows the more invalid an action is
                    penalty = INVALID_ACTION_PENALTY * torch.sum(torch.abs(valid_action - action))


            observation, reward, terminated, truncated, _ = env.step([valid_action.item()]) # gymnasium response to the action

            observations.append(observation.tolist()+ [action.item()])
            rewards.append(reward)

            reward = torch.Tensor([reward], device=DEVICE) + penalty

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # move to the next state for the next loop
            state = next_state

            if terminated:
                a=1

            if done:
                # force final update
                apply_learning_step(memory, UPDATE_DELAY, env,
                                    stabalising_critic_net_1, target_critic_net_1,
                                    stabalising_critic_net_2, target_critic_net_2,
                                    stabalising_policy_net, target_policy_net, 
                                    stabalising_intrinsic_net, target_intrinsic_net)
                episode_durations.append(step + 1)
                plot(observations, rewards, i_episode)
                print(f'Episode {i_episode} ended with step {step}')
                break
            else:
                # Perform one step of the optimization
                apply_learning_step(memory, step, env,
                                    stabalising_critic_net_1, target_critic_net_1,
                                    stabalising_critic_net_2, target_critic_net_2,
                                    stabalising_policy_net, target_policy_net, 
                                    stabalising_intrinsic_net, target_intrinsic_net)


def plot(observations: list[list[float]], rewards: list[float], episode: int):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(observations, columns=['y', 'dy', 'hue'])
    df['x'] = range(len(df))
    sns.scatterplot(data=df, x='x', y='y', hue='hue')
    plt.savefig(f'fig/td3_observations_ep_{episode}')

    plt.close()

    df = pd.DataFrame(observations, columns=['y', 'dy', 'hue'])
    df['x'] = range(len(df))
    sns.scatterplot(data=df, x='y', y='dy', hue='hue')
    plt.savefig(f'fig/td3_phase_ep_{episode}')

    plt.close()

    # df = pd.DataFrame(rewards, columns=['y'])
    # df['x'] = range(len(df))
    # sns.scatterplot(data=df, x='x', y='y')
    # plt.savefig(f'fig/td3_rewards_ep_{episode}')

    # plt.close()


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    run(env)
