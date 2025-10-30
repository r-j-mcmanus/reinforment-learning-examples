import gymnasium as gym
from gymnasium import Env

from itertools import count

import torch
from torch import Tensor

from CriticNet import StabilisingCriticNet, TargetCriticNet
from replay_memory import ReplayMemory
from policy import StabilisingPolicyNet, TargetPolicyNet
from optimise import apply_learning_step
from constants import *

# ensure reproducibility
seed = 42
torch.manual_seed(seed)

torch.autograd.set_detect_anomaly(True)

# following 1509.02971
# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING


def run(env: Env):

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.shape
    assert isinstance(n_actions, tuple)
    n_actions = n_actions[0]

    memory = ReplayMemory(10_000)

    stabalising_critic_net = StabilisingCriticNet(n_observations, n_actions)
    target_critic_net = TargetCriticNet(n_observations, n_actions)
    target_critic_net.load_state_dict(stabalising_critic_net.state_dict()) # insure initially equal

    # as the action space is continuouse we must use a policy directly and not argmax(Q)
    stabalising_policy_net = StabilisingPolicyNet(n_observations, n_actions) 
    target_policy_net = TargetPolicyNet(n_observations, n_actions)
    target_policy_net.load_state_dict(stabalising_policy_net.state_dict()) # insure initially equal

    # tracking how long the episode lasted
    episode_durations = []
                
    for i_episode in range(NUM_EPISODES):

        # randomly initialise the environment, get the corrisponding state
        state, _ = env.reset()
        # format approprietly
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        observations = []
        rewards = []

        for step in count():
            assert isinstance(state, Tensor)
            action = stabalising_policy_net.noisy_actions(state, env)

            observation, reward, terminated, truncated, _ = env.step([action.item()]) # gymnasium response to the action
           
            observations.append(observation.tolist())
            rewards.append(reward)

            reward = torch.Tensor([reward], device=DEVICE)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            if step % 100 == 0:
                print('---------')
                print(f'step {step}')
                print(f'Transition {(state, action, next_state, reward)}')

            # move to the next state for the next loop
            state = next_state

            if done and not truncated:
                a=1

            # Perform one step of the optimization
            apply_learning_step(memory, step,
                                stabalising_critic_net, target_critic_net,
                                stabalising_policy_net, target_policy_net)

            if done:
                print(f'episode {i_episode} step {step}')
                episode_durations.append(step + 1)
                plot(observations, rewards, i_episode)
                break
    print(episode_durations)


def plot(observations: list[list[float]], rewards: list[float], episode: int):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(observations, columns=['y', 'hue'])
    df['x'] = range(len(df))
    sns.scatterplot(data=df, x='x', y='y', hue='hue')
    plt.savefig(f'fig/td3_observations_ep_{episode}')

    df = pd.DataFrame(rewards, columns=['y'])
    df['x'] = range(len(df))
    sns.scatterplot(data=df, x='x', y='y')
    plt.savefig(f'fig/td3_rewards_ep_{episode}')



if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0")
    run(env)
