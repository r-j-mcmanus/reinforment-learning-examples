import random
import gymnasium as gym
from gymnasium import Env
from itertools import count
from torch import Tensor

import torch

from dqn import StabilisingDQN, TargetDQN
from replay_memory import ReplayMemory
from optimise import apply_learning_step
from constants import *

# ensure reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# 2007.14430 munchausen reinforcement learning - entropy like penalty to reward


def run(env: Env):

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 500

    # tracking how long the episode lasted
    episode_durations = []

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n # type: ignore

    memory = ReplayMemory(10_000)
    
    stabilising_critic_net = StabilisingDQN(n_observations, n_actions)
    target_net = TargetDQN(n_observations, n_actions)
    target_net.load_state_dict(stabilising_critic_net.state_dict()) # insure initially equal
     
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for step in count():
            assert isinstance(state, Tensor)
            
            action = stabilising_critic_net.eps_greedy_action(state, env)

            observation, reward, terminated, truncated, _ = env.step(action.item()) 

            reward = torch.Tensor([reward], device=DEVICE)

            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # move to the next state for the next loop
            state = next_state

            # Perform one step of the optimization (on the policy network)
            apply_learning_step(memory, step, stabilising_critic_net, target_net)

            if done:
                print(f'episode {i_episode} step {step}')
                episode_durations.append(step + 1)
                break
    print(episode_durations)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    run(env)
