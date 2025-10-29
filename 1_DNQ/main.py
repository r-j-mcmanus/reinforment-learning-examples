import gymnasium as gym
from gymnasium import Env

from itertools import count

import torch
from torch import Tensor

from dqn import StabilisingCriticNet, TargetCriticNet
from replay_memory import ReplayMemory
from optimise import apply_learning_step
from constants import *


# ensure reproducibility
seed = 42
torch.manual_seed(seed)

# see Human-level control through deep reinforcement learning
# and Playing Atari with Deep Reinforcement Learning
# for initial implementation details

def run(env: Env):
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 10_000
    else:
        num_episodes = 1_000

    # tracking how long the episode lasted
    episode_durations = []

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n # type: ignore

    memory = ReplayMemory(10_000)

    stabalising_critic_net = StabilisingCriticNet(n_observations, n_actions)
    target_critic_net = TargetCriticNet(n_observations, n_actions)
    target_critic_net.load_state_dict(stabalising_critic_net.state_dict()) # insure initially equal
    
    for i_episode in range(num_episodes):

        # randomly initialise the enviroment, get the corrisponding state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        # unsqueeze inserts a dimention into a tensor
        # e.g torch.unsqueeze(x, 1)
        # tensor([[ 1],
        #         [ 2],
        #         [ 3],
        #         [ 4]])

        for t in count():
            assert isinstance(state, Tensor)
            action = stabalising_critic_net.eps_greedy_action(state, env)

            observation, reward, terminated, truncated, _ = env.step(action.item()) # gymnasium response to the action
            # truncated = if the episode ended due to a time limit, step limit, or other artificial cutoff.
            # terminated = if the episode ended because the agent reached a terminal state

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
            apply_learning_step(memory, stabalising_critic_net, target_critic_net)

            if done:
                print(f'episode {i_episode} step {t}')
                episode_durations.append(t + 1)
                # plot_durations()
                break
    print(episode_durations)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    run(env)
