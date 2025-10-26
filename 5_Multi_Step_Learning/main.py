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

# see Human-level control through deep reinforcement learning
# and Playing Atari with Deep Reinforcement Learning
# for initial implementation details


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

    memory = ReplayMemory(100_000)
    
    # we use both a target net and a policy net to stabalise training
    # this is as when training the policy network TODO
    stabalising_net = StabilisingDQN(n_observations, n_actions)
    target_net = TargetDQN(n_observations, n_actions)
    target_net.load_state_dict(stabalising_net.state_dict()) # insure initially equal

    # tracking how long the episode lasted
    episode_durations = []
                
    for i_episode in range(num_episodes):
        assert isinstance(state, Tensor)

        # randomly initialise the enviroment, get the corrisponding state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for t in count():
            assert isinstance(state, Tensor)
            
            action = stabalising_net.eps_greedy_action(state)

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
            apply_learning_step(memory, UPDATE_DELAY, stabalising_net, target_net)

            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break

