import random
import gymnasium as gym
from itertools import count

import torch

from CriticNet import StabilisingCriticNet, TargetCriticNet
from replay_memory import ReplayMemory
from policy import StabilisingPolicyNet, TargetPolicyNet
from optimise import apply_learning_step
from constants import *

# ensure reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# following 1509.02971
# CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING


def run(env):
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.shape[0]

    memory = ReplayMemory(100_000)

    stabalising_critic_net = StabilisingCriticNet(n_observations, n_actions)
    target_critic_net = TargetCriticNet(n_observations, n_actions)
    target_critic_net.load_state_dict(stabalising_critic_net.state_dict()) # insure initially equal

    
    stabalising_policy_net = StabilisingPolicyNet(n_observations, n_actions) 
    target_policy_net = TargetPolicyNet(n_observations, n_actions)
    target_policy_net.load_state_dict(stabalising_policy_net.state_dict()) # insure initially equal

    # tracking how long the episode lasted
    episode_durations = []
                
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
            action = stabalising_policy_net(state)

            # move into policy
            # add noise
            action += torch.normal(mean=0.0, std=EXPLORATION_STD, size=action.shape).to(DEVICE)
            # clamp to allowed values
            action = action.clamp(torch.tensor(env.action_space.low, device=DEVICE),
                                  torch.tensor(env.action_space.high, device=DEVICE))

            observation, reward, terminated, truncated, _ = env.step(action.item()) # gymnasium response to the action

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

            # Perform one step of the optimization
            apply_learning_step(memory, 
                                stabalising_critic_net, target_critic_net,
                                stabalising_policy_net, target_policy_net)


            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break

