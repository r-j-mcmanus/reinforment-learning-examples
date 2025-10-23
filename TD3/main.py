import random
import gymnasium as gym
from gymnasium import Env
from itertools import count
from torch import Tensor

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

    stabalising_critic_net_1 = StabilisingCriticNet(n_observations, n_actions)
    target_critic_net_1 = TargetCriticNet(n_observations, n_actions)
    target_critic_net_1.load_state_dict(stabalising_critic_net_1.state_dict()) # insure initially equal

    stabalising_critic_net_2 = StabilisingCriticNet(n_observations, n_actions)
    target_critic_net_2 = TargetCriticNet(n_observations, n_actions)
    target_critic_net_2.load_state_dict(stabalising_critic_net_2.state_dict()) # insure initially equal

    stabalising_policy_net = StabilisingPolicyNet(n_observations, n_actions) 
    target_policy_net = TargetPolicyNet(n_observations, n_actions)
    target_policy_net.load_state_dict(stabalising_policy_net.state_dict()) # insure initially equal

    # tracking how long the episode lasted
    episode_durations = []
                
    action_low, action_high = None, None
    if isinstance(env.action_space, gym.spaces.Box):
        action_low = torch.tensor(env.action_space.low, device=DEVICE)
        action_high = torch.tensor(env.action_space.high, device=DEVICE)


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

            action = stabalising_policy_net.noisy_forward(state, action_low, action_high)

            # Following penalty is not from a paper but to address an issue i saw, check if it does anything!

            # penalize invalid actions
            penalty = torch.tensor(0.0, device=DEVICE)
            valid_action = action.clone()
            if isinstance(action_low, Tensor) and isinstance(action_high, Tensor):
                if torch.any(action < action_low) or torch.any(action > action_high):
                    valid_action = torch.clamp(action, action_low, action_high)
                    # add a continouse penalty that grows the more invalid an action is
                    penalty = INVALID_ACTION_PENALTY * torch.sum(torch.abs(valid_action - action))

            observation, reward, terminated, truncated, _ = env.step(valid_action.item()) # gymnasium response to the action

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

            if done:
                # force final update
                apply_learning_step(memory, UPDATE_DELAY, action_low, action_high,
                                    stabalising_critic_net_1, target_critic_net_1,
                                    stabalising_critic_net_2, target_critic_net_2,
                                    stabalising_policy_net, target_policy_net)
                episode_durations.append(t + 1)
                break
            else:
                # Perform one step of the optimization
                apply_learning_step(memory, t, action_low, action_high,
                                    stabalising_critic_net_1, target_critic_net_1,
                                    stabalising_critic_net_2, target_critic_net_2,
                                    stabalising_policy_net, target_policy_net)
