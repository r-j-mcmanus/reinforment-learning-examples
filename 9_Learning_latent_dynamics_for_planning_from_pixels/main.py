
import gymnasium as gym
import numpy as np
import torch

from rssm import RSSM
from rssm_training import train_rssm
from constants import *
from replay_memory import ReplayMemory

# Create the environment
env = gym.make('Ant-v5') #, render_mode = 'human')

# Reset the environment to get the initial observation
obs, info = env.reset()
obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

# Number of steps to run
num_steps = 1000

# size of the hidden state
state_size = 32
obs_size = env.observation_space.shape[0] # type: ignore
action_size = env.action_space.shape[0] # type: ignore

# Instantiate the RSSM model for planning
model = RSSM(state_size, obs_size, action_size)

# Simple replay memory to store observations and actions
memory = ReplayMemory(100_000)

for step in range(num_steps):
    # Sample a random action from the action space
    action = env.action_space.sample()
    
    # Take a step in the environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    reward = torch.Tensor([reward], device=DEVICE)

    done = terminated or truncated

    if terminated:
        next_obs = None
    else:
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # Store the transition in memory
    memory.push(obs, action, next_obs, reward)

    train_rssm(model, memory, overshooting_distance=3)

    # Render the environment (optional)
    #env.render()
    
    # Check if the episode is done
    if terminated or truncated:
        print(f'episode {0} step {step}')
        # episode_durations.append(step + 1)
        # plot_durations()
        break

# Close the environment
env.close()
