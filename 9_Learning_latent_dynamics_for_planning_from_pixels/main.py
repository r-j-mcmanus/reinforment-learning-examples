
import gymnasium as gym
import numpy as np
import torch

from rssm import RSSM
from rssm_training import train_rssm
from constants import *
from replay_memory import ReplayMemory


def main():
    # Create the environment
    env = gym.make('Ant-v5') #, render_mode = 'human')
    obs_size: int = env.observation_space.shape[0] # type: ignore
    action_size: int = env.action_space.shape[0] # type: ignore

    # Number of steps to run
    max_num_steps = 1000
    num_episodes = 100

    # Simple replay memory to store observations and actions
    memory = ReplayMemory(100_000)
    seed_ep_count = 10 # how many episodes we want to warm up on
    record_ep_step = 10 # how many episodes we allow before we record a new episode, allows for policy to change so we likely get an episode with new features
    memory = warm_up_memory(env, memory, seed_ep_count, max_num_steps)

    # Instantiate the RSSM model for planning
    state_size = 32# size of the hidden state
    rssm = RSSM(state_size, obs_size, action_size)
    horizon_size = 3 # how far into the future do both train the RSSM and plan
    rssm_train_step = 100 # when adding to 
    rssm = train_rssm(rssm, memory, horizon_size) # initial training on warm up data

    for episode in range(num_episodes):

        record_ep = (episode % record_ep_step == 0) # bool indicating we record to episode in memory 
        
        # Reset the environment to get the initial observation
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        memory.new_episode()

        for step in range(max_num_steps):
            # TODO follow a policy
            action = env.action_space.sample()
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            reward = torch.Tensor([reward], device=DEVICE)

            done = terminated or truncated

            if done:
                next_obs = None
            else:
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            if record_ep:
                memory.push(obs, torch.from_numpy(action), next_obs, reward)

            # Render the environment (optional)
            #env.render()
            
            # Check if the episode is done
            if done:
                print(f'episode {0} step {step}')
                # episode_durations.append(step + 1)
                # plot_durations()
                break

        # train with the newly added episode
        if record_ep:
            rssm = train_rssm(rssm, memory, overshooting_distance=horizon_size)
    
    # Close the environment
    env.close()


def warm_up_memory(env, memory: ReplayMemory, seed_ep_count: int, max_num_steps: int) -> ReplayMemory:
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    for episode in range(seed_ep_count):
        for step in range(max_num_steps):
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
            memory.push(obs, torch.from_numpy(action), next_obs, reward)
            
            # Check if the episode is done
            if done:
                break
    return memory

if __name__ == "__main__":
    main()