import random

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium import Env
import torch
from torch import Tensor

from rssm import RSSM
from rssm_training import train_rssm
from constants import *
from episode_memory import EpisodeMemory
from actor_training import latent_learning
from critic import Critic
from policy import Policy

def main():
    set_global_seed()

    # dataframes for tracking training data
    df_rssm = pd.DataFrame()
    df_ll = pd.DataFrame()

    # Create the environment
    env = gym.make('Ant-v5', render_mode = 'rgb_array')
    # env = RecordVideo(env, video_folder="9_Learning_latent_dynamics_for_planning_from_pixels/videos/", 
    #                    name_prefix="ant_run", episode_trigger=lambda x: True)

    obs_size: int = env.observation_space.shape[0] # type: ignore
    action_size: int = env.action_space.shape[0] # type: ignore

    # Number of steps to run
    max_num_steps = 999
    num_episodes = 100

    # Simple replay memory to store observations and actions
    memory = EpisodeMemory()
    record_ep_step = 10 # How many env interactions we allow before we train the actor and RSSM
    memory = warm_up_memory(env, memory)

    # Instantiate the RSSM model for planning
    rssm = RSSM(obs_size, action_size)
    rssm, df_rssm = train_rssm(rssm, memory, episode=-1, df=df_rssm, beta_growth=True) # initial training on warm up data

    critic = Critic()
    actor = Policy(action_size)

    for episode in range(num_episodes):

        walk_env(env, memory, rssm, actor, episode)

        if (episode % record_ep_step == 0) :
            print(f'Training RSSM on episode {episode}')
            rssm, df_rssm = train_rssm(rssm, memory, episode, df_rssm)
            
            print(f'Training Actor Critic on episode {episode}')
            df_ll = latent_learning(rssm, memory, critic, actor, episode, df_ll)
                

    # Close the environment
    env.close()


def walk_env(env: Env, memory: EpisodeMemory, rssm: RSSM, actor: Policy, episode: int):
    # Reset the environment to get the initial observation
    
    with torch.no_grad():
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        h = torch.zeros(Constants.World.hidden_state_dimension, device=DEVICE)
        s = rssm.posterior(obs, h).sample

        total_reward = 0

        for step in range(Constants.World.max_number_steps):
            assert isinstance(obs, Tensor)
            action = actor.predict(s)
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action.numpy())
            
            reward = torch.Tensor([reward], device=DEVICE)

            total_reward += reward.item()

            done = terminated or truncated

            if done:
                next_obs = None
            else:
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

            # Store the transition in memory
            memory.add_step(obs, 
                            action,
                            torch.Tensor([float(next_obs is not None)], device=DEVICE), # mask for terminal states
                            reward)

            # Render the environment (optional)
            #env.render()
            
            # Check if the episode is done
            if done:
                print(f'Env: steps {step} episode {episode} total reward {total_reward}')
                # episode_durations.append(step + 1)
                # plot_durations()
                break

            assert isinstance(next_obs, Tensor)
            obs = next_obs
            h = rssm.deterministic_step(s, action, h)
            s = rssm.posterior(obs, h).sample
        memory.end_episode()


def warm_up_memory(env, memory: EpisodeMemory) -> EpisodeMemory:
    with torch.no_grad():
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        while len(memory) < Constants.World.batch_size:
            for step in range(Constants.World.max_number_steps):
                # Sample a random action from the action space
                action = env.action_space.sample()
                
                # Take a step in the environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                reward = torch.Tensor([reward], device=DEVICE)

                done = terminated or truncated

                if terminated:
                    next_obs = None
                else:
                    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

                # Store the transition in memory
                memory.add_step(obs, 
                                torch.from_numpy(action), 
                                torch.Tensor([float(next_obs is not None)], device=DEVICE), # mask for terminal states
                                reward)
                
                # Check if the episode is done
                if done:
                    break
                assert isinstance(next_obs, Tensor)
                obs = next_obs

            memory.end_episode()
        return memory


def set_global_seed(seed: int = 42):
    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic behavior (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed set to {seed}]")

if __name__ == "__main__":
    main()