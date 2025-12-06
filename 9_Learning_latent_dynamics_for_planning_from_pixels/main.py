import random
import pickle
from pathlib import Path

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
from ddgp_training import train_ddpg
from ddgp import DDPG

from plots import plot_env_data, plot_rssm_data, plot_ll_data

def main():
    set_global_seed()

    # dataframes for tracking training data
    df_rssm = pd.DataFrame()
    df_ll = pd.DataFrame()
    df_env = pd.DataFrame()

    # Create the environment
    env = gym.make('Ant-v5', render_mode = 'rgb_array', terminate_when_unhealthy=False)
    # env = RecordVideo(env, video_folder="9_Learning_latent_dynamics_for_planning_from_pixels/videos/", 
    #                    name_prefix="ant_run", episode_trigger=lambda x: True)

    obs_size: int = env.observation_space.shape[0] # type: ignore
    action_size: int = env.action_space.shape[0] # type: ignore

    # Number of steps to run
    max_num_steps = 999
    num_episodes = Constants.Common.num_episodes

    # Simple replay memory to store observations and actions
    record_ep_step = 10 # How many env interactions we allow before we train the actor and RSSM
    path = Path("9_Learning_latent_dynamics_for_planning_from_pixels/cache/my_object.pkl")
    if path.exists():
        with path.open("rb") as f:
            memory: EpisodeMemory = pickle.load(f)
    else:
        memory = EpisodeMemory()
        memory = warm_up_memory(env, memory)
        with path.open("wb") as f:
            pickle.dump(memory, f)


    # Instantiate the RSSM model for planning
    rssm = RSSM(obs_size, action_size)
    rssm, df_rssm = train_rssm(rssm, memory, episode=0, epoch=0, df=df_rssm) # initial training on warm up data

    ddpg = DDPG(action_size, 1)

    for episode in range(1, num_episodes+1):

        df_env = walk_env(env, memory, rssm, ddpg, episode, df_env)

        if True: # (episode % record_ep_step == 0) :
            plot_env_data(df_env, 0)
            for epoch in range(Constants.World.epoch_count):
                print(f'Training RSSM on episode {episode}')
                rssm, df_rssm = train_rssm(rssm, memory, episode, epoch, df_rssm)
                
                # no point training actor until we start caring about the kl divergence
                if len(df_rssm) > 150:
                    print(f'Training Actor Critic on episode {episode}')
                    df_ll = train_ddpg(rssm, memory, ddpg, episode, epoch, df_ll)
            plot_rssm_data(df_rssm, 0)
            plot_ll_data(df_ll, 0)


    # Close the environment
    env.close()


def walk_env(env: Env, memory: EpisodeMemory, rssm: RSSM, ddpg: DDPG, episode: int, df: pd.DataFrame) -> pd.DataFrame:
    # Reset the environment to get the initial observation
    with torch.no_grad():
        obs, _ = env.reset()
        # obs is an np.ndarray
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        h = torch.zeros(Constants.World.hidden_state_dimension, device=DEVICE)
        s = rssm.representation(obs, h).sample

        total_reward = 0

        for step in range(Constants.World.max_number_steps):
            assert isinstance(obs, Tensor)
            action = ddpg.select_action(s) # actor.predict(s)
            action = action.squeeze()
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
            # need to detach and clone 
            memory.add_step(obs.detach().clone(), 
                            action.detach().clone(),
                            torch.Tensor([1-done], device=DEVICE), # mask for terminal states
                            reward.detach().clone())
            
            # Check if the episode is done
            if done:
                print(f'Env: steps {step} episode {episode} total reward {total_reward}')
                # episode_durations.append(step + 1)
                # plot_durations()
                break

            assert isinstance(next_obs, Tensor)
            obs = next_obs
            h = rssm.deterministic_step(s, action, h)
            s = rssm.representation(obs, h).sample
        
        memory.end_episode()

        row = {}
        row['episode'] = episode
        row['steps'] = step
        row['total_reward'] = total_reward
        _df = pd.DataFrame([row])
        df = pd.concat([df, _df], ignore_index=True)
    return df


def warm_up_memory(env, memory: EpisodeMemory) -> EpisodeMemory:
    with torch.no_grad():
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        while len(memory) < Constants.World.capacity_episodes:
            for step in range(Constants.World.max_number_steps):
                # Sample a random action from the action space
                action = env.action_space.sample()
                
                # Take a step in the environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                reward = torch.Tensor([reward], device=DEVICE)

                done = terminated or truncated

                # Store the transition in memory
                memory.add_step(obs, 
                                torch.from_numpy(action), 
                                torch.Tensor([1-done], device=DEVICE), # mask for terminal states
                                reward)
                
                # Check if the episode is done
                if done:
                    break
                else:
                    obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

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

    # CuDNN deterministic behaviour (optional)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed set to {seed}]")

if __name__ == "__main__":
    main()