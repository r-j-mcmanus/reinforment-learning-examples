import logging
import random
from itertools import count
from pathlib import Path

import gymnasium as gym
from gymnasium import Env
import torch
from torch import Tensor
import numpy as np
import pandas as pd

# ensure reproducibility
seed = 42
torch.manual_seed(seed)

torch.autograd.set_detect_anomaly(True)

from replay_memory import ReplayMemory
from drop_q import DropQ
from constants import DEVICE


def run():

    NUM_EPISODES = 400
    RECORD_EVERY = 10
    WARM_UP = 200

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    env = gym.make('Ant-v5', render_mode='rgb_array', max_episode_steps=200)
    # wrap the env in the record video
    env = gym.wrappers.RecordVideo(
        env=env, 
        video_folder=Path(__file__).resolve().parent / 'videos', 
        name_prefix="training", 
        episode_trigger=lambda x: x % RECORD_EVERY == 0 )

    env = gym.wrappers.RecordEpisodeStatistics(env)

    state, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high
    action_min = env.action_space.low

    memory = ReplayMemory(1_000_000)
    model = DropQ(obs_dim, action_dim, action_max, action_min)

    # tracking how long the episode lasted
    episode_durations = []
                
    for i_episode in range(NUM_EPISODES):

        # randomly initialise the environment, get the corresponding state
        state, _ = env.reset()
        # format appropriately
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        rewards = []
        logs = pd.DataFrame(columns=['reward','steps'])

        for step in count():
            assert isinstance(state, Tensor)
            if len(memory) < WARM_UP:
                action = env.action_space.sample()
            else:
                critic_loss, actor_loss = model.update(memory)
                action = model.get_action(state)

            observation, reward, terminated, truncated, _ = env.step(action) # gymnasium response to the action
           
            z_axis = env.unwrapped.data.xmat[env.unwrapped.model.body("torso").id].reshape(3,3)[:,2]
            if z_axis[2] < 0:
                terminated = True
            
            rewards.append(reward)

            done = terminated or truncated

            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward, terminated)

            # move to the next state for the next loop
            state = next_state

            if done:
                if len(logs) == 0:
                    logs = pd.DataFrame({
                            'reward': [sum(rewards)],
                            'steps': [step + 1],
                    })
                else:
                    logs = pd.concat([logs, pd.DataFrame([{
                            'reward': sum(rewards),
                            'steps': step + 1,
                    }])], ignore_index=True)

                print(f'episode {i_episode} steps {step} rewards {sum(rewards)}')
                episode_durations.append(step + 1)
                break
    
    logs.to_csv('logs', header=True)
    env.close()


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
    set_global_seed()
    run()
