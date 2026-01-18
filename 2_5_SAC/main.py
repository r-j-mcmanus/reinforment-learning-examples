import gymnasium as gym
from gymnasium import Env

import logging
from itertools import count

import torch
from torch import Tensor

# ensure reproducibility
seed = 42
torch.manual_seed(seed)

torch.autograd.set_detect_anomaly(True)

from replay_memory import ReplayMemory
from sac import SAC
from constants import DEVICE


def run():

    NUM_EPISODES = 1_000
    RECORD_EVERY = 10
    WARM_UP = 200

    logging.basicConfig(level=logging.INFO, format='%(message)s')


    env = gym.make('Ant-v5', render_mode='rgb_array', max_episode_steps=200)
    # wrap the env in the record video
    env = gym.wrappers.RecordVideo(
        env=env, 
        video_folder="C:/python/RL/2_5_SAC/video/", 
        name_prefix="training", 
        episode_trigger=lambda x: x % RECORD_EVERY == 0 )

    env = gym.wrappers.RecordEpisodeStatistics(env)

    state, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_max = env.action_space.high
    action_min = env.action_space.low

    memory = ReplayMemory(10_000)
    model = SAC(obs_dim, action_dim, action_max, action_min)

    # tracking how long the episode lasted
    episode_durations = []
                
    for i_episode in range(NUM_EPISODES):

        # randomly initialise the environment, get the corresponding state
        state, _ = env.reset()
        # format appropriately
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        rewards = []

        for step in count():
            assert isinstance(state, Tensor)
            if len(memory) < WARM_UP:
                action = env.action_space.sample()
            else:
                # update on one batch
                model.update(memory)
                action = model.get_action(state)

            observation, reward, terminated, truncated, _ = env.step(action) # gymnasium response to the action
           
            #print((observation[1:5] / 3.14159 * 180).astype(int))

            rewards.append(reward)

            done = terminated or truncated

            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward, terminated)

            # move to the next state for the next loop
            state = next_state

            if done:
                print(f'episode {i_episode} steps {step} rewards {sum(rewards)}')
                episode_durations.append(step + 1)
                break
        
    print(episode_durations)

    env.close()

def evaluate_and_record(model: SAC, env: Env):
    obs, _ = env.reset()
    obs= torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    done = False
    while not done:
        action = model.get_action(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        obs= torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        done = terminated or truncated

if __name__ == "__main__":
    run()
