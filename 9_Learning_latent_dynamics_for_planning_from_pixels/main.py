
import gymnasium as gym
from gymnasium import Env
import torch
from torch import Tensor

from rssm import RSSM
from rssm_training import train_rssm
from constants import *
from replay_memory import ReplayMemory
from actor_training import latent_learning
from critic import Critic
from policy import Policy
from latent_memory import LatentMemory

def main():
    # Create the environment
    env = gym.make('Ant-v5') #, render_mode = 'human')
    obs_size: int = env.observation_space.shape[0] # type: ignore
    action_size: int = env.action_space.shape[0] # type: ignore

    # Number of steps to run
    max_num_steps = 1000
    num_episodes = 100

    # Simple replay memory to store observations and actions
    memory = ReplayMemory(Constants.World.dataset_size)
    seed_ep_count = 10 # how many episodes we want to warm up on
    record_ep_step = 10 # how many episodes we allow before we record a new episode, allows for policy to change so we likely get an episode with new features
    memory = warm_up_memory(env, memory, seed_ep_count, max_num_steps)

    # latent memeory 
    latent_memory = LatentMemory()

    # Instantiate the RSSM model for planning
    rssm = RSSM(obs_size, action_size)
    rssm = train_rssm(rssm, memory, latent_memory, beta_growth=True) # initial training on warm up data

    critic = Critic()
    actor = Policy(action_size)

    for episode in range(num_episodes):

        record_ep = (episode % record_ep_step == 0) # bool indicating we record to episode in memory 
        
        walk_env(env, max_num_steps, record_ep, memory, rssm, actor)
        # we should move to recording episodes rather than steps
        # memory.new_episode()

        latent_learning(rssm, latent_memory, critic, actor)

        # train with the newly added episode
        if record_ep:
            rssm = train_rssm(rssm, memory, latent_memory)
    
    # Close the environment
    env.close()


def walk_env(env: Env, max_num_steps: int, record_ep: bool, memory: ReplayMemory, rssm: RSSM, actor: Policy):
    # Reset the environment to get the initial observation
    
    with torch.no_grad():
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

        h = torch.zeros(Constants.World.hidden_state_dimension, device=DEVICE)
        s = rssm.posterior(obs, h).sample

        for step in range(max_num_steps):
            # TODO follow a policy
            assert isinstance(obs, Tensor)
            action = actor.predict(s)
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action.numpy())
            
            reward = torch.Tensor([reward], device=DEVICE)

            done = terminated or truncated

            if done:
                next_obs = None
            else:
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)

            # Store the transition in memory
            if record_ep:
                memory.push(obs, action, next_obs, reward)

            # Render the environment (optional)
            #env.render()
            
            obs = next_obs

            # Check if the episode is done
            if done:
                print(f'walked {step} steps')
                # episode_durations.append(step + 1)
                # plot_durations()
                break

            assert isinstance(obs, Tensor)
            s = rssm.posterior(obs, h).sample
            h = rssm.deterministic_step(s, action, h)


def warm_up_memory(env, memory: ReplayMemory, seed_ep_count: int, max_num_steps: int) -> ReplayMemory:
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    while len(memory) < Constants.World.sequence_length:
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