import itertools
import random

import torch
import numpy as np
# from actors.gnu_actor import GnuActor as Actor
from environment import AssetTrainingEnvironment
from model.td3_pytorch import TD3
from model.replay_buffer import ReplayBuffer
from constants import DEVICE, CONSTANTS

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



# look into interactivebrokers when a real api is needed!
# https://www.reddit.com/r/algotrading/comments/1njxtr5/what_is_for_you_the_best_broker_for_algorithmic/

# for ease use all symbols in the same exchange so the data is consistent amongst them

# all in NASDAQ
SYMBOLS: list[str] = [
    'META', 
    'NFLX', 
    'AAPL', 
    'MSFT', 
    'NVDA', 
    'AMZN',
    'QCOM',
    'INTC'
]

# all in london stock exchange
LSE_SYMBOLS = [
    'RIO.L',    # Materials / Mining
    'DGE.L',    # Consumer Staples (Beverages)
    'RR.L',     # Industrials / Aerospace
    'HSBA.L',   # Financials / Banking
    'SHEL.L',   # Energy
    'AZN.L'     # Healthcare / Pharmaceuticals
]

def main(symbols: list[str],
         replay_len: int,
         start_steps: int, 
         update_after: int,
         update_every: int,
         history_len: int,
         *,
         seed: int = 42):
    """
    Args:
        symbols (list[str]): List of stock symbols to train with

        replay_len (int): Maximum length of replay buffer.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        seed (int): Seed for random number generators.
    """
    set_global_seed(seed)

    env = AssetTrainingEnvironment(symbols, history_len)
    env.render()
    # sets initial portfolio vec to (1,0,...,0) = w_0
    # gives x_t where t=1
    total_rewards: list[float] = []
    actor_critic = TD3(
        min_action=0.0, max_action=1.0,
        feature_dim=env.observation_space.shape[-1],
        history_len=history_len,
        n_assets=len(symbols)
        )
    actor = actor_critic.actor
    
    obs, _ = env.reset()
    obs_index = env._obs_index

    portfolio_len = len(symbols)+1
    prev_portfolio = torch.zeros(portfolio_len).float().to(DEVICE)
    prev_portfolio[0] = 1 # initial portfolio is all cash
    # we store the obs index

    replay_buffer = ReplayBuffer(replay_len)

    for t in itertools.count(start=0):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            with torch.no_grad():
                portfolio = actor(obs, prev_portfolio).squeeze()
        else:
            portfolio = torch.Tensor(env.action_space.sample())

        next_obs, reward, terminated, truncated, _= env.step(portfolio) # how the env changes with our action
        next_obs_index = env._obs_index

        replay_buffer.store(obs_index, portfolio, prev_portfolio, reward, next_obs_index)

        if t >= update_after and t % update_every == 0:
            actor_critic.train(update_every, env, replay_buffer)

        # update for next loop
        obs = next_obs
        obs_index = next_obs_index
        prev_portfolio = portfolio

        if terminated or truncated:
            break

    print('------------')
    print('min        -', min(total_rewards))
    print('max        -', max(total_rewards))
    print('starting 5 -', total_rewards[:5])
    print('ending 5   -', total_rewards[-5:])
    print('------------')
    a=1

if __name__ == '__main__':
    main(symbols=SYMBOLS,
         start_steps=100,
         replay_len=1000, 
         update_after=100,
         update_every=10, 
         history_len=16)