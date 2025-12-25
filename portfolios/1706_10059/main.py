from pprint import pprint
from environment import AssetEnvironment
import numpy as np
from actor import Actor
import random
import torch

from constants import *

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

set_global_seed()

# look into interactivebrokers when a real api is needed!
# https://www.reddit.com/r/algotrading/comments/1njxtr5/what_is_for_you_the_best_broker_for_algorithmic/

# for ease use all symbols in the same exchange so the data is consistent amongst them

# all in NASDAQ
SYMBOLS = [
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
BATCH_SIZE = 200
EPISODE_COUNT = 1000

env = AssetEnvironment(symbols = SYMBOLS)
# sets initial portfolio vec to (1,0,...,0) = w_0
# gives x_t where t=1
total_rewards: list[float] = []

actor= Actor(feature_dim=env.observation_space.shape[-1])

for n in range(EPISODE_COUNT):
    # todo batch multiple sequences at once
    obs, info = env.reset(options={'batch_size': BATCH_SIZE})

    portfolio = torch.zeros(BATCH_SIZE, len(SYMBOLS)+1).float().to(DEVICE)
    portfolio[:, 0] = 1 # initial portfolio is all cash

    rewards: list[torch.Tensor] = []
    while True:
        # the new portfolio we have decided to transition to w_t using the observation X_T and portfolio w_{t-1}
        portfolio = actor.forward(obs, portfolio)

        obs, reward, terminated,truncated, info= env.step(portfolio)
        rewards.append(reward)

        if terminated or truncated:
            break

    total_reward: torch.Tensor = sum(rewards).mean()
    actor.optimizer.zero_grad()
    (-total_reward).backward() # to ascend 
    actor.optimizer.step()
    float_total_reward = float(total_reward.detach().numpy())
    total_rewards.append(float_total_reward)
    if n % 10 == 0:
        print(n, float_total_reward)
print('------------')
print('min        -', min(total_rewards))
print('max        -', max(total_rewards))
print('starting 5 -', total_rewards[:5])
print('ending 5   -', total_rewards[-5:])
print('------------')
a=1