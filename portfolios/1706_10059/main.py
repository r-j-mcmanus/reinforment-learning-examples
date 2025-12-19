from environment import AssetEnvironment
import numpy as np
from actor import Actor
import random
import torch


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

SYMBOLS = ['META', 'NFLX']
BATCH_SIZE = 50

env = AssetEnvironment(symbols = SYMBOLS)
# sets initial portfolio vec to (1,0,...,0) = w_0
# gives x_t where t=1
total_rewards: list[float] = []

actor= Actor()

for n in range(BATCH_SIZE):
    obs, info = env.reset()
    portfolio = info['portfolio']
    rewards: list[torch.Tensor] = []
    while True:
        # the new portfolio we have decided to transiton to w_t using the observationj X_T and portfolio w_{t-1}
        portfolio = actor.forward(obs, portfolio)

        obs, reward, terminated,truncated, info= env.step(portfolio)
        rewards.append(reward)

        if terminated or truncated:
            a=1
            break

    total_reward: torch.Tensor = -sum(rewards)
    actor.optimizer.zero_grad()
    total_reward.backward()
    actor.optimizer.step()
    total_rewards.append(float(total_reward.detach().numpy()))
a=1