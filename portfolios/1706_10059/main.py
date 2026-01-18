from pprint import pprint
from environment import AssetTrainingEnvironment
import numpy as np
# from actors.gnu_actor import GnuActor as Actor
from actors.convolution_actor import ConvolutionActor as Actor
import random
import torch

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


set_global_seed()

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
# SYMBOLS: list[str] = [
#     'META'
# ] 
# all in london stock exchange
LSE_SYMBOLS = [
    'RIO.L',    # Materials / Mining
    'DGE.L',    # Consumer Staples (Beverages)
    'RR.L',     # Industrials / Aerospace
    'HSBA.L',   # Financials / Banking
    'SHEL.L',   # Energy
    'AZN.L'     # Healthcare / Pharmaceuticals
]

def main(symbols: list[str]):
    env = AssetTrainingEnvironment(symbols = symbols)
    # sets initial portfolio vec to (1,0,...,0) = w_0
    # gives x_t where t=1
    total_rewards: list[float] = []

    actor= Actor(feature_dim=env.observation_space.shape[-1])

    for episode in range(CONSTANTS.EPISODE_COUNT):
        total_rewards = batch_train(env, actor, total_rewards)
        if episode % 10 ==0:
            render_current_actor(env, actor)

    print('------------')
    print('min        -', min(total_rewards))
    print('max        -', max(total_rewards))
    print('starting 5 -', total_rewards[:5])
    print('ending 5   -', total_rewards[-5:])
    print('------------')
    a=1


def render_current_actor(env: AssetTrainingEnvironment, actor: Actor):
    with env.render():
        portfolio = torch.zeros(1, len(SYMBOLS)+1).float().to(DEVICE)
        portfolio[:, 0] = 1 # initial portfolio is all cash

        obs, info = env.reset(options={'batch_size': 1, 'obs_index':[100]})
        portfolio: torch.Tensor = actor(obs, portfolio) # action based on current observation

        while True:
            # the new portfolio we have decided to transition to w_t using the observation X_T and portfolio w_{t-1}
            obs, reward, terminated,truncated, info= env.step(portfolio) # how the env changes with our action
            portfolio: torch.Tensor = actor(obs, portfolio) # what action we take given the new environment
            if terminated or truncated:
                break


def batch_train(env: AssetTrainingEnvironment, 
                actor: Actor, 
                total_rewards: list[float],
                log_reward = True,
                batch_size =  CONSTANTS.BATCH_SIZE
                ):
    # initial portfolio is all cash
    portfolio = torch.zeros(batch_size, len(SYMBOLS)+1).float().to(DEVICE)
    portfolio[:, 0] = 1

    obs, info = env.reset(options={'batch_size': batch_size})

    rewards: list[torch.Tensor] = []
    while True:
        portfolio: torch.Tensor = actor(obs, portfolio) # action based on current observation
        # the new portfolio we have decided to transition to w_t using the observation X_T and portfolio w_{t-1}
        obs, reward, terminated,truncated, info= env.step(portfolio) # how the env changes with our action
        rewards.append(reward)
        if terminated or truncated:
            break

    total_reward: torch.Tensor = torch.stack(rewards).sum(dim=0).mean()
    actor.optimizer.zero_grad()
    (-total_reward).backward() # to ascend 
    actor.optimizer.step()
    float_total_reward = float(total_reward.detach().numpy())
    total_rewards.append(float_total_reward)
    if log_reward:
        print('total_reward' , float_total_reward)
    return total_rewards


if __name__ == '__main__':
    main(SYMBOLS)