from collections import deque, namedtuple
import random

import torch

Sample = namedtuple('Sample', ['rewards', 'actions', 'prev_actions', 'obs_idxs', 'next_obs_idxs'])

class ReplayBuffer:
    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)

    def store(self, obs, act, act_m1, rew, obs_2):
        self._memory.append(dict(
            obs=obs,
            obs_2=obs_2,
            act=act,
            act_m1=act_m1, # unique for portfolio as the previous action is part of the environment
            rew=rew
        ))

    def sample(self, batch_size) -> Sample:
        """As described in 1312.5602: Algorithm 1, Sample random mini-batch of transitions"""
        sample = random.sample(self._memory, batch_size)
        
        rewards = torch.stack([i['rew'] for i in sample])
        actions = torch.stack([i['act'] for i in sample])
        prev_actions = torch.stack([i['act_m1'] for i in sample])
        obs_idxs = [i['obs'] for i in sample]
        next_obs_idxs = [i['obs_2'] for i in sample]

        return Sample(rewards, actions, prev_actions, obs_idxs, next_obs_idxs)
    