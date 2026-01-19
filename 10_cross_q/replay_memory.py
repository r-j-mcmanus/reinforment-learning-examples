from collections import namedtuple, deque
import random

import torch
from torch import Tensor
import numpy as np

from constants import DEVICE

# As described in 1312.5602: Algorithm 1,  store transitions in the replay memory of capacity N

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Memory:
    def __init__(self, capacity):
        pass

    def push(self, *args) -> None:
        raise NotImplemented
    
    def sample(self, batch_size) -> list[Transition]:
        raise NotImplemented

    def __len__(self) -> int:
        raise NotImplemented
    

class ReplayMemory(Memory):
    def __init__(self, capacity): 
        # from doc: Once a bounded length deque is full, when new items are added, a corresponding number of items are discarded from the opposite end.
        self._memory = deque([], maxlen=capacity)

    def push(self, state: Tensor, action: np.ndarray, next_state: Tensor, reward: float, done: bool):
        action = torch.from_numpy(action).to(DEVICE)
        done = float(done)
        self._memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size) -> Transition:
        """As described in 1312.5602: Algorithm 1, Sample random minibatch of transitions"""
        batch_size = min(batch_size, len(self._memory))
        transitions = random.sample(self._memory, batch_size)
        
        state = torch.concat([i.state for i in transitions])
        action = torch.stack([i.action for i in transitions])
        next_state = torch.concat([i.next_state for i in transitions])
        reward = torch.tensor([[i.reward] for i in transitions], device=DEVICE)
        done = torch.tensor([[i.done] for i in transitions], device=DEVICE)
        
        return Transition(state, action, next_state, reward, done)
    
    def __len__(self) -> int:
        return len(self._memory)