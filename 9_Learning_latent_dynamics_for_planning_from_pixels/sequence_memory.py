from collections import namedtuple, deque
import random

import torch
from torch import Tensor

from replay_memory import Memory
from constants import *

# As described in 1312.5602: Algorithm 1,  store transitions in the replay memory of capacity N

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Sequence = namedtuple('Sequence', ('transitions',))


class SequenceMemory:
    """
    Stores entire sequences (episodes) instead of individual transitions.
    Each item is a Sequence(transitions=[Transition, ...]).
    """

    def __init__(self, capacity: int):
        """
        Capacity counts NUMBER OF EPISODES, not transitions.
        """
        self._memory: deque[Sequence] = deque([], maxlen=capacity)

    def push(self, transitions) -> None:
        """
        Push a full episode into memory.

        transitions: list of Transition or list of tuples convertible to Transition
        """
        if len(transitions) == 0:
            return  # ignore empty episodes

        # Convert tuples to Transition objects if needed
        eps = []
        for t in transitions:
            if isinstance(t, Transition):
                eps.append(t)
            else:
                eps.append(Transition(*t))

        self._memory.append(Sequence(transitions=eps))

    def sample(self, batch_size: int) -> list[Sequence]:
        """Uniformly sample full sequences (episodes)."""
        return random.sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)
