from collections import namedtuple, deque
import random

import torch
from torch import Tensor

from constants import *

# As described in 1312.5602: Algorithm 1,  store transitions in the replay memory of capacity N

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
    #TODO go from steps to episodes
    # sample episode proportiaonl to length
    # the uniform step in episode
    def __init__(self, capacity): 
        # from doc: Once a bounded length deque is full, when new items are added, a corresponding number of items are discarded from the opposite end.
        self._memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args):
        self._memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """As described in 1312.5602: Algorithm 1, Sample random minibatch of transitions"""
        return random.sample(self._memory, batch_size)
    
    def __len__(self) -> int:
        return len(self._memory)
    
    def sample_sequential(self) -> Transition:
        """
        Sample batch of sequences, each of length `sequence_length`.
        Sequences are consecutive transitions from memory.

        Returns
        -------
        Transition 
            - state: Torch.tensor with shape (batch_size, sequence_length, observation_size)
            - action: Torch.tensor with shape (batch_size, sequence_length, action_size)
            - reward: Torch.tensor with shape (batch_size, sequence_length, 1)
            - next_state: None
        """
        sequence_length = Constants.World.sequence_length
        batch_size = Constants.World.batch_size

        memory_size = len(self._memory)
        if memory_size < sequence_length:
            raise ValueError("Not enough transitions to form a sequence of the requested length.")

        sequences = []
        while len(sequences) < batch_size:
            # Choose a valid start index so the sequence fits in memory
            start_idx = random.randint(0, memory_size - sequence_length)
            seq = []

            for i in range(sequence_length):
                val = self._memory[start_idx + i]
                if val.next_state is None:
                    break
                seq.append(val)
            if len(seq) != sequence_length:
                continue
                
            # todo what should we do about early ending in seq?
            seq = Transition(*zip(*seq)) 
            seq = Transition(
                state=torch.cat(seq.state), # an element of seq.state is shape [1, observation_size], so cat not stack
                action=torch.stack(seq.action),
                reward=torch.stack(seq.reward),
                next_state=None, # we dont need to worry about the next state being none
            )
            sequences.append(seq)
        
        return Transition(
            state=torch.stack([i.state for i in sequences]),
            action=torch.stack([i.action for i in sequences]),
            reward=torch.stack([i.reward for i in sequences]),
            next_state=None
        )