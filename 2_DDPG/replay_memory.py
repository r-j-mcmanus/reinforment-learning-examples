from collections import namedtuple, deque
import random

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
    def __init__(self, capacity): 
        # from doc: Once a bounded length deque is full, when new items are added, a corresponding number of items are discarded from the opposite end.
        self._memory = deque([], maxlen=capacity)

    def push(self, *args):
        self._memory.append(Transition(*args))

    def sample(self, batch_size) -> list[Transition]:
        """As described in 1312.5602: Algorithm 1, Sample random minibatch of transitions"""
        return random.sample(self._memory, batch_size)
    
    def __len__(self) -> int:
        return len(self._memory)