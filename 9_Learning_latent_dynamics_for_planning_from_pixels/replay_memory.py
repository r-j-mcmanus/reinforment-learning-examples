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
        self._memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args):
        self._memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        """As described in 1312.5602: Algorithm 1, Sample random minibatch of transitions"""
        return random.sample(self._memory, batch_size)
    
    def __len__(self) -> int:
        return len(self._memory)
    
    
    def sample_sequential(self, batch_size: int, sequence_length: int) -> list[list[Transition]]:
        """
        Sample a batch of sequences, each of length `sequence_length`.
        Sequences are consecutive transitions from memory.
        """
        memory_size = len(self._memory)
        if memory_size < sequence_length:
            raise ValueError("Not enough transitions to form a sequence of the requested length.")

        sequences = []
        for _ in range(batch_size):
            # Choose a valid start index so the sequence fits in memory
            start_idx = random.randint(0, memory_size - sequence_length)
            seq = []
            for i in range(sequence_length):
                val = self._memory[start_idx + i]
                if val.next_state is None:
                    break
                seq.append(val)
            sequences.append(seq)
        return sequences
