from collections import namedtuple, deque
import itertools
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
    

class NStepReplayMemory(Memory):
    def __init__(self, capacity: int, n: int, discount: float): 
        # from doc: Once a bounded length deque is full, when new items are added, a corresponding number of items are discarded from the opposite end.
        self._memory: deque[Transition] = deque([], maxlen=capacity)
        self._capacity = capacity
        self.discount = discount
        self.n = n

    def push(self, *args):
        self._memory.append(Transition(*args))

    def sample(self, batch_size) -> list[Transition]:
        start_index = random.sample(range(len(self._memory)), batch_size)

        n_step_samples = [self._make_n_step_transition(i) for i in start_index]

        return n_step_samples

    def _make_n_step_transition(self, i: int) -> Transition:
        initial_trans = self._memory[i]
        final_state = self._memory[i+self.n]
        steps = list(itertools.islice(self._memory, i, min(i+self.n, len(self._memory))))
        reward = sum([self.discount**(step_idx)*trans.reward for  step_idx, trans in enumerate(steps)])
        return Transition(initial_trans.state, 
                          initial_trans.action,
                          final_state.state,
                          reward)

    def __len__(self) -> int:
        return len(self._memory)
    