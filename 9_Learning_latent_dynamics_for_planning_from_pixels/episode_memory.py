from collections import namedtuple, deque
import random
import bisect

import torch
from torch import Tensor

from constants import *

# As described in 1312.5602: Algorithm 1,  store transitions in the replay memory of capacity N

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Episode:
    """Stores a full episode: list[Transition]."""
    def __init__(self):
        self.transitions: list[Transition] = []

    def add(self, *args):
        self.transitions.append(Transition(*args))

    def __len__(self):
        return len(self.transitions)

class EpisodeMemory:
    """
    Stores entire episodes instead of individual transitions.
    Each item is a Sequence(transitions=[Transition, ...]).
    """

    def __init__(self, capacity_episodes: int = 1000, seq_len: int = Constants.World.sequence_length):
        """
        Capacity counts NUMBER OF EPISODES, not transitions.
        """
        self._episodes: deque[Episode] = deque([], maxlen=capacity_episodes)
        self._current_episode = Episode()

        self.seq_len = seq_len

        self._episode_lengths = []   # list[int]
        self._cumulative = []        # prefix sums

    # ----------------------------------------
    # for episode creation 

    def add_step(self, *args):
        """Add a single transition to the current in-progress episode."""
        self._current_episode.add(*args)

    def end_episode(self):
        """Finish current episode, push it into memory."""
        ep = self._current_episode
        L = len(ep)

        if L > 0:
            # append newly completed episode
            self._episodes.append(ep)
            self._episode_lengths.append(L - self.seq_len + 1)

            # enforce alignment with the deque's maxlen (in case of overflow)
            if len(self._episodes) != len(self._episode_lengths):
                self._episode_lengths = self._episode_lengths[-len(self._episodes):]

            # rebuild cumulative lengths for sampling
            self._recompute_cumulative()

        # start fresh
        self._current_episode = Episode()

    def _recompute_cumulative(self):
        """Recompute prefix sum of episode-lengths for weighted sampling."""
        self._cumulative = []
        total = 0
        for v in self._episode_lengths:
            total += max(v, 0)
            self._cumulative.append(total)

        self._total = total

    # ----------------------------------------
    # sampling sequences from episodes

    
    def sample(self, batch_size: int = Constants.World.batch_size) -> Transition:
        """
        Returns a Transition of Tensors of sequences with shape:
            batch_size × seq_len
        Sampling is proportional to episode length.
        """
        out = []
        for _ in range(batch_size):
            out.append(self._sample_one())

        return Transition(
            state=torch.stack([i.state for i in out]),
            action=torch.stack([i.action for i in out]),
            reward=torch.stack([i.reward for i in out]),
            next_state=torch.stack([i.next_state for i in out])
        )

    def _sample_one(self) -> Transition:
        """Sample one fixed-length subsequence from an episode."""
        assert len(self._episodes) > 0, "Replay buffer is empty."

        # 1. sample an episode proportionally to (length - seq_len + 1)
        r = random.randint(1, self._total)
        idx = bisect.bisect_left(self._cumulative, r)
        ep = self._episodes[idx]
        max_start = max(len(ep) - self.seq_len, 0)

        # 2. uniform sample of start index inside selected episode
        start = random.randint(0, max_start)
        seq = ep.transitions[start:start + self.seq_len]

        return Transition(
            state=torch.stack([t.state for t in seq]),
            action=torch.stack([t.action for t in seq]),
            reward=torch.stack([t.reward for t in seq]),
            next_state=torch.stack([t.next_state for t in seq]) # for masking end of episode states 
        )
    
    def sample_for_latent(self, trajectory_count: int = Constants.Behaviors.trajectory_count) -> tuple[dict, Tensor, Tensor]:
        """
        Returns
        -------
        states: shape(,,) a vector of observations for the episode
        actions: shape(,,) a vector of actions for the episode
        """
        # keys are episodes containing obvs we want to latently learn from, 
        # values are the index from that episode we want to work from
        latent_starts: dict[int: list[int]] = {}
        obs = []
        actions = []

        for i in range(trajectory_count):
            # 1. sample an episode proportionally to (length - seq_len + 1)
            r = random.randint(1, self._total)
            ep_idx = bisect.bisect_left(self._cumulative, r)
            ep = self._episodes[ep_idx]

            # 2. uniform sample of start index inside selected episode
            start = random.randint(0, len(ep))

            if latent_starts.get(ep_idx) is None:
                latent_starts[ep_idx] = [start]
            else:
                latent_starts.get(ep_idx).append(start)

        obs = []
        actions = []
        for ep_idx in latent_starts.keys():
            obs.append(
                torch.nn.functional.pad(
                    torch.stack([t.state for t in self._episodes[ep_idx].transitions])
                    , pad = (0,0,0,Constants.World.max_number_steps - len(self._episodes[ep_idx].transitions)) 
                )
            )
            actions.append(
                torch.nn.functional.pad(
                    torch.stack([t.action for t in self._episodes[ep_idx].transitions])
                    , pad = (0,0,0,Constants.World.max_number_steps - len(self._episodes[ep_idx].transitions)) 
                )
            )
        return latent_starts, torch.stack(obs), torch.stack(actions)


    def __len__(self):
        """Total number of transitions across stored episodes."""
        return sum(len(ep) for ep in self._episodes)