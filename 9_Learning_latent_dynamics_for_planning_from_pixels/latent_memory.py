import random

from collections import deque

from torch import Tensor
from constants import *


class LatentMemory:
    def __init__(self) -> None:
        self._s = torch.empty((0,))  # Initialize empty tensors
        self._h = torch.empty((0,))
        self.head = 0 # for tracking were we add to on the memory is at its max size

    def sample(self) -> tuple[Tensor, Tensor]:
        """returns Constants.Behavior.trajectory_count samples of stochastic world states that were used in training the RSSM.
        Sampled with replacment as the actor is probabalistic.
        
        Return
        ------
        tuple[Tensor, Tensor]: stack of stochastic states, stack of determanistic states 
        """
        n = Constants.Behavior.trajectory_count

        # Check if enough rows exist
        total_rows = self._s.size(0)
        if total_rows < n:
            raise ValueError(f"Not enough samples in memory: have {total_rows}, need {n}")

        # Randomly select indices
        idx = torch.randint(0, total_rows, (n,))
        s = self._s[idx].detach()
        h = self._h[idx].detach()

        return s, h

    def add(self, s: torch.Tensor, h: torch.Tensor):
        # Append new rows to existing tensors
        if self._s.numel() == 0:
            self._s = s
            self._h = h
        else:
            self._s = torch.cat([self._s, s])
            self._h = torch.cat([self._h, h])

        self.head = len(self._s)