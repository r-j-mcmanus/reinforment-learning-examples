import random
import math

import torch
from torch import Tensor
import torch.nn as nn


from constants import *
from DNQ.dqn import DQN

class Policy:
    def __init__(self, dqn: nn.Module) -> None:
        self.dqn = dqn

    def select_action(self, env, state) -> Tensor:
        raise NotImplemented


class EpsGreedyPolicy(Policy):
    def __init__(self, dqn: DQN) -> None:
        super(EpsGreedyPolicy, self).__init__(dqn)
        self.steps_done = 0


    def select_action(self, env, state) -> Tensor:
        sample = random.random()
        # late into training we do not want to explore as much
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            # Disabling gradient calculation is useful for inference, when you are sure that you will not call 
            # Tensor.backward(). It will reduce memory consumption for computations.
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.dqn(state).max(1).indices.view(1, 1)
        else:
            # get a random valid action from the env and executes it
            return torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)