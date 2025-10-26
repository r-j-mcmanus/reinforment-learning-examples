import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from constants import *


class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        """we build a fully connected feed forward NN to model the state action function
        in particular the nn maps state to a policy 
        then we can pick from the policy which action to take"""
        super(DQN, self).__init__()
        self._n_actions = n_actions
        self.layer_1 = nn.Linear(n_observations, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        """Called with either a single observation of the enviroment to predict the best next action, or with batches during optimisation"""
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

    def eps_greedy_action(self, state: Tensor) -> Tensor:
        if random.random() < 0.5:
            return torch.tensor(random.choice(range(self._n_actions)), device=DEVICE)
        
        return self.greedy_predict(state)

    def greedy_predict(self, state: Tensor) -> Tensor:
        action_distribution = self(state)
        return torch.argmax(action_distribution, dim=1)
    
    def state_action_value(self, states: Tensor, actions: Tensor):
        return self(states).gather(1, actions)


class StabilisingDQN(DQN):
    def __init__(self, n_observations: int, actions_dimention: int):
        super().__init__(n_observations, actions_dimention)
        # by passing self.parameters, the optimiser knows which network is optimised
        self.optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE, amsgrad=True)

    def optimise(self, predicted_state_action_values: Tensor, expected_state_action_values: Tensor):
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted_state_action_values, expected_state_action_values)

        self.optimizer.zero_grad() # removes previously found gradients
        loss.backward() # computes the gradients of the loss with respect to all model parameters

        # In-place gradient clipping at max abs value of 100
        # prevents any gradient from becoming too large
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        # apply gradient decent using the optimizer
        self.optimizer.step()


class TargetDQN(DQN):
    def soft_update(self, stabilising_net: StabilisingDQN):
        stabilising_net_state_dict = stabilising_net.state_dict()
        target_net_state_dict = self.state_dict()
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.load_state_dict(target_net_state_dict)

