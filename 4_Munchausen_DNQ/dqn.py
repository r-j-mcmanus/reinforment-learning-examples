import random
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from gymnasium import Env

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

        self.steps_done = 0


    def forward(self, x: Tensor) -> Tensor:
        """Called with either a single observation of the environment to predict the best next action, or with batches during optimisation"""
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

    def eps_greedy_action(self, state: Tensor, env: Env) -> Tensor:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() < eps_threshold:
            return torch.tensor([env.action_space.sample()], device=DEVICE)
        return self.greedy_predict(state)

    def greedy_predict(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            action_distribution = self(state)
        return torch.argmax(action_distribution, dim=1)
    
    def state_action_value(self, states: Tensor, actions: Tensor):
        with torch.no_grad():
            return self(states).gather(1, actions)


class StabilisingDQN(DQN):
    def __init__(self, n_observations: int, actions_dimension: int):
        super().__init__(n_observations, actions_dimension)
        # by passing self.parameters, the optimiser knows which network is optimised
        self.optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.count = 0

    def optimise(self, predicted_state_action_values: Tensor, expected_state_action_values: Tensor, step: int):
        """
        Compute Huber loss between predicted and expected Q-values
        and perform one step of optimization
        """
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted_state_action_values.squeeze(1), expected_state_action_values)

        self.optimizer.zero_grad() # removes previously found gradients
        loss.backward() # computes the gradients of the loss with respect to all model parameters

        # prevents any gradient from becoming too large
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10)
        # apply gradient decent using the optimizer
        self.optimizer.step()

        if self.count % 100 == 0:
            print(f'Citic loss {loss.item()}')
        self.count +=1 


class TargetDQN(DQN):
    def soft_update(self, stabilising_net: StabilisingDQN):
        """
        Soft update target network:
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        """
        stabilising_net_state_dict = stabilising_net.state_dict()
        target_net_state_dict = self.state_dict()
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.load_state_dict(target_net_state_dict)
