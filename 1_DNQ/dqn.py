import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import random
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
