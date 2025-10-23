import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        """we build a fully connected feed forward NN to model the state action function
        in particular the nn maps state to a policy 
        then we can pick from the policy which action to take"""
        super(DQN, self).__init__()

        self.layer_1 = nn.Linear(n_observations, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, n_actions)

    def forward(self, x: Tensor) -> Tensor:
        """Called with either a single observation of the enviroment to predict the best next action, or with batches during optimisation"""
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)
