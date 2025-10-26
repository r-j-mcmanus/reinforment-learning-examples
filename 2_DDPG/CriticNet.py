import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from constants import *

class BaseNet(nn.Module):
    def __init__(self, n_observations: int, actions_dimention: int):
        """A fully connected feed forward NN to model the state-value function.

        arguments
        ---------
        n_observations: int
            The number of observations of the enviroment state that we pass to the model
        actions_dimention: int
            The dimentionality of the continuous action space"""
        super().__init__()

        self.layer_1 = nn.Linear(n_observations + actions_dimention, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 1)

        # Layer normalization or dropout can help with stability.
        # Weight initialization: initialize the final layer weights to small values to prevent large Q-values early in training

    def forward(self, x: Tensor) -> Tensor:
        """Called with batches during optimisation"""
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)
    
    def grad_action(self, state: Tensor, action: Tensor) -> Tensor:
        # clone(): Creates a copy of the action tensor so you don’t modify the original.
        # detach(): Ensures the new tensor is not connected to any previous computation graph.
        # requires_grad_(True): Tells PyTorch to track operations on this tensor so gradients can be computed with respect to it.
        action = action.clone().detach().requires_grad_(True)
        # This performs a forward pass through the critic network.
        # critic_net takes state and action as input and returns a scalar Q-value.
        # Since action has requires_grad=True, PyTorch builds a computation graph that links the output q_value to action
        q_value = self(torch.cat([state, action], dim=1))
        # This triggers backpropagation from q_value (a scalar) through the computation graph.
        # PyTorch computes the gradient of q_value with respect to all tensors that have requires_grad=True — in this case, action.
        q_value.backward(torch.ones_like(q_value))
        # This retrieves the gradient of q_value with respect to action
        action_grad = action.grad

        if not isinstance(action_grad, Tensor):
            raise Exception

        return action_grad


class StabilisingCriticNet(BaseNet):
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


class TargetCriticNet(BaseNet):
    def __init__(self, n_observations: int, actions_dimention: int):
        super().__init__(n_observations, actions_dimention)

    def soft_update(self, stabilising_net: StabilisingCriticNet):
        stabilising_net_state_dict = stabilising_net.state_dict()
        target_net_state_dict = self.state_dict()
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.load_state_dict(target_net_state_dict)