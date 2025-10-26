from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from constants import *

from CriticNet import StabilisingCriticNet

class Policy:
    def __init__(self) -> None:
        pass

    def select_action(self, state: Tensor) -> Tensor:
        with torch.no_grad():
            action = self.forward(state)
        return action

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class BasePolicyNet(nn.Module, Policy):
    def __init__(self, n_observations: int, actions_dimention: int):
        """A fully connected feed forward NN to model the policy function.

        arguments
        ---------
        n_observations: int
            The number of observations of the enviroment state that we pass to the model
        actions_dimention: int
            The dimentionality of the continuous action space
        """
        super().__init__()

        self.layer_1 = nn.Linear(n_observations, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, actions_dimention)

    def forward(self, x: Tensor) -> Tensor:
        """Called with either a single observation of the enviroment to predict the best next action, or with batches during optimisation"""
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)
    
    def grad_weights(self, state: Tensor) -> Tensor:
        """returns the gradient of the policy with respect to the model weights for each state as a len(state), len(weights) Tensor"""
        grads = []

        for single_state in state:
            single_state = single_state.unsqueeze(0)  # Add batch dimension
            self.zero_grad()  # Clear previous gradients

            output = self.forward(single_state)
            output_sum = output.sum()  # Sum to get scalar output for gradient computation

            output_sum.backward()

            _grad = []
            for param in self.parameters():
                if isinstance(param.grad, Tensor):
                    _grad.append(param.grad.view(-1))
                else:
                    raise Exception
                
            grads.append(torch.cat(_grad))

        return torch.stack(grads)  # Shape: (batch_size, total_number_of_parameters)


class StabilisingPolicyNet(BasePolicyNet):
    def __init__(self, n_observations: int, actions_dimention: int):
        super().__init__(n_observations, actions_dimention)
        # by passing self.parameters, the optimiser knows which network is optimised

    def optimise(self, critic_net: StabilisingCriticNet, state_batch: Tensor):
        self.zero_grad()  # Clear gradients once for the whole batch

        stabalised_action_predictions = self(state_batch)

        grad_critic_net_wrt_action = critic_net.grad_action(state_batch, stabalised_action_predictions) # shape: (batch_size, action_dim)
        grad_policy_wrt_weights = self.grad_weights(state_batch) # shape: (batch_size, total_params)

        # Compute dot product for each sample and average
        weighted_grads = grad_critic_net_wrt_action.unsqueeze(2) * grad_policy_wrt_weights.unsqueeze(1)  # (batch_size, action_dim, total_params)
        final_grad = weighted_grads.mean(dim=0).sum(dim=0)  # (total_params,)

        with torch.no_grad(): # turns off tracking of parameters, speeds up calculations
            start = 0
            for layer in [self.layer_1, self.layer_2, self.layer_3]:
                weight = layer.weight
                num_params = weight.numel() # the correct way to get the number of weights, len will give the number of rows
                update = final_grad[start: start + num_params].view_as(weight) # reshapes that slice to match the shape of the layer’s weight tensor
                weight += update
                start += num_params



class TargetPolicyNet(BasePolicyNet):
    def __init__(self, n_observations: int, actions_dimention: int):
        super().__init__(n_observations, actions_dimention)

    def soft_update(self, stabilising_net: StabilisingPolicyNet):
        stabilising_net_state_dict = stabilising_net.state_dict()
        target_net_state_dict = self.state_dict()
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.load_state_dict(target_net_state_dict)