
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from constants import *


class CriticBaseNet(nn.Module):
    def __init__(self, n_observations: int, actions_dimension: int, *, activation_function=F.elu):
        """A fully connected feed forward NN to model the state-value function.

        arguments
        ---------
        n_observations: int
            The number of observations of the environment state that we pass to the model
        actions_dimension: int
            The dimensionality of the continuous action space"""
        super().__init__()

        self.activation_function = activation_function

        self.layer_1 = nn.Linear(n_observations + actions_dimension, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, 1)

        # Layer normalization or dropout can help with stability.
        # Weight initialization: initialize the final layer weights to small values to prevent large Q-values early in training

    def forward(self, x: Tensor) -> Tensor:
        """Called with batches during optimisation"""
        x = self.activation_function(self.layer_1(x))
        x = self.activation_function(self.layer_2(x))
        return self.layer_3(x)


class Critic:
    """
    Following 2010.02193, the actor and critic are both MLPs with ELU activations (1511.07289), which speed up learning in
    deep neural networks and leads to higher classification accuracies.

    f_elu(x) = { x                    if x > 0
                 alpha * (exp(x) - 1) if x <= 0 }
    """
    def __init__(self):
        state_dimension = Constants.World.latent_state_dimension
        self.stabilising_net = StabilisingCriticNet(state_dimension, 0)
        self.target_net = TargetCriticNet(state_dimension, 0)

    def predicted(self, state: Tensor) ->  Tensor:
        return self.stabilising_net(state).squeeze()
    
    def target(self, state: Tensor) ->  Tensor:
        return self.target_net(state).squeeze()
    
    def optimise(self, value: Tensor, target: Tensor) -> float:
        target = target.detach()
        loss_scalar = self.stabilising_net.optimise(value, target)
        return loss_scalar

    def soft_update(self):
        self.target_net.soft_update(self.stabilising_net)


class StabilisingCriticNet(CriticBaseNet):
    def __init__(self, n_observations: int, actions_dimension: int):
        super().__init__(n_observations, actions_dimension)
        self.optimizer = optim.AdamW(self.parameters(), lr=Constants.Behaviors.critic_learning_rate, amsgrad=True)

    def optimise(self, predicted_state_action_values: Tensor, expected_state_action_values: Tensor) -> float:
        """
        Uses smooth l1 loss to perform gradient decent according to the adamW optimizer for the critic network.

        Arguments
        ---------
        predicted_state_action_values: Tensor: shape = (imagination_horizon, trajectory_count), comes from this critic network
        expected_state_action_values: Tensor: Shape = (imagination_horizon ,trajectory_count), comes from the slowly updating target network

        Notes
        -----
        - trajectory_count = how many trajectories are followed when dreaming
        - imagination_horizon = how many steps we go when dreaming
        - Compute element-wise SmoothL1 loss with reduction='none'
        - Sum losses across the imagination horizon for each trajectory
        - Mean across the batch / trajectories
        - Backprop, clip gradients by norm, step optimizer
        - Return scalar loss for logging
        """
        
        # see eq 5 in 2010.02193
        # Element-wise loss: shape (B, 1)
        element_loss = nn.functional.smooth_l1_loss(
            predicted_state_action_values, expected_state_action_values, reduction='none'
        )
        # Mean across trajectories, summed across horizon
        loss = element_loss.mean()

        self.optimizer.zero_grad()
        loss.backward() 
        max_grad_norm = 100.0
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        # apply gradient decent using the optimizer
        self.optimizer.step()

        return loss.item()


class TargetCriticNet(CriticBaseNet):
    def __init__(self, n_observations: int, actions_dimension: int):
        super().__init__(n_observations, actions_dimension)

    def soft_update(self, stabilising_net: StabilisingCriticNet):
        stabilising_net_state_dict = stabilising_net.state_dict()
        target_net_state_dict = self.state_dict()
        t = Constants.Behaviors.tau
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*t + target_net_state_dict[key]*(1-t)
        self.load_state_dict(target_net_state_dict)
