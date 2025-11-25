import math

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gymnasium import Env
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

from constants import *

from critic import Critic
from state import State


class Policy:
    def __init__(self, actions_dimention: int):
        state_dimention = Constants.World.latent_state_dimension
        self.stabalising_net = StabilisingPolicyNet(state_dimention, actions_dimention)
        self.target_net = TargetPolicyNet(state_dimention, actions_dimention)

    def optimise(self, critic: Critic, L_targets: Tensor, dreamt_s: Tensor):
        self.stabalising_net.optimise(critic, L_targets, dreamt_s)

    def soft_update(self):
        self.target_net.soft_update(self.stabalising_net)

    def predict(self, state: Tensor) -> Tensor:
        return self.stabalising_net(state).sample


class ContinuousPolicyNet(nn.Module):
    def __init__(self, state_dimention: int, actions_dimention: int, *, mean_only = False, activation_function=F.elu):
        """A fully connected feed forward NN to model the policy function.

        arguments
        ---------
        n_observations: int
            The number of observations of the enviroment state that we pass to the model
        actions_dimention: int
            The dimentionality of the continuous action space
        """
        super().__init__()

        self.action_dim = actions_dimention

        self.layer_1 = nn.Linear(state_dimention, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_mean = nn.Linear(128, actions_dimention)
        self.layer_std = nn.Linear(128, actions_dimention)
        # self.layer_std = nn.Linear(128, actions_dimention * actions_dimention) would also need to make it posative definite!
        # can be easly done by doing self.layer_lower_triangular (L)
        # and std = L * L.transpose

        self.activation_function = activation_function

        self.steps_done = 0
        self.mean_only = mean_only

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x: Tensor) -> State:
        """Called with either a single observation of the enviroment to predict the best next action, or with batches during optimisation"""
        x = self.activation_function(self.layer_1(x))
        x = self.activation_function(self.layer_2(x))
        mean = self.layer_mean(x)
        stddev = F.softplus(self.layer_std(x)) + Constants.Common.min_stddev
        sample = mean if self.mean_only else MultivariateNormal(mean.squeeze(), torch.diag_embed(stddev.squeeze())).rsample()
        return State(mean, stddev, sample)    


class DiscretePolicyNet(nn.Module):
    def __init__(self, state_dimention: int, actions_dimention: int, *, activation_function=F.elu):
        """A fully connected feed forward NN to model the policy function.

        arguments
        ---------
        n_observations: int
            The number of observations of the enviroment state that we pass to the model
        actions_dimention: int
            The dimentionality of the continuous action space
        """
        super().__init__()

        self.layer_1 = nn.Linear(state_dimention, 128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, actions_dimention)

        self.activation_function = activation_function

        self.steps_done = 0
    
    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.activation_function(self.layer_1(x))
        x = self.activation_function(self.layer_2(x))
        logits = self.layer_3(x)  # raw scores for each action
        return Categorical(logits=logits).sample() # creates categorical distribution

    
class StabilisingPolicyNet(ContinuousPolicyNet):
    def __init__(self, state_dimention: int, actions_dimention: int):
        super().__init__(state_dimention, actions_dimention)
        self.optimizer = optim.AdamW(self.parameters(), lr=Constants.Behavior.actor_learning_rate, amsgrad=True)

        # by passing self.parameters, the optimiser knows which network is optimised

    def optimise(self, critic: Critic, L_targets: Tensor, dreamt_s: Tensor):
        """Update actor policy using the sampled policy gradient
        
        See eq 6 in 2010.02193

        the loss is the reinfocement + dynamics backpropigation + entropy regularization
        marginalized over the action probability and the world parameters

        arguments
        ---------
        critic: Critic - the critic netword for predicting the state value
        L_targets: Tensor - the Lambda proces state value from the target network
        dreamt_s: Tensor - states dreams sequences for the trajectory we train over
        """
        # dreamt_s.shape = ([sequence_length, trajectory_count, latent_state_dimension])
        # L_targets = ([sequence_length, trajectory_count, 1])
        
        # 0 for continuous actions
        # 1 for discrete actions
        rho = 0.5 # to test both 
        eta = Constants.Behavior.actor_entopy_loss_scale

        unflatten_shape = dreamt_s.shape[:-1]

        #flatten for passing though nets
        dreamt_s_flatten = dreamt_s.view(-1, dreamt_s.size(-1))
        predicted_actions_state_flatten: State = self(dreamt_s_flatten)
        q_values_flatten = critic.predicted(dreamt_s_flatten)
        
        #unflatten to get structure for sum and mean
        q_values = q_values_flatten.view(*unflatten_shape, -1)
        predicted_actions_state: State = predicted_actions_state_flatten.view(*unflatten_shape, -1) 

        if rho != 0:
            reinforce_weights = (L_targets - q_values).squeeze(-1).detach()
            log_prob = predicted_actions_state.sample_log_prob()
            reinforce_loss = - rho * (log_prob * reinforce_weights).sum(dim=0).mean()
        else:
            reinforce_loss = 0

        if rho != 1:
            dynamic_backpropagation = - (1 - rho) * L_targets.squeeze(-1).sum(dim=0).mean()
        else:
            dynamic_backpropagation = 0

        # TODO when moving to non-diag covar matrix, will need changing to generic det 
        entropy_regularizer = - eta * predicted_actions_state.entropy().sum(dim=0).mean()
        
        actor_loss = reinforce_loss + dynamic_backpropagation + entropy_regularizer
        self.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        self.optimizer.step()

class TargetPolicyNet(ContinuousPolicyNet):
    def __init__(self, state_dimention: int, actions_dimention: int):
        super().__init__(state_dimention, actions_dimention)

    def soft_update(self, stabilising_net: StabilisingPolicyNet):
        stabilising_net_state_dict = stabilising_net.state_dict()
        target_net_state_dict = self.state_dict()
        t = Constants.Behavior.tau
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*t + target_net_state_dict[key]*(1-t)
        self.load_state_dict(target_net_state_dict)
