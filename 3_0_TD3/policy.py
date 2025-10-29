import math

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gymnasium import Env
import torch.optim as optim

from constants import *

from CriticNet import StabilisingCriticNet



class BasePolicyNet(nn.Module):
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

        self.steps_done = 0
        self.std_start = EXPLORATION_STD_START
        self.std_end = EXPLORATION_STD_END
        self.std_decay = EXPLORATION_STD_DECAY


    def forward(self, x: Tensor) -> Tensor:
        """Called with either a single observation of the enviroment to predict the best next action, or with batches during optimisation"""
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)
    
    def noisy_actions(self, state: Tensor, env: Env):
        std = self.std_end + (self.std_start - self.std_end) * math.exp(-1. * self.steps_done / self.std_decay)
        self.steps_done += 1

        action = self(state)
        # add noise
        action += torch.normal(mean=0.0, std=std, size=action.shape).to(DEVICE)
        # clamp to allowed values
        action = action.clamp(torch.tensor(env.action_space.low, device=DEVICE),
                                torch.tensor(env.action_space.high, device=DEVICE))
        return action
    

class StabilisingPolicyNet(BasePolicyNet):
    def __init__(self, n_observations: int, actions_dimention: int):
        super().__init__(n_observations, actions_dimention)
        self.optimizer = optim.AdamW(self.parameters(), lr=LEARNING_RATE, amsgrad=True)

        # by passing self.parameters, the optimiser knows which network is optimised

    def optimise(self, critic_net: StabilisingCriticNet, state_batch: Tensor, step: int):
        """update actor policy using the sampled policy gradient"""
        predicted_actions = self(state_batch) # μ(s)
        q_values = critic_net(torch.cat([state_batch, predicted_actions], dim=1)) # Q(s, μ(s))
        actor_loss = -q_values.mean() # We want to maximize Q, so minimize -Q

        self.optimizer.zero_grad()
        actor_loss.backward() # Computes ∇θμ Q(s, μ(s))
        torch.nn.utils.clip_grad_value_(self.parameters(), 100)
        self.optimizer.step()

        if step % 100 == 0:
            print(f"Actor Loss: {actor_loss.item()}")



class TargetPolicyNet(BasePolicyNet):
    def __init__(self, n_observations: int, actions_dimention: int):
        super().__init__(n_observations, actions_dimention)

    def soft_update(self, stabilising_net: StabilisingPolicyNet):
        stabilising_net_state_dict = stabilising_net.state_dict()
        target_net_state_dict = self.state_dict()
        for key in stabilising_net_state_dict:
            target_net_state_dict[key] = stabilising_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.load_state_dict(target_net_state_dict)