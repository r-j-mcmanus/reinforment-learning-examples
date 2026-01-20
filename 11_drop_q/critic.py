from itertools import chain

import torch
from torch import Tensor
import torch.optim as optim

from networks.critic_network import CriticNetwork

class Critic:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        lr: float
    ):
        self._critic_1 = CriticNetwork(observation_dim, action_dim)
        self._critic_2 = CriticNetwork(observation_dim, action_dim)
        self._target_1 = CriticNetwork(observation_dim, action_dim)
        self._target_2 = CriticNetwork(observation_dim, action_dim)
        
        self._target_1.requires_grad_(False)
        self._target_2.requires_grad_(False)
        self._target_1.load_state_dict(self._critic_1.state_dict()) # insure initially equal        
        self._target_2.load_state_dict(self._critic_2.state_dict())

        params = chain(self._critic_1.parameters(), self._critic_2.parameters()) 
        self._optimizer = optim.AdamW(params, lr=lr) # target networks update differently
        self._huber_loss = torch.nn.SmoothL1Loss()

    def update(self, observations: Tensor, observations_p1: Tensor, rewards: Tensor, actions: Tensor, actions_p1: Tensor, lop_pi_p1: Tensor, done: Tensor, alpha: Tensor, batch_size: int) -> float:
        q_1: Tensor = self._critic_1(observations, actions)
        q_2: Tensor = self._critic_2(observations, actions)

        with torch.no_grad():
            t_1_p1: Tensor = self._target_1(observations_p1, actions_p1)
            t_2_p1: Tensor = self._target_2(observations_p1, actions_p1)

            t_p1 = torch.min(t_1_p1, t_2_p1).detach()

            backup = rewards + (1 - done) * 0.99 * (t_p1 - alpha * lop_pi_p1)

        loss_1 = self._huber_loss(q_1, backup).mean()
        loss_2 = self._huber_loss(q_2, backup).mean()
        loss: Tensor = loss_1 + loss_2

        self._optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(chain(self._critic_1.parameters(), self._critic_2.parameters()), 1.0)
        self._optimizer.step()

        return loss_1.item(), loss_2.item()

    def get_value(self, obs: Tensor, action: Tensor) -> Tensor:
        out_1 = self._critic_1(obs, action)
        out_2 = self._critic_2(obs, action)
        return (out_1 + out_2) / 2
        
    def soft_update(self, tau: float=0.005):
        self._soft_update(self._critic_1, self._target_1, tau)
        self._soft_update(self._critic_2, self._target_2, tau)

    def _soft_update(self, critic: torch.nn.Module, target: torch.nn.Module, tau: float):
        critic_net_state_dict = critic.state_dict()
        target_net_state_dict = target.state_dict()
        update_dict = {}
        for key in target_net_state_dict:
            update_dict[key] = critic_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target.load_state_dict(update_dict)
