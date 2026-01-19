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
        *, 
        lr = 0.005, 
        gamma = 0.99
    ):
        self._gamma = gamma

        self._critic_1 = CriticNetwork(observation_dim, action_dim)
        self._critic_2 = CriticNetwork(observation_dim, action_dim)
        
        params = chain(self._critic_1.parameters(), self._critic_2.parameters()) 
        self._optimizer = optim.AdamW(params, lr=lr) # target networks update differently
        self._huber_loss = torch.nn.SmoothL1Loss()

    def update(self, observations: Tensor, observations_p1: Tensor, rewards: Tensor, actions: Tensor, actions_p1: Tensor, lop_pi_p1: Tensor, done: Tensor, alpha: Tensor, batch_size: int) -> float:
        # joined to pass through the network in a single call
        joined_observations = torch.concat([observations, observations_p1])
        joined_actions = torch.concat([actions, actions_p1])
        
        # pass through once for speed
        q_joined_1: Tensor = self._critic_1(joined_observations, joined_actions)
        q_joined_2: Tensor = self._critic_2(joined_observations, joined_actions)

        # get the q values from slicing
        q_1 = q_joined_1[:batch_size]
        q_1_p1 = q_joined_1[batch_size:].detach()

        q_2 = q_joined_2[:batch_size]
        q_2_p1 = q_joined_2[batch_size:].detach()

        q_p1 = torch.min(q_1_p1, q_2_p1) # min over double Q function

        backup = rewards + self._gamma * (1 - done) * (q_p1 - alpha * lop_pi_p1)

        loss = self._huber_loss(q_1, backup).mean() + self._huber_loss(q_2, backup).mean()

        self._optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(chain(self._critic_1.parameters(), self._critic_2.parameters()), 1.0)
        self._optimizer.step()

        return loss.item()

    def get_value(self, obs: Tensor, action: Tensor) -> Tensor:
        out_1 = self._critic_1(obs, action)
        out_2 = self._critic_2(obs, action)
        return torch.min(out_1, out_2) # min over double Q function
