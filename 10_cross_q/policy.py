import torch
from torch import Tensor
import torch.optim as optim

from networks.policy_network import PolicyNetwork

class Policy:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_max: float,
        action_min: float,
        *, 
        lr = 0.001
    ):
        self._actor = PolicyNetwork(observation_dim, action_dim, action_max, action_min)
        self._optimizer = optim.AdamW(self._actor.parameters(), lr=lr) # target networks update differently

    def get_action(self, obs: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor | None]:
        return self._actor.sample(obs, deterministic)
    
    def update(self, loss: Tensor):
        self._optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 1.0)
        self._optimizer.step()
        