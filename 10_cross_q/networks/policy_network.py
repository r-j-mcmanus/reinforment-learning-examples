import torch
import torch.nn as nn
from torch.distributions import Normal
from torch import Tensor

from constants import DEVICE
from networks.utility import LayerNormedReLU

class PolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_max: float,
        action_min: float,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Action scaling
        action_max = torch.tensor(action_max, dtype=torch.float32, device=DEVICE)
        action_min = torch.tensor(action_min, dtype=torch.float32, device=DEVICE)

        self.register_buffer("action_scale", (action_max - action_min) / 2.0)
        self.register_buffer("action_bias", (action_max + action_min) / 2.0)

        # Network
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            LayerNormedReLU(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            LayerNormedReLU(hidden_dim),
        ).to(DEVICE)

        self.mean_layer = nn.Linear(hidden_dim, action_dim, device=DEVICE)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim, device=DEVICE)

    def forward(self, obs):
        """
        Returns mean and log_std of the Gaussian (before tanh)
        """
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: Tensor, deterministic: bool) -> tuple[Tensor, Tensor | None]:
        """
        Samples an action using reparameterization trick
        Returns:
            action
            log_prob
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        if deterministic:
            z = mean
        else:
            normal = Normal(mean, std)
            z = normal.rsample()

        # Tanh squashing
        tanh_action = torch.tanh(z)

        # Rescale to environment action space
        action = tanh_action * self.action_scale + self.action_bias

        # Log-probability correction (important!)
        log_prob = None
        if not deterministic:
            log_prob = normal.log_prob(z)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            # Tanh correction
            log_prob -= torch.sum(
                torch.log(1 - tanh_action.pow(2) + 1e-6),
                dim=-1,
                keepdim=True,
            )

        return action, log_prob
