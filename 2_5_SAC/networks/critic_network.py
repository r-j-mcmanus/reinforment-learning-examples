import torch
import torch.nn as nn

from constants import DEVICE

class CriticNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.action_dim = action_dim

        # Network
        self.net = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(DEVICE)

    def forward(self, obs, action):
        """
        Returns mean and log_std of the Gaussian (before tanh)
        """
        x = torch.concat([obs, action], dim = -1)
        x = self.net(x)
        return x
