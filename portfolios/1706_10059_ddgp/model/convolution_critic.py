import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from constants import *

class ConvolutionCriticNetwork(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        history_len: int,
        n_assets: int,   # symbols (excluding cash)
        *,
        kernel_size_1: int = 3,
        hidden1: int = 2,
        hidden2: int = 20,
        critic_hidden: int = 64,
        lr: float = 0.005,
    ):
        super().__init__()

        # -------- State encoder (same as actor) --------
        self.conv1 = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=hidden1,
            kernel_size=kernel_size_1,
            device=DEVICE
        )

        self._hidden2 = hidden2
        self.conv2 = nn.Conv1d(
            in_channels=hidden1,
            out_channels=hidden2,
            kernel_size=history_len - 2,
            device=DEVICE
        )

        self._leakyReLU = nn.LeakyReLU(0.01)

        # -------- Critic head --------
        # + (n_assets + 2) for past action as part of obs, and the current action we are getting the value of
        self.fc1 = nn.Linear((hidden2+2) * n_assets, critic_hidden, device=DEVICE)
        self.fc2 = nn.Linear(critic_hidden, critic_hidden, device=DEVICE)
        self.q_out = nn.Linear(critic_hidden, 1, device=DEVICE)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, obs: Tensor, prev_action: Tensor, action: Tensor) -> Tensor:
        """
        Parameters
        ----------
        obs : Tensor
            Shape (batch, symbols, history, features)
        prev_action : Tensor
            the previous action is part of the observation, passed separately as it is treated differently than the historic data
            Shape (batch, symbols + 1)  # includes cash
        action : Tensor
            Shape (batch, symbols + 1)  # includes cash
        """

        batch_size, n_symbols, history_len, n_feature = obs.shape

        # ---- Format observations ----
        x = obs.permute(0, 1, 3, 2) # (batch, symbols, features, history)
        x = x.reshape(-1, x.shape[2], x.shape[3]) # (batch * symbols, features, history)

        # ---- Conv encoder ----
        x = self._leakyReLU(self.conv1(x))
        x = self._leakyReLU(self.conv2(x)).squeeze(-1)  # (batch * symbols, hidden2)

        # ---- Concatenate actions ----
        x = x.reshape(batch_size, n_symbols, self._hidden2)
        x = torch.cat([x, prev_action[:, :-1].unsqueeze(-1), action[:, :-1].unsqueeze(-1)], dim=-1)
        x = x.flatten(1)   # (batch, symbols * (hidden2 + 2))

        # ---- Critic MLP ----
        x = self._leakyReLU(self.fc1(x))
        x = self._leakyReLU(self.fc2(x))
        q = self.q_out(x)

        return q.squeeze(-1)
