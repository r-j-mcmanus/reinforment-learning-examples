import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

import numpy as np

from constants import *

class Actor(nn.Module):
    def __init__(self, 
                 feature_dim: int = 3, 
                 kernel_size: int = 3, 
                 sample_len: int = 50, 
                 hidden1: int = 8,
                 hidden2: int = 4,
                 lr = 3e-5):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=feature_dim,
            out_channels=hidden1,
            kernel_size=kernel_size
        )

        self.conv2 = torch.nn.Conv1d(
            in_channels=hidden1,
            out_channels=hidden2,
            kernel_size=kernel_size
        )

        # Collapse time dimension safely
        self.pool = nn.AdaptiveAvgPool1d(1)

        # +1 as we add the previous portfolio (minus cash values)
        self.fc = torch.nn.Linear(hidden2 + 1, 1)

        self.cash_bias = nn.Parameter(torch.zeros(1))

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, obs: Tensor, w_tm1: Tensor) -> Tensor:
        """
        
        Arguments
        ---------
        obs - the observation of shape (Batch, History, Features, Symbols)
        w_tm1 - the previous portfolio of shape (Batch, Symbols + 1) where the extra symbol is cash help in the [0]th element of the dimension
        """
        
        assert isinstance(obs, Tensor)
        assert isinstance(w_tm1, Tensor)
        
        # format the data for the network
        #x = obs.permute(2,1,0)

        x = obs.permute(0, 3, 2, 1)          # (batch, symbols, features, history)
        x = x.reshape(-1, x.shape[2], x.shape[3]) # (batch * symbols, features, history)

        # Conv over time per asset
        # pass time series through the conv networks
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # squish all data along the time dimension
        x = self.pool(x).squeeze(-1)

        # concat previous weights (exclude cash)
        x = torch.concat([x, w_tm1[:,1:].reshape(-1).unsqueeze(dim=1)], dim=-1)
        
        # vote on asset distribution
        x = self.fc(x)
        
        # worried i should use x.reshape(obs.shape[3], obs.shape[1])
        # to check pass the same obs to all samples
        x = x.reshape(obs.shape[0], obs.shape[3]) # (batch, symbols)

        # prepend cash
        x = torch.concat([self.cash_bias.expand(obs.shape[0], 1), x], dim=-1)

        # get asset weights
        # x = F.softmax(x - x.max(dim=-1).values, dim=-1) # more stable, try and fix
        x = F.softmax(x, dim=-1)

        # out output is a batch of portfolios, so we should have the same shape as the input batch
        assert x.shape == w_tm1.shape

        return x