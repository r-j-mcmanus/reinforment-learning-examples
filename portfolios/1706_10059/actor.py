import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

import numpy as np

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

    def forward(self, x: Tensor, w_tm1: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        assert isinstance(w_tm1, Tensor)
        
        # format the data for the network
        x = x.permute(2,1,0)

        # pass time series through the conv networks
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # squish all data along the time dimension
        x = self.pool(x).squeeze(-1)

        # concat previous weights (exclude cash)
        x = torch.concat([x.squeeze(), w_tm1[1:].unsqueeze(dim=1)], dim=-1)
        
        # vote on asset distribution
        x = self.fc(x)
        
        # prepend cash
        x = torch.concat([self.cash_bias, x.squeeze()])

        # get asset weights
        x = F.softmax(x, dim=0)
        return x