import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from constants import *

class ConvolutionActor(nn.Module):
    def __init__(self,
                 feature_dim: int, 
                 *,
                 kernel_size_1: int = 3, 
                 kernel_size_2: int = 3, 
                 hidden1: int = 2,
                 hidden2: int = 20,
                 lr = 0.005):
        super().__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=feature_dim,
            out_channels=hidden1,
            kernel_size=kernel_size_1, 
            device=DEVICE
        )

        self.conv2 = torch.nn.Conv1d(
            in_channels=hidden1,
            out_channels=hidden2,
            kernel_size=kernel_size_2, 
            device=DEVICE
        )

        # Collapse time dimension safely
        self.pool = nn.AdaptiveAvgPool1d(1, device=DEVICE)

        # +1 as we add the previous portfolio (minus cash values)
        self.fc_0 = torch.nn.Linear(hidden2 + 1, hidden2, device=DEVICE)
        self.fc_1 = torch.nn.Linear(hidden2, 1, device=DEVICE)

        self.cash_bias = nn.Parameter(torch.tensor([-10.0]), device=DEVICE)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, device=DEVICE)
        self._leakyReLU = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False, device=DEVICE)

    def forward(self, obs: Tensor, w_tm1: Tensor) -> Tensor:
        """
        
        Arguments
        ---------
        obs - the observation of shape (Batch, Symbols, History, Features)
        w_tm1 - the detached previous portfolio of shape (Batch, Symbols + 1) where the extra symbol is cash help in the [0]th element of the dimension
        """
        
        assert isinstance(obs, Tensor)
        assert isinstance(w_tm1, Tensor)
        
        # format the data for the network
        x = obs.permute(0, 1, 3, 2)          # (batch, symbols, features, history)
        x = x.reshape(-1, x.shape[2], x.shape[3]) # (batch * symbols, features, history)

        # Conv over time per asset
        # pass time series through the conv networks
        x = self._leakyReLU(self.conv1(x)) # shape (symbol * batch, hidden1, history - 2)
        x = self.conv2(x) # shape (symbol * batch, hidden2, history - 4)

        # squish all data along the time dimension
        x = self._leakyReLU(self.pool(x).squeeze(-1)) # shape (symbol * batch, hidden2)

        # concat previous weights (exclude cash)
        x = torch.concat([x, w_tm1[:,1:].reshape(-1).unsqueeze(dim=1)], dim=-1) # shape (symbol * batch, hidden2 + 1)
        
        # vote on asset distribution
        w = self.fc_0(w) # (symbol * batch, hidden+1)
        w = self.fc_1(w) # (symbol * batch, 1)
        
        # worried i should use x.reshape(obs.shape[3], obs.shape[1])
        # to check pass the same obs to all samples
        x = x.reshape(obs.shape[0], obs.shape[1]) # (batch, symbols)

        # prepend cash bias
        x = torch.concat([self.cash_bias.expand(obs.shape[0], 1), x], dim=-1)

        # get asset weights
        # x = F.softmax(x - x.max(dim=-1).values, dim=-1) # more stable, try and fix
        x = F.softmax(x, dim=-1)
        #x = torch.concat([torch.zeros(obs.shape[0], 1), x], dim=-1)

        # out output is a batch of portfolios, so we should have the same shape as the input batch
        assert x.shape == w_tm1.shape

        return x