import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim

from constants import *

class GnuActor(nn.Module):
    def __init__(self,
                 feature_dim: int, 
                 *,
                 hidden: int = 20,
                 lr = 0.05):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden,
            batch_first=True,
            device=DEVICE)

        self._hid_dim = hidden

        # +1 as we add the previous portfolio (minus cash values)
        self.fc_0 = torch.nn.Linear(hidden + 1, hidden, device=DEVICE)
        self.fc_1 = torch.nn.Linear(hidden, 1, device=DEVICE)

        self.cash_bias = nn.Parameter(torch.tensor([-10.0]))

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

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
        x = ((obs-1)*1000).reshape(-1, obs.shape[2], obs.shape[3]) # (batch * symbols, history, features)

        h_0 = torch.ones(1, x.shape[0], self._hid_dim, device=x.device)
        _, h = self.gru(x, h_0) # is set to batch_first = True

        # concat previous weights (exclude cash)
        w = torch.concat([h.squeeze(0), w_tm1[:,1:].reshape(-1).unsqueeze(dim=1)], dim=-1) # shape (symbol * batch, hidden2 + 1)
        
        # vote on asset distribution
        w = self.fc_0(w) # (symbol * batch, hidden+1)
        w = self.fc_1(w) # (symbol * batch, 1)
        
        # worried i should use x.reshape(obs.shape[3], obs.shape[1])
        # to check pass the same obs to all samples
        w = w.reshape(obs.shape[0], obs.shape[1]) # (batch, symbols)

        # prepend cash bias
        #x = torch.concat([self.cash_bias.expand(obs.shape[0], 1), x], dim=-1)

        # get asset weights
        # x = F.softmax(x - x.max(dim=-1).values, dim=-1) # more stable, try and fix
        w = F.softmax(w, dim=-1)
        w = torch.concat([torch.zeros(obs.shape[0], 1), w], dim=-1)

        # out output is a batch of portfolios, so we should have the same shape as the input batch
        assert w.shape == w_tm1.shape

        return w