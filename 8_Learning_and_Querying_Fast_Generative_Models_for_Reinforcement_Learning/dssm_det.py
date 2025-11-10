import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DSSM_DET(nn.Module):
    def __init__(self, state_size: int, obs_size: int, action_size: int, mean_only=False, activation=F.elu, hidden_size=32, num_layers=1):
        """
        https://arxiv.org/pdf/1802.03006

        Deterministic State Space Model (SSM) implemented in PyTorch.

        This model defines RNN based system where the transitions between states are encoded in the hidden
        states of the RNN.

        Args:
            state_size (int): Dimensionality of the latent state space.
            mean_only (bool): If True, sampling is deterministic using the mean only.
            activation (callable): Activation function used in hidden layers (default: ELU).

        Attributes:
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.obs_size = obs_size
        self.mean_only = mean_only
        self.activation = activation

        self._hidden_size = hidden_size
        self._num_layers = num_layers

        # Transition network
        self._rnn = nn.RNN(state_size + action_size, hidden_size, num_layers)

    def forward(self, obs: Tensor, action: Tensor, h0=None) -> Tensor:
        """
        Forward pass through the deterministic state space model.

        Args:
            obs (Tensor): Observation tensor of shape (seq_len, batch_size, obs_size).
            action (Tensor): Action tensor of shape (seq_len, batch_size, action_size).

        Returns:
            Tensor: The hidden states representing the latent states.
        """
        inputs = torch.cat([obs, action], dim=2)
        _, batch_size, _ = inputs.size()
        if h0 is None:
            h0 = torch.zeros(self._rnn.num_layers, batch_size, self._rnn.hidden_size)
        _, hn = self._rnn(inputs, h0)
        return hn