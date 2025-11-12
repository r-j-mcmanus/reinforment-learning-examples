import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Decoder(nn.Module):
    def __init__(self, input_size: int, obs_size: int, activation=F.elu,
                 hidden_size: int = 32, output_activation=None):
        """
        Decoder network for reconstructing observations or predicting rewards.
        Example input for RSSM: [h_t, z_t].

        Args:
            input_size (int): Size of latent input.
            obs_size (int): Size of output.
            activation (callable): Activation function (default: ELU).
            hidden_size (int): Size of hidden layers.
        """
        super().__init__()
        self.input_size = input_size
        self.obs_size = obs_size
        self.activation = activation

        self._output_activation = output_activation

        # Decoder
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, obs_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input: Tensor) -> Tensor:
        x = self.activation(self.fc1(input))
        x = self.activation(self.fc2(x))
        x = self.fc_out(x)
        if self._output_activation is not None:
            x = self._output_activation(x)
        return x