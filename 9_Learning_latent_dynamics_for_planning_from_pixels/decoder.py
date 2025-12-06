import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from constants import Constants


class Decoder(nn.Module):
    def __init__(self, input_size: int, output_size: int, activation=F.elu,
                 hidden_size: int = 32, output_activation=None):
        """
        Decoder network for reconstructing observations or predicting rewards.
        Example input for RSSM: [h_t, z_t].

        Args:
            input_size (int): Size of latent input.
            output_size (int): Size of output.
            activation (callable): Activation function (default: ELU).
            hidden_size (int): Size of hidden layers.
        """
        super().__init__()

        width = Constants.Common.MLP_width
        self.l1 = nn.Linear(input_size, width)
        self.l2 = nn.Linear(width, width)
        # Output mean and log_std
        self.mean = nn.Linear(width, output_size)
        self.log_std = nn.Linear(width, output_size)

        self.LOG_STD_MIN = -26
        self.LOG_STD_MAX = 2

        self.apply(lambda l: _orthogonal_init(l, gain=0.01))

    def forward(self, hidden_state: Tensor, latent_state: Tensor, 
                deterministic: bool = False, reparameterize: bool = True) -> tuple[Tensor, Normal]:
        x = torch.concat([hidden_state, latent_state], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        mean: Tensor = self.mean(x)
        log_std: Tensor = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        normal_dist = torch.distributions.Normal(mean, std)
        if deterministic:
            z = mean
        else:
            # Reparameterization trick
            z = normal_dist.rsample() if reparameterize else normal_dist.sample()

        return z, normal_dist

    def loss(self, h: Tensor, latent: Tensor, target: Tensor):
        reconstructed_observation, conditional_distribution = self.forward(h, latent)
        return - conditional_distribution.log_prob(target).mean()


def _orthogonal_init(layer, gain=1.0):
    """Orthogonal initialization maintains stable variance through deep networks and works 
    very well with tanh / ReLU activations."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)