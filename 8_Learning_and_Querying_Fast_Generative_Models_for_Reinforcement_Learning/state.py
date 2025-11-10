from dataclasses import dataclass
from torch import Tensor

@dataclass
class State:
    """the distribution in the latent space an input gets mapped to"""
    mean: Tensor
    stddev: Tensor
    sample: Tensor