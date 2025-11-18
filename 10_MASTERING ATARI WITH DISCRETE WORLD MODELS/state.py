from dataclasses import dataclass
from torch import Tensor

@dataclass
class State:
    """the distribution in the latent space an input gets mapped to"""
    mean: Tensor
    stddev: Tensor
    sample: Tensor

    def detach(self):
        """Returns a detached version of the state."""
        return State(
            mean=self.mean.detach(),
            stddev=self.stddev.detach(),
            sample=self.sample.detach()
        )