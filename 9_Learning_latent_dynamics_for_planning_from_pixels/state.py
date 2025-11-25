from dataclasses import dataclass
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

@dataclass
class State:
    """the distribution in the latent space an input gets mapped to.
    
    Attributes
    ----------
    mean : Tensor
    stddev : Tensor
    sample : Tensor
    """
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

    @property
    def shape(self) -> dict:
        """Returns a detached version of the state."""
        return {
            'mean':self.mean.shape,
            'stddev':self.stddev.shape,
            'sample':self.sample.shape
        }
    
    def sample_log_prob(self) -> Tensor:
        dist = MultivariateNormal(self.mean.squeeze(), torch.diag_embed(self.stddev.squeeze()))
        return dist.log_prob(self.sample)
    
    def entropy(self) -> Tensor:
        dist = MultivariateNormal(self.mean.squeeze(), torch.diag_embed(self.stddev.squeeze()))
        return dist.entropy()
    
    def view(self, *args):
        return State(
            mean=self.mean.view(*args) ,
            stddev=self.stddev.view(*args),
            sample=self.sample.view(*args) 
        )
        