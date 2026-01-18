import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from torch import Tensor

from constants import Constants

# Assuming State class supports categorical variables (logits, probs, sample)
# You might need to update your State class definition as well.
# For this example, we'll assume a dummy structure for State:
class CategoricalState:
    def __init__(self, logits: Tensor, probs: Tensor, sample: Tensor):
        self.logits = logits
        self.probs = probs
        self.sample = sample

class CategoricalTransition(nn.Module):
    def __init__(self, input_size: int, *, activation=F.elu):
        """
        https://arxiv.org/pdf/1802.03006

        This model defines the transitions between states, where the distributions is
        modelled as a Gaussian distribution. 

        Args:
            input_size (int): The input size for the transition model, e.g the state and 
                action for SSM, the hidden RNN vector in RSSM.
            activation (callable): Activation function used in hidden layers (default: ELU).

        Attributes:
            transition_fc1 (nn.Linear): First layer of the transition network.
            transition_mean (nn.Linear): Layer to compute the mean of the transition distribution.
            transition_stddev (nn.Linear): Layer to compute the stddev of the transition distribution.
        """
        super().__init__()
        self.activation = activation

        self.num_latents = Constants.World.hidden_state_dimension
        self.num_categories = Constants.World.discrete_latent_classes
        
        self.state_size = self.num_latents * self.num_categories
        
        self.input_size = input_size

        hidden_size = Constants.Common.MLP_width
        self.transition_fc1 = nn.Linear(self.input_size, hidden_size)
        self.transition_fc2 = nn.Linear(hidden_size, hidden_size)
        self.transition_logits = nn.Linear(hidden_size, self.state_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, input: Tensor, mean_only: bool = False) -> CategoricalState:
        """
        Given the input, computes the prior distribution
        over the next state z_t using a categorical distribution, applying
        Straight-Through (ST) gradient estimation for the sample.
        """
        # --- Shared MLP Path ---
        hidden = self.activation(self.transition_fc1(input))
        hidden = self.activation(self.transition_fc2(hidden))
        
        # --- Logits Calculation ---
        # Output is (Batch_Size, N * C)
        flat_logits: Tensor = self.transition_logits(hidden)
        
        # Reshape logits to (Batch_Size, N, C)
        batch_shape = flat_logits.shape[:-1]
        logits = flat_logits.view(*batch_shape, self.num_latents, self.num_categories)
        
        # --- Probabilities and Distribution ---
        # Probabilities: Use softmax over the last dimension (categories C)
        probs: Tensor = F.softmax(logits, dim=-1)
        
        # Define the product of One-Hot Categorical distributions
        # NOTE: .detach() is REMOVED to allow gradient flow for ST estimation.
        dist = OneHotCategorical(probs=probs)  

        # --- Sampling (with Straight-Through Gradient) ---
        if mean_only:
            # If mean_only, the "sample" is the probability vector (Expected Value)
            sample: Tensor = probs
        else:
            # 1. Sample from the distribution (forward pass is a one-hot tensor)
            # Sample shape is (Batch_Size, N, C)
            one_hot_sample: Tensor = dist.sample()  
            
            # 2. Apply Straight-Through gradient:
            # In the backward pass, the gradient of the sample will be defined by the 
            # gradient of the probability tensor (probs).
            # This is done by adding the difference between probs and the sample, 
            # but DETACHING that difference so it doesn't affect the forward pass.
            sample: Tensor = one_hot_sample + (probs - one_hot_sample).detach()
            
            # The sampled one-hot vectors need to be flattened back to (Batch_Size, N * C)
            sample = sample.flatten(start_dim=-2)
            
        # The probability tensor (probs) also needs to be flattened (Batch_Size, N * C)
        flat_probs = probs.flatten(start_dim=-2)
        
        # The logits are also stored in flattened form
        return CategoricalState(flat_logits, flat_probs, sample)


class CategoricalRepresentation(CategoricalTransition):
    def __init__(self, input_size: int, obs_size: int, activation=F.elu):
        """
        https://arxiv.org/pdf/1802.03006

        This model defines the transitions between states, where the distributions is
        modelled as a Gaussian distribution. 

        Args:
            input_size (int): The input size for the transition model, e.g the state and 
                action for SSM, the hidden RNN vector in RSSM.
            activation (callable): Activation function used in hidden layers (default: ELU).
            min_stddev (float): Minimum standard deviation to ensure numerical stability.
        """
        super().__init__(input_size+obs_size, activation=activation)

    def forward(self, input, obs) -> CategoricalState:
            return super().forward(torch.concat([input, obs], dim=-1))
    