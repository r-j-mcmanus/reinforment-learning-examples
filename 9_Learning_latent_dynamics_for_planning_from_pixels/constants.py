from typing import Any
import torch


class _UNMUTABLE:
    def __setattr__(self, name: str, value: Any) -> None:
        raise Exception(f'Cannot modify value of {name}')


# 2010.02193 appendix D
class Constants(_UNMUTABLE):

    class World(_UNMUTABLE):
        dataset_size = 2**20
        latent_dataset_size = 2**20
        batch_size = 2**5
        sequence_length = 2**4
        latent_state_dimension = 2**5 # the dimention of the latent space we map to with the stochastic world model
        hidden_state_dimension = 2**5 # the dimention of the hidden space we map to with the determanistic world model
        discreate_latent_classes = 2**5
        epoch_count = 2 # 2**5 # 600
        kl_loss_scale = 0.1
        kl_balancing = 0.8
        world_model_learning_rate = 2e-4
        beta_growth_rate = 50 # used in inital training of the world model

    class Behavior(_UNMUTABLE):
        trajectory_count = 2**10 # how many trajectories are followed when dreaming
        imagination_horizon = 2**4  # how many steps we go when dreaming
        discount_factor = 0.995
        lambda_target_parameter = 0.95
        actor_gradient_mixing = 1 # ratio of reinforce to dynamic backpropigation in the actor loss, discreat action --> 1, continuous --> 0
        actor_entopy_loss_scale = 1e-3 # for entropy regularisation in the actor loss
        actor_learning_rate = 4e-5
        critic_learning_rate = 1e-4
        slow_critic_update_interval = 100
        tau = 0.001 # used in the soft update

    class Common(_UNMUTABLE):
        policy_steps_per_gradient_step = 2**2
        gradient_clipping = 100
        adam_eps = 1e-5 # for adam w optimisation (1711.05101)
        weight_decay = 1e-6
        min_stddev = 1e-5
        MLP_width = 2**5 # the width of all hidden layers in FFN
        MLP_depth = 2 # TODO use this


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# asserts on size of values that have to be true: