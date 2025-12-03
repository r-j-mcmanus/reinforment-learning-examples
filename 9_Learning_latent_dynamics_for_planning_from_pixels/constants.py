from typing import Any
import torch


# 2010.02193 appendix D
class Constants():

    class World():
        dataset_size = 2**20
        latent_dataset_size = 2**20
        batch_size = 2**7
        sequence_length = 2**4
        latent_state_dimension = 2**5 # the dimension of the latent space we map to with the stochastic world model
        hidden_state_dimension = 2**5 # the dimension of the hidden space we map to with the deterministic world model
        discrete_latent_classes = 2**5
        epoch_count = 2**5 # 600
        kl_loss_scale = 0.1
        kl_balancing = 0.8
        world_model_learning_rate = 2e-4
        beta_growth_rate = 50 # used in initial training of the world model
        max_number_steps = 999

    class Behaviors():
        trajectory_count = 2**10 # how many trajectories are followed when dreaming
        imagination_horizon = 2**4  # how many steps we go when dreaming
        discount_factor = 0.995
        lambda_target_parameter = 0.95
        actor_gradient_mixing = 0 # ratio of reinforce to dynamic back-prorogation in the actor loss, discrete action --> 1, continuous --> 0
        actor_entropy_loss_scale = 1e-3 # for entropy regularisation in the actor loss
        actor_learning_rate = 4e-5
        critic_learning_rate = 1e-4
        slow_critic_update_interval = 100
        tau = 0.001 # used in the soft update
        latent_epoch_count = 3 # 2 ** 5 # how many epochs we do when latent learning

    class Common():
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