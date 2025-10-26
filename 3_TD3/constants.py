import torch

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LEARNING_RATE = 3e-4 # for adam w optimisation (1711.05101)
UPDATE_DELAY = 2 # a larger d would result in a larger benefit with respect to accumulating errors, for fair comparison, the critics are only trained once per time step, and training the actor for too few iterations would cripple learning.
POLICY_NOISE = 0.2
NOISE_CLAMP = 0.5
INVALID_ACTION_PENALTY = 0.1

EXPLORATION_STD = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)