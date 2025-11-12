import torch

BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001 # used in the soft update
M_TAU = 0.05 # used in the munchaisen value prediction
LEARNING_RATE = 3e-4 # for adam w optimisation (1711.05101)
UPDATE_DELAY = 2 # a larger d would result in a larger benefit with respect to accumulating errors, for fair comparison, the critics are only trained once per time step, and training the actor for too few iterations would cripple learning.
ALPHA = 0.9
MUNCHAUSEN_LOWER_BOUND = -1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)