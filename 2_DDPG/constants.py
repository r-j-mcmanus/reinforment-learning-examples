import torch
import math

if torch.cuda.is_available() or torch.backends.mps.is_available():
    NUM_EPISODES = 1_000
else:
    NUM_EPISODES = 100

BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 3e-4 # for adam w optimisation (1711.05101)

EXPLORATION_STD_START = 1
EXPLORATION_STD_END = 0.01
EXPLORATION_STD_DECAY = NUM_EPISODES / math.log(EXPLORATION_STD_START / EXPLORATION_STD_END) 

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)