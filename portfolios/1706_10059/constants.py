from typing import Any
import torch
import pydantic
from pydantic import BaseModel

class _Constants(BaseModel):
    BATCH_SIZE: int = 21
    EPISODE_COUNT: int = 3_000
    INPUT_PERIOD_LEN: int = 50
    MAX_STEPS: int = 30


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

CONSTANTS = _Constants()