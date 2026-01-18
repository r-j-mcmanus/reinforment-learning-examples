import torch
from pydantic import BaseModel

class _Constants(BaseModel):
    pass


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

CONSTANTS = _Constants()