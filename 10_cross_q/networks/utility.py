import torch
from torch import Tensor


class LayerNormedReLU(torch.nn.Module):
    def __init__(
        self,
        width: int
    ):
        super().__init__()

        self._net = torch.nn.Sequential(
         torch.nn.LayerNorm(width),
         torch.nn.ReLU()
        )

    def __call__(self, x: Tensor):
        return self._net(x)


class BatchedBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            #torch.nn.Dropout(dropout_p),
            torch.nn.LayerNorm(out_features),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(out_features)
        )

    def forward(self, x: Tensor):
        return self.net(x)
    

class ResidualBatchedBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        dropout_p: float = 0.1
    ):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            torch.nn.Dropout(dropout_p),
            torch.nn.LayerNorm(in_features),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(in_features)
        )

    def forward(self, x: Tensor):
        return x + self.net(x)
