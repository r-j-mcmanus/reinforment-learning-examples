import torch
from torch import Tensor
from torch import nn

from torchrl.modules.models.batchrenorm import BatchRenorm1d

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


class DropQBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.01,
    ):
        super().__init__()

        self._net = nn.Sequential(
            nn.Linear(in_features, out_features),   # weight layer?
            nn.Dropout(p=dropout_p),
            nn.LayerNorm(out_features),
            nn.ReLU()
        )

    def __call__(self, x: Tensor):
        return self._net(x)
    
    
class ResDropQBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),   # weight layer?
            nn.Dropout(p=dropout_p),
            nn.LayerNorm(out_features),
            nn.ReLU()
        )
        self.projection = torch.nn.Linear(in_features, out_features)

    def __call__(self, x: Tensor):
        return self.projection(x) + self.net(x)


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
    

class ResidualNormBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.LayerNorm(out_features),
            torch.nn.ReLU(),
        )
        self.projection = torch.nn.Linear(in_features, out_features)

    def forward(self, x: Tensor):
        return self.projection(x) + self.net(x)


class RenormBatchedBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            torch.nn.ReLU(),
            BatchRenorm1d(out_features,momentum=0.99)
        )

    def forward(self, x: Tensor):
        return self.net(x)
    