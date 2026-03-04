"""Projection head: residual linear layer that starts as identity transform."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class ProjectionHead(nn.Module):
    def __init__(self, dim: int = 1536):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # residual: x + linear(x); at init linear(x)=0 so output = normalize(x) = x
        return F.normalize(x + self.linear(x), dim=1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        return True

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
