from dataclasses import dataclass

import torch
import torch.nn as nn

_PROJECTOR_LINEAR_INIT_STD = 0.02


@dataclass
class MLPProjectorConfig:
    type: str = "mlp"
    hidden_dim: int | None = None
    # Number of MLP blocks. One block is: Linear -> SiLU -> Linear.
    num_layers: int = 1


def _init_projector_linear(module: nn.Linear) -> None:
    nn.init.normal_(module.weight, mean=0.0, std=_PROJECTOR_LINEAR_INIT_STD)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class _MLPBlock(nn.Module):
    """A single MLP block: Linear -> SiLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_projector_linear(self.fc1)
        _init_projector_linear(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MLPProjector(nn.Module):
    """Projector made of stacked MLP blocks.

    Notes:
        ``num_layers`` counts MLP blocks, not Linear layers.
        Each block is ``Linear -> SiLU -> Linear``.
        So ``num_layers=1`` means a standard single MLP block.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        blocks: list[nn.Module] = []
        block_in_dim = in_dim
        for layer_idx in range(num_layers):
            block_out_dim = out_dim if layer_idx == num_layers - 1 else hidden_dim
            blocks.append(_MLPBlock(block_in_dim, hidden_dim, block_out_dim))
            block_in_dim = block_out_dim
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)
