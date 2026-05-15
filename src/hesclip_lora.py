from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LoRAHyperParams:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0


class LoRALinear(nn.Linear):
    """Small standalone LoRA linear layer, adapted from the HesClip pattern."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.weight.requires_grad = False
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            result = result + (
                self.lora_dropout(x)
                @ self.lora_A.transpose(0, 1)
                @ self.lora_B.transpose(0, 1)
            ) * self.scaling
        return result


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
