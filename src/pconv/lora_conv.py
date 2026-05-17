from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class LoRAInjectResult:
    injected: int
    trainable: int
    total: int


class LoRAConv2d(nn.Module):
    def __init__(
        self,
        base_conv: nn.Conv2d,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if base_conv.groups != 1:
            raise ValueError("LoRAConv2d only supports Conv2d with groups=1.")
        self.base_conv = base_conv
        for param in self.base_conv.parameters():
            param.requires_grad = False
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scale = self.alpha / float(rank)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Conv2d(base_conv.in_channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(
            rank,
            base_conv.out_channels,
            kernel_size=base_conv.kernel_size,
            stride=base_conv.stride,
            padding=base_conv.padding,
            dilation=base_conv.dilation,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @property
    def stride(self):
        return self.base_conv.stride

    @property
    def padding(self):
        return self.base_conv.padding

    @property
    def dilation(self):
        return self.base_conv.dilation

    @property
    def bias(self):
        return self.base_conv.bias

    @property
    def in_channels(self):
        return self.base_conv.in_channels

    @property
    def out_channels(self):
        return self.base_conv.out_channels

    @property
    def kernel_size(self):
        return self.base_conv.kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_conv(x) + self.lora_up(self.dropout(self.lora_down(x))) * self.scale


def _target_keywords(target: str, custom_keywords: Iterable[str] | None) -> list[str]:
    if custom_keywords is not None:
        return [key for key in custom_keywords if key]
    if target == "all":
        return []
    if target == "encoder":
        return ["enc"]
    if target == "decoder":
        return ["dec"]
    return [target]


def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def should_inject(module_name: str, target: str, custom_keywords: Iterable[str] | None) -> bool:
    keywords = _target_keywords(target, custom_keywords)
    if not keywords:
        return True
    return any(keyword in module_name for keyword in keywords)


def inject_lora_into_conv2d(
    module: nn.Module,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    target: str = "decoder",
    custom_keywords: Iterable[str] | None = None,
    prefix: str = "",
) -> int:
    injected = 0
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, LoRAConv2d):
            continue
        if isinstance(child, nn.Conv2d) and should_inject(full_name, target, custom_keywords):
            setattr(module, name, LoRAConv2d(child, rank=rank, alpha=alpha, dropout=dropout))
            injected += 1
        else:
            injected += inject_lora_into_conv2d(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target=target,
                custom_keywords=custom_keywords,
                prefix=full_name,
            )
    return injected


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
    target: str = "decoder",
    custom_keywords: Iterable[str] | None = None,
    freeze_base: bool = True,
) -> LoRAInjectResult:
    if freeze_base:
        freeze_model(model)
    injected = inject_lora_into_conv2d(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target=target,
        custom_keywords=custom_keywords,
    )
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return LoRAInjectResult(injected=injected, trainable=trainable, total=total)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_down" in name or "lora_up" in name
    }
