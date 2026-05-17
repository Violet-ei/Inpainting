from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .partialconv2d import PartialConv2d


class PConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batch_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.pconv = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            multi_channel=True,
            return_mask=True,
        )
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, mask = self.pconv(x, mask)
        x = self.norm(x)
        x = self.activation(x)
        return x, mask


class PConvUNet(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = PConvBlock(input_channels, 64, 7, 2, 3, batch_norm=False, activation="relu")
        self.enc2 = PConvBlock(64, 128, 5, 2, 2, activation="relu")
        self.enc3 = PConvBlock(128, 256, 5, 2, 2, activation="relu")
        self.enc4 = PConvBlock(256, 512, 3, 2, 1, activation="relu")
        self.enc5 = PConvBlock(512, 512, 3, 2, 1, activation="relu")
        self.enc6 = PConvBlock(512, 512, 3, 2, 1, activation="relu")
        self.enc7 = PConvBlock(512, 512, 3, 2, 1, activation="relu")
        self.enc8 = PConvBlock(512, 512, 3, 2, 1, activation="relu")
        self.dec8 = PConvBlock(1024, 512, 3, 1, 1, activation="leaky_relu")
        self.dec7 = PConvBlock(1024, 512, 3, 1, 1, activation="leaky_relu")
        self.dec6 = PConvBlock(1024, 512, 3, 1, 1, activation="leaky_relu")
        self.dec5 = PConvBlock(1024, 512, 3, 1, 1, activation="leaky_relu")
        self.dec4 = PConvBlock(768, 256, 3, 1, 1, activation="leaky_relu")
        self.dec3 = PConvBlock(384, 128, 3, 1, 1, activation="leaky_relu")
        self.dec2 = PConvBlock(192, 64, 3, 1, 1, activation="leaky_relu")
        self.dec1 = PConvBlock(64 + input_channels, output_channels, 3, 1, 1, batch_norm=False, activation="tanh")

    @staticmethod
    def _expand_input_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        return mask.to(device=x.device, dtype=x.dtype)

    @staticmethod
    def _upsample_pair(x: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = target.shape[-2:]
        x = F.interpolate(x, size=size, mode="nearest")
        mask = F.interpolate(mask, size=size, mode="nearest")
        return x, mask

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        m0 = self._expand_input_mask(x, valid_mask)
        e1, m1 = self.enc1(x, m0)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)
        e6, m6 = self.enc6(e5, m5)
        e7, m7 = self.enc7(e6, m6)
        e8, m8 = self.enc8(e7, m7)
        d, dm = self._upsample_pair(e8, m8, e7)
        d, dm = self.dec8(torch.cat([d, e7], dim=1), torch.cat([dm, m7], dim=1))
        d, dm = self._upsample_pair(d, dm, e6)
        d, dm = self.dec7(torch.cat([d, e6], dim=1), torch.cat([dm, m6], dim=1))
        d, dm = self._upsample_pair(d, dm, e5)
        d, dm = self.dec6(torch.cat([d, e5], dim=1), torch.cat([dm, m5], dim=1))
        d, dm = self._upsample_pair(d, dm, e4)
        d, dm = self.dec5(torch.cat([d, e4], dim=1), torch.cat([dm, m4], dim=1))
        d, dm = self._upsample_pair(d, dm, e3)
        d, dm = self.dec4(torch.cat([d, e3], dim=1), torch.cat([dm, m3], dim=1))
        d, dm = self._upsample_pair(d, dm, e2)
        d, dm = self.dec3(torch.cat([d, e2], dim=1), torch.cat([dm, m2], dim=1))
        d, dm = self._upsample_pair(d, dm, e1)
        d, dm = self.dec2(torch.cat([d, e1], dim=1), torch.cat([dm, m1], dim=1))
        d, dm = self._upsample_pair(d, dm, x)
        d, dm = self.dec1(torch.cat([d, x], dim=1), torch.cat([dm, m0], dim=1))
        return d
