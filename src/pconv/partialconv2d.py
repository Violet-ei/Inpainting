from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        multi_channel: bool = True,
        return_mask: bool = True,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.multi_channel = multi_channel
        self.return_mask = return_mask
        mask_in_channels = in_channels if multi_channel else 1
        mask_out_channels = out_channels if multi_channel else 1
        self.slide_winsize = mask_in_channels * kernel_size[0] * kernel_size[1]
        self.register_buffer(
            "weight_mask_updater",
            torch.ones(mask_out_channels, mask_in_channels, kernel_size[0], kernel_size[1]),
            persistent=False,
        )

    def _prepare_mask(self, input: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return torch.ones_like(input)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.to(device=input.device, dtype=input.dtype)
        if self.multi_channel and mask.shape[1] == 1 and input.shape[1] != 1:
            mask = mask.expand(-1, input.shape[1], -1, -1)
        if not self.multi_channel and mask.shape[1] != 1:
            mask = mask[:, :1]
        if mask.shape[-2:] != input.shape[-2:]:
            mask = F.interpolate(mask, size=input.shape[-2:], mode="nearest")
        return mask

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        mask = self._prepare_mask(input, mask)
        with torch.no_grad():
            update_mask = F.conv2d(
                mask,
                self.weight_mask_updater.to(device=input.device, dtype=input.dtype),
                bias=None,
                stride=self.input_conv.stride,
                padding=self.input_conv.padding,
                dilation=self.input_conv.dilation,
                groups=1,
            )
            mask_ratio = self.slide_winsize / (update_mask + 1e-8)
            update_mask = torch.clamp(update_mask, 0.0, 1.0)
            mask_ratio = mask_ratio * update_mask
        raw_out = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            bias = self.input_conv.bias.view(1, -1, 1, 1)
            output = (raw_out - bias) * mask_ratio + bias
            output = output * update_mask
        else:
            output = raw_out * mask_ratio
        if self.return_mask:
            return output, update_mask
        return output
