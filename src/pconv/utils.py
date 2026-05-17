from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().float().clamp(-1.0, 1.0)
    image = (image + 1.0) * 0.5
    image = image.clamp(0.0, 1.0)
    array = image.permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_tensor_image(image: torch.Tensor, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image).save(path)
    return path


def composite_tensor(pred: torch.Tensor, damaged: torch.Tensor, damage_mask: torch.Tensor) -> torch.Tensor:
    if damage_mask.shape[1] == 1:
        damage_mask = damage_mask.expand(-1, damaged.shape[1], -1, -1)
    return pred * damage_mask + damaged * (1.0 - damage_mask)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = (pred + 1.0) * 0.5
    target = (target + 1.0) * 0.5
    if mask.shape[1] == 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp_min(1.0)
    mse = (((pred - target) ** 2) * mask).sum() / denom
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def load_state_dict_flexible(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = False) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        state_dict = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    incompatible = model.load_state_dict(cleaned, strict=strict)
    return {"missing_keys": list(incompatible.missing_keys), "unexpected_keys": list(incompatible.unexpected_keys)}


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_metric: float,
    args: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def set_batchnorm_eval(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()


def count_parameters(model: torch.nn.Module) -> dict[str, float]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {"trainable": trainable, "total": total, "percent": 100.0 * trainable / max(total, 1)}


def resize_mask_like(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if mask.shape[-2:] == x.shape[-2:]:
        return mask
    return F.interpolate(mask, size=x.shape[-2:], mode="nearest")
