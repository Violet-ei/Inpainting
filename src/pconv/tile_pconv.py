from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..auto_mask import AutoMaskConfig, generate_auto_mask
from .infer_pconv_lora import build_model
from .utils import choose_device, composite_tensor, tensor_to_pil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile-based long-image inference for PConv-LoRA.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--mask_mode", default="manual", choices=["manual", "auto"])
    parser.add_argument("--auto_mask_output", default=None)
    parser.add_argument("--output", required=True)

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained_pconv", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target", default="decoder", choices=["decoder", "encoder", "all"])
    parser.add_argument("--device", default=None)

    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--no_preserve_unmasked", action="store_true")

    parser.add_argument("--auto_mask_sensitivity", type=float, default=1.15)
    parser.add_argument("--auto_mask_dilate", type=int, default=5)
    parser.add_argument("--auto_mask_blur", type=int, default=3)
    parser.add_argument("--auto_mask_min_area", type=int, default=24)
    parser.add_argument("--auto_mask_max_coverage", type=float, default=0.35)
    return parser.parse_args()


def build_auto_mask_config(args: argparse.Namespace) -> AutoMaskConfig:
    return AutoMaskConfig(
        sensitivity=args.auto_mask_sensitivity,
        dilate=args.auto_mask_dilate,
        blur=args.auto_mask_blur,
        min_area=args.auto_mask_min_area,
        max_coverage=args.auto_mask_max_coverage,
    )


def prepare_mask(args: argparse.Namespace, image: Image.Image) -> Image.Image:
    if args.mask_mode == "manual":
        if args.mask is None:
            raise ValueError("--mask is required when --mask_mode manual.")
        mask = Image.open(args.mask).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return mask

    mask = generate_auto_mask(image, build_auto_mask_config(args)).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    if args.auto_mask_output:
        out_path = Path(args.auto_mask_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(out_path)

    return mask


def make_positions(length: int, patch: int, overlap: int) -> tuple[list[int], int]:
    if length <= patch:
        return [0], length
    stride = patch - overlap
    if stride <= 0:
        raise ValueError("--overlap must be smaller than --patch_size.")
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions, patch


def feather_window(h: int, w: int, overlap: int, x: int, y: int, image_w: int, image_h: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w]

    left = xx.astype(np.float32) if x > 0 else np.full((h, w), overlap, dtype=np.float32)
    right = (w - 1 - xx).astype(np.float32) if x + w < image_w else np.full((h, w), overlap, dtype=np.float32)
    top = yy.astype(np.float32) if y > 0 else np.full((h, w), overlap, dtype=np.float32)
    bottom = (h - 1 - yy).astype(np.float32) if y + h < image_h else np.full((h, w), overlap, dtype=np.float32)

    dist = np.minimum(np.minimum(left, right), np.minimum(top, bottom))
    return np.clip(dist / max(overlap, 1), 0.0, 1.0).astype(np.float32)


def image_to_tensor(image: Image.Image, resolution: int) -> torch.Tensor:
    image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    array = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def mask_to_tensor(mask: Image.Image, resolution: int) -> torch.Tensor:
    mask = mask.convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)
    array = np.asarray(mask).astype(np.float32) / 255.0
    array = (array >= 0.5).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


@torch.no_grad()
def run_patch(model: torch.nn.Module, patch_img: Image.Image, patch_mask: Image.Image, args: argparse.Namespace) -> Image.Image:
    device = next(model.parameters()).device

    damaged = image_to_tensor(patch_img, args.resolution).unsqueeze(0).to(device)
    damage_mask = mask_to_tensor(patch_mask, args.resolution).unsqueeze(0).to(device)
    valid_mask = 1.0 - damage_mask

    pred = model(damaged, valid_mask)
    if args.no_preserve_unmasked:
        result = pred
    else:
        result = composite_tensor(pred, damaged, damage_mask)

    result_img = tensor_to_pil(result[0])
    if result_img.size != patch_img.size:
        result_img = result_img.resize(patch_img.size, Image.Resampling.BICUBIC)
    return result_img


def main() -> None:
    args = parse_args()

    image = Image.open(args.image).convert("RGB")
    mask = prepare_mask(args, image)

    image_w, image_h = image.size
    xs, patch_w = make_positions(image_w, args.patch_size, args.overlap)
    ys, patch_h = make_positions(image_h, args.patch_size, args.overlap)

    device = choose_device(args.device)
    model = build_model(args, device)

    accum = np.zeros((image_h, image_w, 3), dtype=np.float32)
    weight = np.zeros((image_h, image_w, 1), dtype=np.float32)

    total = len(xs) * len(ys)
    index = 0

    for y in ys:
        for x in xs:
            index += 1
            box = (x, y, x + patch_w, y + patch_h)
            patch_img = image.crop(box)
            patch_mask = mask.crop(box)

            patch_mask_np = np.asarray(patch_mask).astype(np.float32)
            win = feather_window(patch_h, patch_w, args.overlap, x, y, image_w, image_h)[..., None]

            if patch_mask_np.max() == 0:
                patch_out_np = np.asarray(patch_img).astype(np.float32)
            else:
                print(f"[{index}/{total}] inpaint tile x={x}, y={y}, size={patch_w}x{patch_h}")
                result = run_patch(model, patch_img, patch_mask, args)
                result_np = np.asarray(result).astype(np.float32)
                orig_np = np.asarray(patch_img).astype(np.float32)
                alpha = (patch_mask_np / 255.0)[..., None]
                alpha = np.clip(alpha, 0.0, 1.0)
                patch_out_np = orig_np * (1.0 - alpha) + result_np * alpha

            accum[y:y + patch_h, x:x + patch_w] += patch_out_np * win
            weight[y:y + patch_h, x:x + patch_w] += win

    output_np = accum / np.clip(weight, 1e-6, None)
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    out = Image.fromarray(output_np)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
