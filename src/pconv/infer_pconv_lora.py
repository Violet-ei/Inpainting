from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from ..image_utils import find_matching_file, list_images, load_mask, load_rgb, pil_to_tensor
from .lora_conv import apply_lora_to_model
from .pconv_unet import PConvUNet
from .utils import choose_device, composite_tensor, load_state_dict_flexible, save_tensor_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PConv or PConv-LoRA inpainting.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_dir", default="outputs/pconv_lora")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pretrained_pconv", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target", default="decoder", choices=["decoder", "encoder", "all"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_preserve_unmasked", action="store_true")
    return parser.parse_args()


def checkpoint_args(path: str | Path | None) -> dict:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("args"), dict):
        return payload["args"]
    return {}


def build_model(args: argparse.Namespace, device: torch.device) -> PConvUNet:
    if args.checkpoint is None and args.pretrained_pconv is None:
        raise ValueError("Provide --checkpoint for PConv-LoRA or --pretrained_pconv for baseline PConv inference.")
    ckpt_args = checkpoint_args(args.checkpoint)
    rank = int(ckpt_args.get("rank", args.rank))
    alpha = ckpt_args.get("lora_alpha", args.lora_alpha)
    dropout = float(ckpt_args.get("lora_dropout", args.lora_dropout))
    target = ckpt_args.get("lora_target", args.lora_target)
    model = PConvUNet()
    if args.pretrained_pconv:
        load_state_dict_flexible(model, args.pretrained_pconv, strict=False)
    if args.checkpoint:
        apply_lora_to_model(model, rank=rank, alpha=alpha, dropout=dropout, target=target, freeze_base=True)
        load_state_dict_flexible(model, args.checkpoint, strict=False)
    model.eval()
    return model.to(device)


def run_single(args: argparse.Namespace, model: PConvUNet, image_path: str | Path, mask_path: str | Path, output_path: str | Path) -> Path:
    device = next(model.parameters()).device
    damaged = pil_to_tensor(load_rgb(image_path), args.resolution, is_mask=False).unsqueeze(0).to(device)
    damage_mask = pil_to_tensor(load_mask(mask_path), args.resolution, is_mask=True).unsqueeze(0).float().to(device)
    valid_mask = 1.0 - damage_mask
    with torch.no_grad():
        pred = model(damaged, valid_mask)
        if args.no_preserve_unmasked:
            result = pred
        else:
            result = composite_tensor(pred, damaged, damage_mask)
    return save_tensor_image(result[0], output_path)


def run_batch(args: argparse.Namespace, model: PConvUNet) -> list[Path]:
    if args.input_dir is None:
        raise ValueError("--input_dir is required in batch mode.")
    if args.mask_dir is None:
        raise ValueError("--mask_dir is required for PConv batch inference.")
    output_dir = Path(args.output_dir)
    written = []
    for image_path in tqdm(list_images(args.input_dir), desc="pconv"):
        mask_path = find_matching_file(args.mask_dir, image_path.stem)
        if mask_path is None:
            continue
        output_path = output_dir / image_path.name
        written.append(run_single(args, model, image_path, mask_path, output_path))
    return written


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = build_model(args, device)
    if args.image:
        if args.mask is None:
            raise ValueError("--mask is required for single-image PConv inference.")
        output = Path(args.output or Path(args.output_dir) / Path(args.image).name)
        path = run_single(args, model, args.image, args.mask, output)
        print(f"Saved {path}")
    else:
        paths = run_batch(args, model)
        print(f"Saved {len(paths)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
