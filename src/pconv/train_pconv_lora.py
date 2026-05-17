from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import PConvInpaintDataset, collate_pconv_batch
from .lora_conv import apply_lora_to_model
from .pconv_unet import PConvUNet
from .utils import (
    choose_device,
    composite_tensor,
    count_parameters,
    load_state_dict_flexible,
    masked_l1_loss,
    masked_psnr,
    save_checkpoint,
    set_batchnorm_eval,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PConv-LoRA on clean/damaged/mask pairs.")
    parser.add_argument("--train_root", default="data/train")
    parser.add_argument("--val_root", default="data/val")
    parser.add_argument("--pretrained_pconv", default=None)
    parser.add_argument("--output_dir", default="checkpoints/pconv/lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target", default="decoder", choices=["decoder", "encoder", "all"])
    parser.add_argument("--hole_weight", type=float, default=1.0)
    parser.add_argument("--valid_weight", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--strict_pretrained", action="store_true")
    parser.add_argument(
        "--allow_random_base",
        action="store_true",
        help="Allow training LoRA on a randomly initialized frozen PConv base. This is only useful for debugging.",
    )
    return parser.parse_args()


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = PConvInpaintDataset(args.train_root, resolution=args.resolution)
    val_dataset = PConvInpaintDataset(args.val_root, resolution=args.resolution)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pconv_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pconv_batch,
    )
    return train_loader, val_loader


def make_model(args: argparse.Namespace, device: torch.device) -> PConvUNet:
    model = PConvUNet()
    if args.pretrained_pconv:
        info = load_state_dict_flexible(model, args.pretrained_pconv, strict=args.strict_pretrained)
        print(f"Loaded pretrained PConv from {args.pretrained_pconv}")
        print(f"Missing keys: {len(info['missing_keys'])}, unexpected keys: {len(info['unexpected_keys'])}")
    elif not args.allow_random_base:
        raise ValueError(
            "--pretrained_pconv is required for PConv-LoRA training. "
            "Use --allow_random_base only for smoke-testing code paths."
        )
    else:
        print("Warning: training LoRA on a random frozen PConv base model.")
    result = apply_lora_to_model(
        model,
        rank=args.rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target=args.lora_target,
        freeze_base=True,
    )
    print(f"Injected LoRA Conv2d layers: {result.injected}")
    print(f"Trainable parameters: {result.trainable} / {result.total} ({100.0 * result.trainable / max(result.total, 1):.4f}%)")
    return model.to(device)


def compute_loss(pred: torch.Tensor, clean: torch.Tensor, damaged: torch.Tensor, damage_mask: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    valid_mask = 1.0 - damage_mask
    hole_loss = masked_l1_loss(pred, clean, damage_mask)
    valid_loss = masked_l1_loss(pred, damaged, valid_mask)
    return args.hole_weight * hole_loss + args.valid_weight * valid_loss


@torch.no_grad()
def validate(model: PConvUNet, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        clean = batch["clean"].to(device)
        damaged = batch["damaged"].to(device)
        damage_mask = batch["damage_mask"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        pred = model(damaged, valid_mask)
        composite = composite_tensor(pred, damaged, damage_mask)
        loss = compute_loss(pred, clean, damaged, damage_mask, args)
        psnr = masked_psnr(composite, clean, damage_mask)
        batch_size = clean.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_psnr += float(psnr.detach().cpu()) * batch_size
        count += batch_size
    return {"val_loss": total_loss / max(count, 1), "masked_psnr": total_psnr / max(count, 1)}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_loaders(args)
    model = make_model(args, device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_loss = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        set_batchnorm_eval(model)
        running = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            clean = batch["clean"].to(device)
            damaged = batch["damaged"].to(device)
            damage_mask = batch["damage_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred = model(damaged, valid_mask)
                loss = compute_loss(pred, clean, damaged, damage_mask, args)
            scaler.scale(loss).backward()
            if args.gradient_clip and args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = clean.shape[0]
            running += float(loss.detach().cpu()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=running / max(seen, 1))
        metrics = validate(model, val_loader, device, args)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(seen, 1)
        history.append(metrics)
        print(metrics)
        save_checkpoint(output_dir / "latest.pth", model, optimizer, epoch, best_loss, args)
        if metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            save_checkpoint(output_dir / "best.pth", model, optimizer, epoch, best_loss, args)
        write_json(output_dir / "history.json", {"history": history, "parameters": count_parameters(model)})


if __name__ == "__main__":
    main()
