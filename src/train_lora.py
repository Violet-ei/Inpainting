from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .datasets import (
    DEFAULT_STYLE_PROMPT,
    CaptionImageDataset,
    collate_caption_batch,
    downsample_mask,
    random_inpaint_mask,
)
from .train_common import (
    add_lora_to_unet,
    import_training_stack,
    load_inpaint_components,
    save_lora_weights,
    save_training_metadata,
    set_seed,
    tokenize_captions,
    trainable_parameter_summary,
    weight_dtype_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ancient-painting LoRA for SD inpainting."
    )
    parser.add_argument("--pretrained_model_name_or_path", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--train_data_dir", default="data/lora_train")
    parser.add_argument("--output_dir", default="checkpoints/ancient_painting_lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--target_modules",
        default="to_q,to_k,to_v,to_out.0",
        help="Comma-separated UNet module names for LoRA.",
    )
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default_caption", default=DEFAULT_STYLE_PROMPT)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    )
    stack = import_training_stack()
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_inpaint_components(
        args.pretrained_model_name_or_path, args.revision
    )

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    add_lora_to_unet(
        unet,
        rank=args.rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
    )
    if args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    dataset = CaptionImageDataset(
        args.train_data_dir,
        resolution=args.resolution,
        default_caption=args.default_caption,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_caption_batch,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * steps_per_epoch
    lr_scheduler = stack["get_scheduler"](
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    weight_dtype = weight_dtype_for(args.mixed_precision)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()

    if accelerator.is_main_process:
        summary = trainable_parameter_summary(accelerator.unwrap_model(unet))
        print(
            f"Trainable UNet parameters: {summary['trainable']} / "
            f"{summary['total']} ({summary['percent']:.4f}%)"
        )

    global_step = 0
    progress = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="style-lora",
    )
    for _ in range(args.num_train_epochs):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device, dtype=weight_dtype
                )
                masks = random_inpaint_mask(
                    pixel_values.shape[0], args.resolution
                ).to(accelerator.device, dtype=weight_dtype)
                masked_images = pixel_values * (1.0 - masks)

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                masked_latents = vae.encode(masked_images).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                latent_masks = downsample_mask(
                    masks, latents.shape[-2], latents.shape[-1]
                )
                model_input = torch.cat(
                    [noisy_latents, latent_masks, masked_latents], dim=1
                )

                input_ids = tokenize_captions(
                    tokenizer, batch["captions"], accelerator.device
                )
                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(model_input, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                if (
                    args.checkpointing_steps > 0
                    and global_step % args.checkpointing_steps == 0
                    and accelerator.is_main_process
                ):
                    checkpoint_dir = Path(args.output_dir) / f"step-{global_step}"
                    save_lora_weights(checkpoint_dir, accelerator.unwrap_model(unet))
                if global_step >= max_train_steps:
                    break
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet).to(torch.float32)
        save_lora_weights(args.output_dir, unwrapped_unet)
        save_training_metadata(
            args.output_dir,
            {
                "type": "style_lora",
                "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
                "resolution": args.resolution,
                "rank": args.rank,
                "lora_alpha": args.lora_alpha,
                "target_modules": args.target_modules,
                "max_train_steps": max_train_steps,
                "train_data_dir": args.train_data_dir,
            },
        )
        print(f"Saved LoRA weights to {args.output_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
