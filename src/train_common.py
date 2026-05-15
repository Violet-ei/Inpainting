from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weight_dtype_for(mixed_precision: str):
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def tokenize_captions(tokenizer, captions: list[str], device) -> torch.Tensor:
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids.to(device)


def import_training_stack():
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer
    from diffusers import UNet2DConditionModel

    return {
        "AutoencoderKL": AutoencoderKL,
        "DDPMScheduler": DDPMScheduler,
        "StableDiffusionInpaintPipeline": StableDiffusionInpaintPipeline,
        "UNet2DConditionModel": UNet2DConditionModel,
        "get_scheduler": get_scheduler,
        "LoraConfig": LoraConfig,
        "CLIPTextModel": CLIPTextModel,
        "CLIPTokenizer": CLIPTokenizer,
    }


def load_inpaint_components(model_id: str, revision: str | None = None):
    stack = import_training_stack()
    tokenizer = stack["CLIPTokenizer"].from_pretrained(
        model_id, subfolder="tokenizer", revision=revision
    )
    text_encoder = stack["CLIPTextModel"].from_pretrained(
        model_id, subfolder="text_encoder", revision=revision
    )
    vae = stack["AutoencoderKL"].from_pretrained(
        model_id, subfolder="vae", revision=revision
    )
    unet = stack["UNet2DConditionModel"].from_pretrained(
        model_id, subfolder="unet", revision=revision
    )
    noise_scheduler = stack["DDPMScheduler"].from_pretrained(
        model_id, subfolder="scheduler", revision=revision
    )
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def add_lora_to_unet(
    unet,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Iterable[str],
) -> None:
    stack = import_training_stack()
    config = stack["LoraConfig"](
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
        lora_dropout=dropout,
    )
    unet.add_adapter(config)


def trainable_parameter_summary(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(total, 1)
    return {"trainable": trainable, "total": total, "percent": pct}


def save_lora_weights(output_dir: str | Path, unet) -> None:
    from diffusers import StableDiffusionInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionInpaintPipeline.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=state_dict,
    )


def save_full_inpaint_pipeline(
    output_dir: str | Path,
    model_id: str,
    unet,
    text_encoder,
    vae,
    tokenizer,
    revision: str | None = None,
) -> None:
    stack = import_training_stack()
    pipe = stack["StableDiffusionInpaintPipeline"].from_pretrained(
        model_id,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        revision=revision,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.save_pretrained(output_dir)


def save_training_metadata(output_dir: str | Path, payload: dict) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
