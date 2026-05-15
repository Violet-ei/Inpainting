from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .auto_mask import AutoMaskConfig, generate_auto_mask
from .image_utils import (
    composite_prediction,
    ensure_same_size,
    load_mask,
    load_rgb,
    resize_to_multiple,
)


DEFAULT_PROMPT = (
    "ancient Chinese painting, restore damaged areas, preserve original style, "
    "natural texture, aged paper, traditional brush strokes"
)
DEFAULT_NEGATIVE_PROMPT = (
    "modern style, realistic photo, blurry, distorted face, extra details, "
    "oversaturated, cartoon, text, watermark"
)


@dataclass
class InpaintConfig:
    model_id: str = "runwayml/stable-diffusion-inpainting"
    lora_weights: Optional[str] = None
    device: Optional[str] = None
    dtype: str = "auto"
    max_size: Optional[int] = 768
    disable_safety_checker: bool = True


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


class AncientPaintingInpainter:
    def __init__(self, config: InpaintConfig) -> None:
        self.config = config
        self.pipe = None

    @classmethod
    def from_env(cls) -> "AncientPaintingInpainter":
        return cls(
            InpaintConfig(
                model_id=os.getenv(
                    "MODEL_ID", "runwayml/stable-diffusion-inpainting"
                ),
                lora_weights=os.getenv("LORA_WEIGHTS") or None,
                dtype=os.getenv("TORCH_DTYPE", "auto"),
                max_size=int(os.getenv("MAX_SIZE", "768")),
            )
        )

    def load(self):
        if self.pipe is not None:
            return self.pipe
        from diffusers import StableDiffusionInpaintPipeline

        dtype = _resolve_dtype(self.config.dtype)
        pipe_kwargs = {"torch_dtype": dtype}
        if self.config.disable_safety_checker:
            pipe_kwargs["safety_checker"] = None
            pipe_kwargs["requires_safety_checker"] = False

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.config.model_id, **pipe_kwargs
        )
        if self.config.lora_weights:
            pipe.load_lora_weights(self.config.lora_weights)

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        self.pipe = pipe
        return pipe

    def restore(
        self,
        image: Image.Image | str | Path,
        mask: Image.Image | str | Path | None = None,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 1.0,
        seed: Optional[int] = None,
        preserve_unmasked: bool = True,
        blur_mask: float = 0.0,
        auto_mask_config: AutoMaskConfig | None = None,
    ) -> Image.Image:
        damaged = load_rgb(image) if isinstance(image, (str, Path)) else image.convert("RGB")
        damaged = resize_to_multiple(
            damaged, multiple=8, max_size=self.config.max_size
        )
        if mask is None:
            mask_image = generate_auto_mask(damaged, auto_mask_config)
        else:
            mask_image = load_mask(mask) if isinstance(mask, (str, Path)) else mask.convert("L")
            mask_image = resize_to_multiple(
                mask_image,
                multiple=8,
                max_size=self.config.max_size,
                resample=Image.Resampling.NEAREST,
            )
        damaged, mask_image = ensure_same_size(damaged, mask_image)

        pipe = self.load()
        generator = None
        if seed is not None and seed >= 0:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=damaged,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        ).images[0]

        if preserve_unmasked:
            result = composite_prediction(damaged, result, mask_image, blur_radius=blur_mask)
        return result
