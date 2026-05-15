from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .image_utils import find_matching_file, list_images, load_mask, load_rgb, pil_to_tensor


DEFAULT_STYLE_PROMPT = (
    "ancient Chinese painting, traditional brush strokes, aged paper texture"
)


def _read_caption(caption_path: Optional[Path], default_caption: str) -> str:
    if caption_path is None or not caption_path.exists():
        return default_caption
    text = caption_path.read_text(encoding="utf-8").strip()
    return text or default_caption


def random_inpaint_mask(
    batch_size: int,
    resolution: int,
    min_rects: int = 1,
    max_rects: int = 4,
    scratch_count: int = 8,
) -> torch.Tensor:
    masks = torch.zeros(batch_size, 1, resolution, resolution, dtype=torch.float32)
    for b in range(batch_size):
        rects = random.randint(min_rects, max_rects)
        for _ in range(rects):
            rw = random.randint(max(8, resolution // 16), max(16, resolution // 3))
            rh = random.randint(max(8, resolution // 16), max(16, resolution // 3))
            x0 = random.randint(0, max(0, resolution - rw))
            y0 = random.randint(0, max(0, resolution - rh))
            masks[b, :, y0 : y0 + rh, x0 : x0 + rw] = 1.0
        for _ in range(scratch_count):
            x0 = random.randint(0, resolution - 1)
            y0 = random.randint(0, resolution - 1)
            x1 = random.randint(0, resolution - 1)
            y1 = random.randint(0, resolution - 1)
            thickness = random.randint(1, max(2, resolution // 80))
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            xs = torch.linspace(x0, x1, steps=steps).long().clamp(0, resolution - 1)
            ys = torch.linspace(y0, y1, steps=steps).long().clamp(0, resolution - 1)
            for dx in range(-thickness, thickness + 1):
                for dy in range(-thickness, thickness + 1):
                    xx = (xs + dx).clamp(0, resolution - 1)
                    yy = (ys + dy).clamp(0, resolution - 1)
                    masks[b, :, yy, xx] = 1.0
    return masks


class CaptionImageDataset(Dataset):
    def __init__(
        self,
        train_data_dir: str | Path,
        resolution: int = 512,
        default_caption: str = DEFAULT_STYLE_PROMPT,
    ) -> None:
        root = Path(train_data_dir)
        image_dir = root / "images" if (root / "images").exists() else root
        caption_dir = root / "captions"
        self.images = list_images(image_dir)
        if not self.images:
            raise FileNotFoundError(f"No training images found in {image_dir}")
        self.caption_dir = caption_dir
        self.default_caption = default_caption
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]
        caption_path = self.caption_dir / f"{image_path.stem}.txt"
        image = load_rgb(image_path)
        return {
            "pixel_values": pil_to_tensor(image, self.resolution, is_mask=False),
            "caption": _read_caption(caption_path, self.default_caption),
            "stem": image_path.stem,
        }


class PairedInpaintDataset(Dataset):
    def __init__(
        self,
        train_root: str | Path,
        resolution: int = 512,
        default_caption: str = DEFAULT_STYLE_PROMPT,
        captions_dir: str | Path | None = None,
        bootstrap_outputs_dir: str | Path | None = None,
        bootstrap_probability: float = 0.0,
    ) -> None:
        root = Path(train_root)
        self.clean_dir = root / "clean"
        self.damaged_dir = root / "damaged"
        self.mask_dir = root / "mask"
        self.captions_dir = Path(captions_dir) if captions_dir else None
        self.bootstrap_outputs_dir = (
            Path(bootstrap_outputs_dir) if bootstrap_outputs_dir else None
        )
        self.bootstrap_probability = bootstrap_probability
        self.default_caption = default_caption
        self.resolution = resolution
        self.samples = []

        for clean_path in list_images(self.clean_dir):
            damaged_path = find_matching_file(self.damaged_dir, clean_path.stem)
            mask_path = find_matching_file(self.mask_dir, clean_path.stem)
            if damaged_path is not None and mask_path is not None:
                self.samples.append((clean_path, damaged_path, mask_path))
        if not self.samples:
            raise FileNotFoundError(
                f"No paired samples found under {root}. Expected clean/damaged/mask."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _caption_for(self, stem: str) -> str:
        if self.captions_dir is None:
            return self.default_caption
        return _read_caption(self.captions_dir / f"{stem}.txt", self.default_caption)

    def _conditioning_path(self, damaged_path: Path) -> Path:
        if self.bootstrap_outputs_dir is None or self.bootstrap_probability <= 0:
            return damaged_path
        if random.random() > self.bootstrap_probability:
            return damaged_path
        generated = find_matching_file(self.bootstrap_outputs_dir, damaged_path.stem)
        return generated if generated is not None else damaged_path

    def __getitem__(self, index: int) -> dict:
        clean_path, damaged_path, mask_path = self.samples[index]
        cond_path = self._conditioning_path(damaged_path)
        clean = load_rgb(clean_path)
        conditioning = load_rgb(cond_path)
        mask = load_mask(mask_path)
        return {
            "pixel_values": pil_to_tensor(clean, self.resolution, is_mask=False),
            "conditioning_pixel_values": pil_to_tensor(
                conditioning, self.resolution, is_mask=False
            ),
            "mask_values": pil_to_tensor(mask, self.resolution, is_mask=True),
            "caption": self._caption_for(clean_path.stem),
            "stem": clean_path.stem,
        }


def collate_caption_batch(examples: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in examples]),
        "captions": [item["caption"] for item in examples],
        "stems": [item["stem"] for item in examples],
    }


def collate_paired_batch(examples: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in examples]),
        "conditioning_pixel_values": torch.stack(
            [item["conditioning_pixel_values"] for item in examples]
        ),
        "mask_values": torch.stack([item["mask_values"] for item in examples]),
        "captions": [item["caption"] for item in examples],
        "stems": [item["stem"] for item in examples],
    }


def downsample_mask(mask: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
    return F.interpolate(mask, size=(latent_height, latent_width), mode="nearest")
