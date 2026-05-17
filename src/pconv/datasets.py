from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..image_utils import find_matching_file, list_images, load_mask, load_rgb, pil_to_tensor


class PConvInpaintDataset(Dataset):
    def __init__(self, root: str | Path, resolution: int = 512) -> None:
        self.root = Path(root)
        self.clean_dir = self.root / "clean"
        self.damaged_dir = self.root / "damaged"
        self.mask_dir = self.root / "mask"
        self.resolution = resolution
        self.samples = []
        for clean_path in list_images(self.clean_dir):
            damaged_path = find_matching_file(self.damaged_dir, clean_path.stem)
            mask_path = find_matching_file(self.mask_dir, clean_path.stem)
            if damaged_path is not None and mask_path is not None:
                self.samples.append((clean_path, damaged_path, mask_path))
        if not self.samples:
            raise FileNotFoundError(f"No paired samples found under {self.root}. Expected clean/damaged/mask.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        clean_path, damaged_path, mask_path = self.samples[index]
        clean = pil_to_tensor(load_rgb(clean_path), self.resolution, is_mask=False)
        damaged = pil_to_tensor(load_rgb(damaged_path), self.resolution, is_mask=False)
        damage_mask = pil_to_tensor(load_mask(mask_path), self.resolution, is_mask=True).float()
        valid_mask = 1.0 - damage_mask
        return {
            "clean": clean,
            "damaged": damaged,
            "damage_mask": damage_mask,
            "valid_mask": valid_mask,
            "stem": clean_path.stem,
        }


def collate_pconv_batch(examples: list[dict]) -> dict:
    return {
        "clean": torch.stack([item["clean"] for item in examples]),
        "damaged": torch.stack([item["damaged"] for item in examples]),
        "damage_mask": torch.stack([item["damage_mask"] for item in examples]),
        "valid_mask": torch.stack([item["valid_mask"] for item in examples]),
        "stems": [item["stem"] for item in examples],
    }
