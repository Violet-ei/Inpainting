from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_matching_file(root: str | Path, stem: str) -> Optional[Path]:
    root = Path(root)
    for suffix in sorted(IMAGE_EXTENSIONS):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = [p for p in root.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(matches)[0] if matches else None


def load_rgb(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_mask(path: str | Path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    mask = Image.open(path)
    mask = ImageOps.exif_transpose(mask).convert("L")
    if size is not None:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(mask)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def ensure_same_size(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    return image, mask


def resize_to_multiple(
    image: Image.Image,
    multiple: int = 8,
    max_size: Optional[int] = None,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    width, height = image.size
    if max_size is not None and max(width, height) > max_size:
        scale = max_size / float(max(width, height))
        width = int(round(width * scale))
        height = int(round(height * scale))
    width = max(multiple, (width // multiple) * multiple)
    height = max(multiple, (height // multiple) * multiple)
    if image.size == (width, height):
        return image
    return image.resize((width, height), resample)


def center_crop_square(image: Image.Image, size: int, resample: Image.Resampling) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), resample)


def pil_to_tensor(image: Image.Image, size: int, is_mask: bool = False):
    import torch

    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    image = center_crop_square(image, size, resample)
    array = np.asarray(image)
    if is_mask:
        array = (array > 127).astype(np.float32)[None, ...]
        return torch.from_numpy(array)
    array = array.astype(np.float32) / 127.5 - 1.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def tensor_mask_to_pil(mask) -> Image.Image:
    array = mask.detach().cpu().float().numpy()
    if array.ndim == 3:
        array = array[0]
    array = (array > 0.5).astype(np.uint8) * 255
    return Image.fromarray(array, mode="L")


def composite_prediction(
    damaged: Image.Image,
    prediction: Image.Image,
    mask: Image.Image,
    blur_radius: float = 0.0,
) -> Image.Image:
    from PIL import ImageFilter

    damaged, mask = ensure_same_size(damaged.convert("RGB"), mask.convert("L"))
    prediction = prediction.convert("RGB").resize(damaged.size, Image.Resampling.LANCZOS)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return Image.composite(prediction, damaged, mask)


def save_image(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def common_stems(*roots: str | Path) -> list[str]:
    stem_sets = []
    for root in roots:
        stem_sets.append({p.stem for p in list_images(root)})
    if not stem_sets:
        return []
    return sorted(set.intersection(*stem_sets))


def copy_placeholder_dirs(paths: Iterable[str | Path]) -> None:
    for path in paths:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")
