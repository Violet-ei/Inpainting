from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from .image_utils import center_crop_square, list_images, save_image


def _random_color() -> tuple[int, int, int]:
    palette = [
        (245, 242, 230),
        (235, 225, 190),
        (80, 70, 55),
        (180, 160, 115),
        (255, 255, 255),
    ]
    return random.choice(palette)


def _add_scratches(image: Image.Image, mask: Image.Image, count: int) -> None:
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    width, height = image.size
    for _ in range(count):
        x0, y0 = random.randint(0, width - 1), random.randint(0, height - 1)
        x1 = int(np.clip(x0 + random.randint(-width // 2, width // 2), 0, width - 1))
        y1 = int(np.clip(y0 + random.randint(-height // 2, height // 2), 0, height - 1))
        thickness = random.randint(1, max(2, width // 120))
        color = _random_color()
        draw.line((x0, y0, x1, y1), fill=color, width=thickness)
        mask_draw.line((x0, y0, x1, y1), fill=255, width=thickness + 2)


def _add_stains(image: Image.Image, mask: Image.Image, count: int) -> None:
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    mask_draw = ImageDraw.Draw(mask)
    for _ in range(count):
        rx = random.randint(max(8, width // 30), max(12, width // 6))
        ry = random.randint(max(8, height // 30), max(12, height // 6))
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        bbox = (cx - rx, cy - ry, cx + rx, cy + ry)
        color = random.choice([(90, 65, 35, 65), (190, 150, 75, 85), (60, 55, 45, 55)])
        overlay_draw.ellipse(bbox, fill=color)
        mask_draw.ellipse(bbox, fill=255)
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    image.alpha_composite(overlay)


def _add_missing_regions(image: Image.Image, mask: Image.Image, count: int) -> None:
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    width, height = image.size
    for _ in range(count):
        rw = random.randint(max(12, width // 16), max(16, width // 4))
        rh = random.randint(max(12, height // 16), max(16, height // 4))
        x0 = random.randint(0, max(0, width - rw))
        y0 = random.randint(0, max(0, height - rh))
        if random.random() < 0.5:
            bbox = (x0, y0, x0 + rw, y0 + rh)
            draw.rectangle(bbox, fill=_random_color())
            mask_draw.rectangle(bbox, fill=255)
        else:
            points = [
                (x0 + random.randint(0, rw), y0 + random.randint(0, rh))
                for _ in range(random.randint(5, 9))
            ]
            draw.polygon(points, fill=_random_color())
            mask_draw.polygon(points, fill=255)


def synthesize_damage(clean: Image.Image) -> tuple[Image.Image, Image.Image]:
    damaged = clean.convert("RGBA")
    mask = Image.new("L", clean.size, 0)
    _add_scratches(damaged, mask, count=random.randint(4, 12))
    _add_stains(damaged, mask, count=random.randint(1, 4))
    _add_missing_regions(damaged, mask, count=random.randint(1, 3))
    mask = mask.filter(ImageFilter.MaxFilter(size=5))
    mask = mask.point(lambda v: 255 if v > 0 else 0)
    return damaged.convert("RGB"), mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create damaged images and masks.")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--output_root", default="data")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_root = Path(args.output_root) / args.split
    clean_out = output_root / "clean"
    damaged_out = output_root / "damaged"
    mask_out = output_root / "mask"
    for path in (clean_out, damaged_out, mask_out):
        path.mkdir(parents=True, exist_ok=True)

    images = list_images(args.clean_dir)
    if args.limit > 0:
        images = images[: args.limit]
    for image_path in tqdm(images, desc=f"damage-{args.split}"):
        clean = Image.open(image_path).convert("RGB")
        clean = center_crop_square(clean, args.size, Image.Resampling.LANCZOS)
        damaged, mask = synthesize_damage(clean)
        name = f"{image_path.stem}.png"
        save_image(clean, clean_out / name)
        save_image(damaged, damaged_out / name)
        save_image(mask, mask_out / name)
    print(f"Wrote {len(images)} samples to {output_root}")


if __name__ == "__main__":
    main()
