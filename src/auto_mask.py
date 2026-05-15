from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from .image_utils import list_images, load_rgb, save_image


@dataclass
class AutoMaskConfig:
    sensitivity: float = 1.15
    dilate: int = 5
    blur: int = 3
    min_area: int = 24
    max_coverage: float = 0.35
    include_dark_stains: bool = True
    include_bright_scratches: bool = True
    include_edges: bool = True


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def _limit_coverage(mask: np.ndarray, score: np.ndarray, max_coverage: float) -> np.ndarray:
    coverage = float((mask > 0).mean())
    if coverage <= max_coverage or coverage == 0:
        return mask
    keep_pixels = max(1, int(mask.size * max_coverage))
    selected_scores = score[mask > 0]
    if selected_scores.size == 0:
        return mask
    threshold_index = max(0, selected_scores.size - keep_pixels)
    threshold = np.partition(selected_scores, threshold_index)[threshold_index]
    limited = np.where((mask > 0) & (score >= threshold), 255, 0).astype(np.uint8)
    return limited


def _postprocess(mask: np.ndarray, score: np.ndarray, config: AutoMaskConfig) -> np.ndarray:
    mask = _remove_small_components(mask, config.min_area)
    if config.dilate > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(config.dilate), _odd(config.dilate))
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if config.blur > 0:
        mask = cv2.GaussianBlur(mask, (_odd(config.blur), _odd(config.blur)), 0)
        mask = np.where(mask > 32, 255, 0).astype(np.uint8)
    mask = _limit_coverage(mask, score, config.max_coverage)
    return mask


def generate_auto_mask(
    image: Image.Image,
    config: AutoMaskConfig | None = None,
) -> Image.Image:
    """Estimate damaged regions from one input image.

    This is a heuristic mask generator for obvious cracks, stains, scratches,
    and missing areas. A hand-edited mask is still preferable when precision is
    important.
    """

    config = config or AutoMaskConfig()
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    height, width = gray.shape
    morph_size = max(9, (min(height, width) // 32) | 1)
    local_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_size, morph_size)
    )

    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, local_kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, local_kernel)
    background = cv2.GaussianBlur(gray, (_odd(morph_size), _odd(morph_size)), 0)
    local_delta = cv2.absdiff(gray, background)

    score = np.maximum.reduce([top_hat, black_hat, local_delta]).astype(np.float32)
    dynamic_threshold = float(score.mean() + config.sensitivity * score.std())
    anomaly = np.where(score > dynamic_threshold, 255, 0).astype(np.uint8)

    candidates = [anomaly]

    if config.include_bright_scratches:
        bright_threshold = max(215, int(np.percentile(value, 97)))
        bright = ((value >= bright_threshold) & (saturation < 95)).astype(np.uint8) * 255
        bright = cv2.bitwise_and(bright, np.where(top_hat > max(8, dynamic_threshold * 0.45), 255, 0).astype(np.uint8))
        candidates.append(bright)

    if config.include_dark_stains:
        dark_threshold = min(80, int(np.percentile(value, 5)))
        dark = ((value <= dark_threshold) & (black_hat > max(8, dynamic_threshold * 0.35))).astype(np.uint8) * 255
        candidates.append(dark)

    if config.include_edges:
        lower = max(20, int(np.percentile(gray, 12)))
        upper = min(220, int(np.percentile(gray, 88)))
        edges = cv2.Canny(gray, lower, upper)
        edges = cv2.bitwise_and(edges, np.where(local_delta > max(6, dynamic_threshold * 0.35), 255, 0).astype(np.uint8))
        candidates.append(edges)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for candidate in candidates:
        mask = cv2.bitwise_or(mask, candidate)

    mask = _postprocess(mask, score, config)
    return Image.fromarray(mask, mode="L")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an inpainting mask from damaged image(s).")
    parser.add_argument("--image", default=None, help="Single damaged image path.")
    parser.add_argument("--input_dir", default=None, help="Batch input directory.")
    parser.add_argument("--output", default=None, help="Single mask output path.")
    parser.add_argument("--output_dir", default="outputs/auto_mask")
    parser.add_argument("--sensitivity", type=float, default=1.15)
    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--min_area", type=int, default=24)
    parser.add_argument("--max_coverage", type=float, default=0.35)
    parser.add_argument("--no_dark_stains", action="store_true")
    parser.add_argument("--no_bright_scratches", action="store_true")
    parser.add_argument("--no_edges", action="store_true")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> AutoMaskConfig:
    return AutoMaskConfig(
        sensitivity=args.sensitivity,
        dilate=args.dilate,
        blur=args.blur,
        min_area=args.min_area,
        max_coverage=args.max_coverage,
        include_dark_stains=not args.no_dark_stains,
        include_bright_scratches=not args.no_bright_scratches,
        include_edges=not args.no_edges,
    )


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    if args.image:
        image = load_rgb(args.image)
        mask = generate_auto_mask(image, config)
        output = Path(args.output or Path(args.output_dir) / f"{Path(args.image).stem}.png")
        save_image(mask, output)
        print(f"Saved {output}")
        return

    if args.input_dir is None:
        raise ValueError("Use --image or --input_dir.")
    output_dir = Path(args.output_dir)
    count = 0
    for image_path in tqdm(list_images(args.input_dir), desc="auto-mask"):
        mask = generate_auto_mask(load_rgb(image_path), config)
        save_image(mask, output_dir / f"{image_path.stem}.png")
        count += 1
    print(f"Saved {count} masks to {output_dir}")


if __name__ == "__main__":
    main()
