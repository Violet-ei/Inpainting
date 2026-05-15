from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .auto_mask import AutoMaskConfig, generate_auto_mask
from .image_utils import find_matching_file, list_images, load_rgb, save_image
from .pipeline import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    AncientPaintingInpainter,
    InpaintConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SD inpainting with optional LoRA.")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--lora_weights", default=None)
    parser.add_argument("--image", default=None, help="Single damaged image path.")
    parser.add_argument("--mask", default=None, help="Optional single mask path.")
    parser.add_argument("--input_dir", default=None, help="Directory of damaged images.")
    parser.add_argument("--mask_dir", default=None, help="Optional directory of masks for batch mode.")
    parser.add_argument("--output", default=None, help="Single output file.")
    parser.add_argument("--output_dir", default="outputs/sd_lora")
    parser.add_argument("--auto_mask_output", default=None, help="Optional path to save the generated mask in single-image mode.")
    parser.add_argument("--auto_mask_output_dir", default=None, help="Optional directory to save generated masks in batch mode.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_size", type=int, default=768)
    parser.add_argument("--no_preserve_unmasked", action="store_true")
    parser.add_argument("--blur_mask", type=float, default=0.0)
    parser.add_argument("--auto_mask_sensitivity", type=float, default=1.15)
    parser.add_argument("--auto_mask_dilate", type=int, default=5)
    parser.add_argument("--auto_mask_blur", type=int, default=3)
    parser.add_argument("--auto_mask_min_area", type=int, default=24)
    parser.add_argument("--auto_mask_max_coverage", type=float, default=0.35)
    return parser.parse_args()


def auto_mask_config_from_args(args: argparse.Namespace) -> AutoMaskConfig:
    return AutoMaskConfig(
        sensitivity=args.auto_mask_sensitivity,
        dilate=args.auto_mask_dilate,
        blur=args.auto_mask_blur,
        min_area=args.auto_mask_min_area,
        max_coverage=args.auto_mask_max_coverage,
    )


def prepare_mask(args: argparse.Namespace, image_path: str | Path):
    if args.mask:
        return args.mask
    image = load_rgb(image_path)
    mask = generate_auto_mask(image, auto_mask_config_from_args(args))
    if args.auto_mask_output:
        save_image(mask, args.auto_mask_output)
    return mask


def run_single(args: argparse.Namespace, inpainter: AncientPaintingInpainter) -> Path:
    if args.image is None:
        raise ValueError("--image is required in single-image mode.")
    output = Path(args.output or Path(args.output_dir) / Path(args.image).name)
    mask = prepare_mask(args, args.image)
    result = inpainter.restore(
        args.image,
        mask,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
        preserve_unmasked=not args.no_preserve_unmasked,
        blur_mask=args.blur_mask,
    )
    return save_image(result, output)


def run_batch(args: argparse.Namespace, inpainter: AncientPaintingInpainter) -> list[Path]:
    if args.input_dir is None:
        raise ValueError("--input_dir is required in batch mode.")
    output_dir = Path(args.output_dir)
    written = []
    for image_path in tqdm(list_images(args.input_dir), desc="inpainting"):
        mask_path = (
            find_matching_file(args.mask_dir, image_path.stem)
            if args.mask_dir is not None
            else None
        )
        local_args = argparse.Namespace(**vars(args))
        local_args.image = str(image_path)
        local_args.mask = str(mask_path) if mask_path is not None else None
        local_args.output = str(output_dir / image_path.name)
        if mask_path is None and args.auto_mask_output_dir:
            local_args.auto_mask_output = str(
                Path(args.auto_mask_output_dir) / f"{image_path.stem}.png"
            )
        written.append(run_single(local_args, inpainter))
    return written


def main() -> None:
    args = parse_args()
    inpainter = AncientPaintingInpainter(
        InpaintConfig(
            model_id=args.model_id,
            lora_weights=args.lora_weights,
            device=args.device,
            dtype=args.dtype,
            max_size=args.max_size,
        )
    )
    if args.image:
        path = run_single(args, inpainter)
        print(f"Saved {path}")
    else:
        paths = run_batch(args, inpainter)
        print(f"Saved {len(paths)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
