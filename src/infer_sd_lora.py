from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .image_utils import find_matching_file, list_images, save_image
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
    parser.add_argument("--mask", default=None, help="Single mask path.")
    parser.add_argument("--input_dir", default=None, help="Directory of damaged images.")
    parser.add_argument("--mask_dir", default=None, help="Directory of masks for batch mode.")
    parser.add_argument("--output", default=None, help="Single output file.")
    parser.add_argument("--output_dir", default="outputs/sd_lora")
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
    return parser.parse_args()


def run_single(args: argparse.Namespace, inpainter: AncientPaintingInpainter) -> Path:
    if args.image is None or args.mask is None:
        raise ValueError("--image and --mask are required in single-image mode.")
    output = Path(args.output or Path(args.output_dir) / Path(args.image).name)
    result = inpainter.restore(
        args.image,
        args.mask,
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
    if args.input_dir is None or args.mask_dir is None:
        raise ValueError("--input_dir and --mask_dir are required in batch mode.")
    output_dir = Path(args.output_dir)
    written = []
    for image_path in tqdm(list_images(args.input_dir), desc="inpainting"):
        mask_path = find_matching_file(args.mask_dir, image_path.stem)
        if mask_path is None:
            continue
        local_args = argparse.Namespace(**vars(args))
        local_args.image = str(image_path)
        local_args.mask = str(mask_path)
        local_args.output = str(output_dir / image_path.name)
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
