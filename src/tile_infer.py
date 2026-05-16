from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from src.auto_mask import AutoMaskConfig, generate_auto_mask
from src.pipeline import AncientPaintingInpainter, InpaintConfig, DEFAULT_PROMPT, DEFAULT_NEGATIVE_PROMPT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--lora_weights", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--mask_mode", default="manual", choices=["manual", "auto"])
    parser.add_argument("--auto_mask_output", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="fp16", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_size", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--blur_mask", type=float, default=0.0)
    parser.add_argument("--auto_mask_sensitivity", type=float, default=1.15)
    parser.add_argument("--auto_mask_dilate", type=int, default=5)
    parser.add_argument("--auto_mask_blur", type=int, default=3)
    parser.add_argument("--auto_mask_min_area", type=int, default=24)
    parser.add_argument("--auto_mask_max_coverage", type=float, default=0.35)
    return parser.parse_args()

def build_auto_mask_config(args):
    return AutoMaskConfig(
        sensitivity=args.auto_mask_sensitivity,
        dilate=args.auto_mask_dilate,
        blur=args.auto_mask_blur,
        min_area=args.auto_mask_min_area,
        max_coverage=args.auto_mask_max_coverage,
    )

def make_positions(length, patch, overlap):
    if length <= patch:
        return [0], length
    stride = patch - overlap
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions, patch

def feather_window(h, w, overlap):
    if overlap <= 0:
        return np.ones((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    left = xx
    right = w - 1 - xx
    top = yy
    bottom = h - 1 - yy
    dist = np.minimum(np.minimum(left, right), np.minimum(top, bottom)).astype(np.float32)
    win = np.clip(dist / max(overlap, 1), 0.0, 1.0)
    return win

def prepare_mask(args, image):
    if args.mask_mode == "manual" and args.mask:
        mask = Image.open(args.mask).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return mask

    mask = generate_auto_mask(image, build_auto_mask_config(args))
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    if args.auto_mask_output:
        Path(args.auto_mask_output).parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.auto_mask_output)

    return mask

def main():
    args = parse_args()

    image = Image.open(args.image).convert("RGB")
    mask = prepare_mask(args, image)

    w, h = image.size
    xs, pw = make_positions(w, args.patch_size, args.overlap)
    ys, ph = make_positions(h, args.patch_size, args.overlap)

    inpainter = AncientPaintingInpainter(
        InpaintConfig(
            model_id=args.model_id,
            lora_weights=args.lora_weights,
            device=args.device,
            dtype=args.dtype,
            max_size=args.max_size,
        )
    )

    accum = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)

    for y in ys:
        for x in xs:
            box = (x, y, x + pw, y + ph)
            patch_img = image.crop(box)
            patch_mask = mask.crop(box)

            patch_mask_np = np.array(patch_mask)
            win = feather_window(ph, pw, args.overlap)[..., None]

            if patch_mask_np.max() == 0:
                patch_np = np.array(patch_img).astype(np.float32)
                accum[y:y+ph, x:x+pw] += patch_np * win
                weight[y:y+ph, x:x+pw] += win
                continue

            result = inpainter.restore(
                patch_img,
                patch_mask,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                strength=args.strength,
                seed=args.seed,
                preserve_unmasked=True,
                blur_mask=args.blur_mask,
            )

            if result.size != (pw, ph):
                result = result.resize((pw, ph), Image.Resampling.BICUBIC)

            result_np = np.array(result).astype(np.float32)
            orig_np = np.array(patch_img).astype(np.float32)
            alpha = (patch_mask_np.astype(np.float32) / 255.0)[..., None]
            alpha = np.clip(alpha * win, 0.0, 1.0)
            mixed_np = orig_np * (1.0 - alpha) + result_np * alpha

            accum[y:y+ph, x:x+pw] += mixed_np
            weight[y:y+ph, x:x+pw] += 1.0

    output_np = accum / np.clip(weight, 1e-6, None)
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)
    out = Image.fromarray(output_np)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()