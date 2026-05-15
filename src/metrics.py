from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .image_utils import find_matching_file, list_images, load_mask, load_rgb


def _to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def compute_pair_metrics(
    pred_path: str | Path,
    clean_path: str | Path,
    mask_path: str | Path | None = None,
) -> dict:
    pred = load_rgb(pred_path)
    clean = load_rgb(clean_path).resize(pred.size, Image.Resampling.LANCZOS)
    pred_arr = _to_array(pred)
    clean_arr = _to_array(clean)
    psnr = float(peak_signal_noise_ratio(clean_arr, pred_arr, data_range=1.0))
    ssim = float(
        structural_similarity(clean_arr, pred_arr, channel_axis=2, data_range=1.0)
    )
    result = {"psnr": psnr, "ssim": ssim}
    if mask_path is not None:
        mask = load_mask(mask_path, size=pred.size)
        mask_arr = (np.asarray(mask) > 127).astype(bool)
        if mask_arr.any():
            diff = (pred_arr - clean_arr) ** 2
            mse = float(diff[mask_arr].mean())
            result["masked_psnr"] = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSNR and SSIM.")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--clean_dir", default="data/val/clean")
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for pred_path in list_images(args.pred_dir):
        clean_path = find_matching_file(args.clean_dir, pred_path.stem)
        if clean_path is None:
            continue
        mask_path = (
            find_matching_file(args.mask_dir, pred_path.stem)
            if args.mask_dir is not None
            else None
        )
        metrics = compute_pair_metrics(pred_path, clean_path, mask_path)
        rows.append({"stem": pred_path.stem, **metrics})

    if not rows:
        raise FileNotFoundError("No matching prediction/clean pairs found.")
    keys = [key for key in rows[0].keys() if key != "stem"]
    summary = {
        key: float(np.mean([row[key] for row in rows if key in row])) for key in keys
    }
    payload = {"count": len(rows), "summary": summary, "items": rows}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
