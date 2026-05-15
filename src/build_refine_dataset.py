from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .image_utils import find_matching_file, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paired refine set from generated repairs and clean images."
    )
    parser.add_argument("--clean_dir", default="data/train/clean")
    parser.add_argument("--restored_dir", default="outputs/sd_lora")
    parser.add_argument("--mask_dir", default="data/train/mask")
    parser.add_argument("--output_root", default="data/refine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    clean_out = output_root / "clean"
    damaged_out = output_root / "damaged"
    mask_out = output_root / "mask"
    for path in (clean_out, damaged_out, mask_out):
        path.mkdir(parents=True, exist_ok=True)

    count = 0
    for clean_path in list_images(args.clean_dir):
        restored_path = find_matching_file(args.restored_dir, clean_path.stem)
        mask_path = find_matching_file(args.mask_dir, clean_path.stem)
        if restored_path is None or mask_path is None:
            continue
        name = f"{clean_path.stem}.png"
        shutil.copy2(clean_path, clean_out / name)
        shutil.copy2(restored_path, damaged_out / name)
        shutil.copy2(mask_path, mask_out / name)
        count += 1
    print(f"Wrote {count} refine samples to {output_root}")


if __name__ == "__main__":
    main()
