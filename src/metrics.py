import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from skimage.metrics import structural_similarity
except Exception:
    structural_similarity = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(path):
    path = Path(path)
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def find_same_name(directory, name):
    if directory is None:
        return None
    directory = Path(directory)
    p = directory / name
    if p.exists():
        return p
    stem = Path(name).stem
    candidates = [q for q in directory.iterdir() if q.stem == stem and q.suffix.lower() in IMAGE_EXTS]
    if not candidates:
        return None
    return sorted(candidates)[0]


def load_rgb(path, size=None, resample=Image.Resampling.BICUBIC):
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, resample)
    return img


def load_mask_bool(path, size):
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(mask).astype(np.float32) / 255.0
    return arr >= 0.5


def image_to_float(image):
    return np.asarray(image).astype(np.float32) / 255.0


def compute_psnr(pred, target, mask=None):
    if mask is None:
        mse = np.mean((pred - target) ** 2)
    else:
        if mask.sum() == 0:
            return None
        diff = pred[mask] - target[mask]
        mse = np.mean(diff ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim(pred, target):
    if structural_similarity is None:
        return None
    try:
        return float(structural_similarity(target, pred, channel_axis=-1, data_range=1.0))
    except TypeError:
        return float(structural_similarity(target, pred, multichannel=True, data_range=1.0))


def mean_ignore_none(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None
    finite_values = [v for v in values if not (isinstance(v, float) and math.isinf(v))]
    if finite_values:
        return float(np.mean(finite_values))
    return float("inf")


def save_detail_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "psnr", "ssim", "masked_psnr", "mask_ratio"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_detail_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {"filename": row["filename"]}
            for key in ["psnr", "ssim", "masked_psnr", "mask_ratio"]:
                value = row.get(key, "")
                if value in ("", "None", "nan"):
                    item[key] = None
                elif value == "inf":
                    item[key] = float("inf")
                else:
                    item[key] = float(value)
            rows.append(item)
    return rows


def compute_rows(pred_dir, clean_dir, mask_dir=None):
    pred_files = list_images(pred_dir)
    rows = []
    missing = []

    for pred_path in pred_files:
        clean_path = find_same_name(clean_dir, pred_path.name)
        if clean_path is None:
            missing.append({"filename": pred_path.name, "missing": "clean"})
            continue

        pred_img = load_rgb(pred_path)
        clean_img = load_rgb(clean_path)

        if pred_img.size != clean_img.size:
            pred_img = pred_img.resize(clean_img.size, Image.Resampling.BICUBIC)

        pred = image_to_float(pred_img)
        clean = image_to_float(clean_img)

        mask_bool = None
        mask_ratio = None
        if mask_dir is not None:
            mask_path = find_same_name(mask_dir, pred_path.name)
            if mask_path is None:
                missing.append({"filename": pred_path.name, "missing": "mask"})
            else:
                mask_bool = load_mask_bool(mask_path, clean_img.size)
                mask_ratio = float(mask_bool.mean())

        rows.append({
            "filename": pred_path.name,
            "psnr": compute_psnr(pred, clean),
            "ssim": compute_ssim(pred, clean),
            "masked_psnr": compute_psnr(pred, clean, mask_bool) if mask_bool is not None else None,
            "mask_ratio": mask_ratio
        })

    summary = {
        "num_images": len(rows),
        "psnr": mean_ignore_none([r["psnr"] for r in rows]),
        "ssim": mean_ignore_none([r["ssim"] for r in rows]),
        "masked_psnr": mean_ignore_none([r["masked_psnr"] for r in rows]),
        "mask_ratio": mean_ignore_none([r["mask_ratio"] for r in rows]),
        "missing": missing
    }
    return rows, summary


def parse_mapping(items):
    result = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid mapping: {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def error_map(pred, clean, mask):
    pred_arr = np.asarray(pred).astype(np.float32)
    clean_arr = np.asarray(clean).astype(np.float32)
    err = np.mean(np.abs(pred_arr - clean_arr), axis=2)

    if mask is not None:
        mask_arr = np.asarray(mask.convert("L").resize(clean.size, Image.Resampling.NEAREST)).astype(np.float32) / 255.0
        err = err * (mask_arr >= 0.5)

    vmax = err.max()
    if vmax > 0:
        err = err / vmax * 255.0
    return Image.fromarray(err.astype(np.uint8)).convert("RGB")


def add_label(img, label, height=28):
    canvas = Image.new("RGB", (img.width, img.height + height), "white")
    canvas.paste(img, (0, height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 6), label, fill="black", font=font)
    return canvas


def make_panel(images, labels, out_path, thumb_width=220):
    labeled = []
    for img, label in zip(images, labels):
        img = ImageOps.contain(img, (thumb_width, thumb_width))
        canvas = Image.new("RGB", (thumb_width, thumb_width), "white")
        canvas.paste(img, ((thumb_width - img.width) // 2, (thumb_width - img.height) // 2))
        labeled.append(add_label(canvas, label))

    width = sum(img.width for img in labeled)
    height = max(img.height for img in labeled)
    panel = Image.new("RGB", (width, height), "white")
    x = 0
    for img in labeled:
        panel.paste(img, (x, 0))
        x += img.width
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def select_cases(rows, top_k):
    valid = [
        row for row in rows
        if row.get("masked_psnr") is not None
        and not math.isinf(row["masked_psnr"])
        and not math.isnan(row["masked_psnr"])
    ]
    valid = sorted(valid, key=lambda x: x["masked_psnr"])
    if not valid:
        return {"worst": [], "middle": [], "best": []}
    worst = valid[:top_k]
    best = valid[-top_k:][::-1]
    mid = len(valid) // 2
    start = max(0, mid - top_k // 2)
    middle = valid[start:start + top_k]
    return {"worst": worst, "middle": middle, "best": best}


def save_case_panels(method_rows, pred_dirs, clean_dir, damaged_dir, mask_dir, output_dir, top_k, direct_single=False):
    saved = {}
    for label, rows in method_rows.items():
        if label not in pred_dirs:
            continue

        pred_dir = pred_dirs[label]
        cases = select_cases(rows, top_k)
        safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        saved[label] = {"best": 0, "middle": 0, "worst": 0}

        for group_name, group_rows in cases.items():
            for row in group_rows:
                name = row["filename"]
                clean_path = find_same_name(clean_dir, name)
                damaged_path = find_same_name(damaged_dir, name)
                mask_path = find_same_name(mask_dir, name)
                pred_path = find_same_name(pred_dir, name)

                if clean_path is None or damaged_path is None or mask_path is None or pred_path is None:
                    continue

                clean = load_rgb(clean_path)
                damaged = load_rgb(damaged_path, clean.size)
                mask = Image.open(mask_path).convert("L").resize(clean.size, Image.Resampling.NEAREST)
                mask_rgb = mask.convert("RGB")
                pred = load_rgb(pred_path, clean.size)
                err = error_map(pred, clean, mask)

                score = row.get("masked_psnr")
                score_text = "nan" if score is None else f"{score:.2f}"

                if direct_single:
                    out_path = Path(output_dir) / group_name / f"{Path(name).stem}_compare.png"
                    result_label = "result"
                else:
                    out_path = Path(output_dir) / safe_label / group_name / f"{Path(name).stem}_compare.png"
                    result_label = label

                make_panel(
                    [clean, damaged, mask_rgb, pred, err],
                    ["clean", "damaged", "mask", result_label, f"error {score_text}"],
                    out_path
                )
                saved[label][group_name] += 1
    return saved


def build_default_detail_csv_path(output_json):
    output_json = Path(output_json)
    if output_json.suffix.lower() == ".json":
        return output_json.with_name(output_json.stem + "_detail.csv")
    return output_json.parent / "metrics_detail.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default=None)
    parser.add_argument("--clean_dir", default=None)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--damaged_dir", default=None)
    parser.add_argument("--output", default="outputs/metrics/metrics.json")
    parser.add_argument("--detail_csv", default=None)
    parser.add_argument("--vis_dir", default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--compare_csv", action="append", default=[])
    parser.add_argument("--compare_pred_dir", action="append", default=[])
    args = parser.parse_args()

    method_rows = {}
    pred_dirs = {}

    if args.pred_dir is not None:
        if args.clean_dir is None:
            raise ValueError("--clean_dir is required when --pred_dir is used")
        rows, summary = compute_rows(args.pred_dir, args.clean_dir, args.mask_dir)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detail_csv = args.detail_csv or str(build_default_detail_csv_path(output_path))
        save_detail_csv(rows, detail_csv)

        summary["detail_csv"] = detail_csv
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(json.dumps(summary, ensure_ascii=False, indent=2))

        method_rows["result"] = rows
        pred_dirs["result"] = args.pred_dir

    compare_csv = parse_mapping(args.compare_csv)
    for name, path in compare_csv.items():
        method_rows[name] = read_detail_csv(path)

    compare_pred_dir = parse_mapping(args.compare_pred_dir)
    for name, path in compare_pred_dir.items():
        pred_dirs[name] = path

    if args.vis_dir is not None:
        if not method_rows:
            raise ValueError("No method results available for case visualization")
        if args.clean_dir is None or args.damaged_dir is None or args.mask_dir is None:
            raise ValueError("--clean_dir, --damaged_dir and --mask_dir are required when --vis_dir is used")

        direct_single = args.pred_dir is not None and len(compare_csv) == 0 and len(compare_pred_dir) == 0

        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        saved = save_case_panels(
            method_rows,
            pred_dirs,
            Path(args.clean_dir),
            Path(args.damaged_dir),
            Path(args.mask_dir),
            vis_dir,
            args.top_k,
            direct_single=direct_single
        )
        print(json.dumps({"case_visualization_dir": str(vis_dir), "saved_cases": saved}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
