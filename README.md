# Ancient Painting Inpainting

This project implements a Stable Diffusion inpainting workflow for damaged
ancient paintings:

- damaged image + binary mask -> repaired image
- optional LoRA style adapter for ancient-painting texture
- paired inpainting fine-tuning with `clean/damaged/mask` data
- second-stage refinement from generated repairs and clean references
- Gradio app entrypoint for Hugging Face Spaces

The LoRA design follows the same engineering idea as the local HesClip project:
freeze the pretrained backbone and train only small low-rank adapter weights.
The actual diffusion training uses Diffusers + PEFT so the saved weights can be
loaded with `pipe.load_lora_weights(...)`.

## Data Layout

```text
data/
  lora_train/
    images/
    captions/
  train/
    clean/
    damaged/
    mask/
  val/
    clean/
    damaged/
    mask/
```

Mask convention: white pixels (`255`) are damaged regions to repair; black
pixels (`0`) are preserved.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version
before installing the rest of the requirements.

## Generate Synthetic Damage

```bash
python -m src.make_damage --clean_dir raw/train_clean --output_root data --split train --size 512
python -m src.make_damage --clean_dir raw/val_clean --output_root data --split val --size 512
```

## Inference

Single image:

```bash
python -m src.infer_sd_lora \
  --image data/val/damaged/0001.png \
  --mask data/val/mask/0001.png \
  --output outputs/sd_lora/0001.png \
  --lora_weights checkpoints/ancient_painting_lora
```

Batch:

```bash
python -m src.infer_sd_lora \
  --input_dir data/val/damaged \
  --mask_dir data/val/mask \
  --output_dir outputs/sd_lora \
  --lora_weights checkpoints/ancient_painting_lora
```

If you do not have LoRA weights yet, omit `--lora_weights` and the script will
run the base Hugging Face inpainting model.

## Train Ancient-Painting LoRA

Put style images in `data/lora_train/images`. Optional captions go in
`data/lora_train/captions` with matching stems, for example
`images/0001.png` and `captions/0001.txt`.

```bash
accelerate launch -m src.train_lora \
  --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
  --train_data_dir data/lora_train \
  --output_dir checkpoints/ancient_painting_lora \
  --resolution 512 \
  --train_batch_size 1 \
  --rank 8 \
  --max_train_steps 1000
```

## Paired Inpainting Fine-Tuning

This uses `data/train/clean`, `data/train/damaged`, and `data/train/mask`.
Default mode trains a LoRA adapter:

```bash
accelerate launch -m src.train_inpaint_lora \
  --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
  --train_root data/train \
  --output_dir checkpoints/inpaint_lora \
  --resolution 512 \
  --train_batch_size 1 \
  --rank 8 \
  --max_train_steps 1000
```

To fine-tune the original UNet instead of LoRA, use:

```bash
accelerate launch -m src.train_inpaint_lora \
  --train_mode full_unet \
  --train_root data/train \
  --output_dir checkpoints/full_inpaint_model \
  --learning_rate 1e-6 \
  --max_train_steps 1000
```

Full UNet fine-tuning is much heavier than LoRA and needs more VRAM.

## Refine From Model Outputs

After generating repairs into `outputs/sd_lora`, you can refine against clean
references in either of two ways.

Use generated repairs directly during paired training:

```bash
accelerate launch -m src.train_inpaint_lora \
  --train_root data/train \
  --bootstrap_outputs_dir outputs/sd_lora \
  --bootstrap_probability 0.5 \
  --output_dir checkpoints/refined_inpaint_lora
```

Or materialize a refine dataset:

```bash
python -m src.build_refine_dataset \
  --clean_dir data/train/clean \
  --restored_dir outputs/sd_lora \
  --mask_dir data/train/mask \
  --output_root data/refine

accelerate launch -m src.train_inpaint_lora \
  --train_root data/refine \
  --output_dir checkpoints/refined_from_outputs_lora
```

## Metrics

```bash
python -m src.metrics \
  --pred_dir outputs/sd_lora \
  --clean_dir data/val/clean \
  --mask_dir data/val/mask \
  --output outputs/metrics.json
```

## Hugging Face Space

Upload this repository to a Gradio Space. Set these Space variables as needed:

```text
MODEL_ID=runwayml/stable-diffusion-inpainting
LORA_WEIGHTS=your-user/your-lora-repo
MAX_SIZE=768
TORCH_DTYPE=auto
```

The Space exposes a `restore` API endpoint. Example client call:

```python
from gradio_client import Client, handle_file

client = Client("your-user/your-space")
result = client.predict(
    image=handle_file("damaged.png"),
    mask=handle_file("mask.png"),
    prompt="ancient Chinese painting, restore damaged areas, preserve original style",
    negative_prompt="modern style, blurry, watermark",
    steps=30,
    guidance_scale=7.5,
    strength=1.0,
    seed=42,
    preserve_unmasked=True,
    api_name="/restore",
)
print(result)
```
