# tile_infer.py 使用说明

`tile_infer.py` 是一个分块推理脚本，用于对大尺寸古画、长图、立轴图或细小划痕较多的图像进行修复。

原来的单图推理会把整张图送入模型。如果图像很长或分辨率较高，细小划痕在缩放后可能变得不明显，导致修复效果变差。`tile_infer.py` 会先把图像切成多个重叠的小块，对每个小块分别进行 inpainting，最后再自动拼接回完整图像。

## 主要作用

- 将大图或长图切成多个 patch 进行局部修复；
- 支持手动 mask 和自动 mask；
- 支持原始 Stable Diffusion Inpainting 模型；
- 支持加载 LoRA 权重；
- 只修复包含白色 mask 的 patch，跳过不需要修复的区域；
- 使用重叠区域融合，减少拼接边界。

## Mask 规则

```text
白色 / 255：需要修复的区域
黑色 / 0：保持不变的区域
```

建议使用 PNG 格式保存 mask。

## 使用方法

### 1. 手动 mask + 原始模型

```bash
python -m src.tile_infer \
  --model_id models/stable-diffusion-inpainting \
  --image test/img_inpainting.jpg \
  --mask test/img_inpainting_mask.png \
  --mask_mode manual \
  --output outputs/test/tiled_base_manual.png \
  --patch_size 512 \
  --overlap 64 \
  --steps 50 \
  --seed 42 \
  --dtype fp16 \
  --device cuda \
  --max_size 512
```

### 2. 手动 mask + LoRA

```bash
python -m src.tile_infer \
  --model_id models/stable-diffusion-inpainting \
  --lora_weights checkpoints/inpaint_lora \
  --image test/img_inpainting.jpg \
  --mask test/img_inpainting_mask.png \
  --mask_mode manual \
  --output outputs/test/tiled_lora_manual.png \
  --patch_size 512 \
  --overlap 64 \
  --steps 50 \
  --seed 42 \
  --dtype fp16 \
  --device cuda \
  --max_size 512
```

### 3. 自动 mask + 原始模型

```bash
python -m src.tile_infer \
  --model_id models/stable-diffusion-inpainting \
  --image test/img_inpainting.jpg \
  --mask_mode auto \
  --auto_mask_output outputs/test/auto_mask.png \
  --output outputs/test/tiled_base_auto.png \
  --patch_size 512 \
  --overlap 64 \
  --steps 50 \
  --seed 42 \
  --dtype fp16 \
  --device cuda \
  --max_size 512
```

### 4. 自动 mask + LoRA

```bash
python -m src.tile_infer \
  --model_id models/stable-diffusion-inpainting \
  --lora_weights checkpoints/inpaint_lora \
  --image test/img_inpainting.jpg \
  --mask_mode auto \
  --auto_mask_output outputs/test/auto_mask.png \
  --output outputs/test/tiled_lora_auto.png \
  --patch_size 512 \
  --overlap 64 \
  --steps 50 \
  --seed 42 \
  --dtype fp16 \
  --device cuda \
  --max_size 512
```

## 常用参数

| 参数 | 说明 |
|---|---|
| `--model_id` | 基础模型路径或 Hugging Face 模型名 |
| `--lora_weights` | LoRA 权重路径，不传则使用原始模型 |
| `--image` | 输入缺损图像 |
| `--mask` | 手动 mask 路径 |
| `--mask_mode` | `manual` 表示使用手动 mask，`auto` 表示自动生成 mask |
| `--auto_mask_output` | 保存自动生成的 mask |
| `--output` | 输出修复结果路径 |
| `--patch_size` | 分块大小，常用 512 |
| `--overlap` | 相邻 patch 的重叠宽度，常用 64 |
| `--steps` | 推理步数 |
| `--device` | 推理设备，如 `cuda` 或 `cpu` |
| `--dtype` | 推理精度，如 `fp16` |
| `--max_size` | 每个 patch 输入模型前的最大尺寸 |

## 推荐参数

```text
patch_size = 512
overlap = 64
steps = 50
max_size = 512
```

如果拼接边界明显，可以把 `overlap` 调大，例如：

```text
overlap = 96
```

如果缺损区域很细小，可以尝试：

```text
patch_size = 384
```
