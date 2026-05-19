# 古画图像修复项目

本项目实现一个古画图像修复实验系统。主方法使用 Stable Diffusion Inpainting + 古画风格 LoRA，baseline 使用 Partial Convolution（PConv）/ PConv-LoRA。模型可以输入受损图片和 mask 输出修复图；在没有 mask 时，也可以先自动估计 mask，再进行修复。

主要能力：

- 只输入一张破损图时，可以自动估计 mask 并生成完整修复图
- 输入 `damaged image + mask`，输出 `repaired image`
- 可以直接调用 Hugging Face 上的 inpainting 大模型
- 可以训练古画风格 LoRA
- 可以使用成对数据训练修复 LoRA
- 可以根据模型输出和原图构造二阶段 refine 数据
- 可以运行 PConv/PConv-LoRA baseline，与 SD-LoRA 做指标和可视化对比
- 可以部署为 Hugging Face Gradio Space，并通过 API 调用

mask 约定：

```text
mask = 255 / 白色：需要修复的受损区域
mask = 0 / 黑色：不需要修改的正常区域
```

## 项目结构

```text
Inpainting/
  app.py
  README.md
  requirements.txt
  .gitignore
  src/
    __init__.py
    image_utils.py
    pipeline.py
    infer_sd_lora.py
    tile_infer.py
    auto_mask.py
    make_damage.py
    datasets.py
    train_common.py
    train_lora.py
    train_inpaint_lora.py
    build_refine_dataset.py
    metrics.py
    hesclip_lora.py
    pconv/
      __init__.py
      partialconv2d.py
      pconv_unet.py
      lora_conv.py
      datasets.py
      train_pconv_lora.py
      infer_pconv_lora.py
      tile_pconv.py
      utils.py
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
  checkpoints/
  outputs/
  test/
    img_inpainting.jpg
    img_inpainting_mask.png
```

## 文件作用

### 根目录文件

| 文件 | 作用 |
|---|---|
| `README.md` | 项目说明文档，包含数据结构、文件作用、安装、训练、推理、评估和部署命令。 |
| `requirements.txt` | 项目依赖列表，包括 `diffusers`、`transformers`、`peft`、`accelerate`、`gradio`、`torch` 等。 |
| `app.py` | Hugging Face Space / Gradio 应用入口。用户上传受损图和可选 mask 后，调用修复 pipeline，输出修复图；同时暴露 `/restore` API。 |
| `.gitignore` | 忽略缓存、模型权重、输出图、训练数据等大文件，但保留必要目录结构。 |

### `src/` 代码文件

| 文件 | 作用 |
|---|---|
| `src/pipeline.py` | 推理核心。封装 `StableDiffusionInpaintPipeline`，负责加载 Hugging Face inpainting 模型、加载 LoRA、处理 prompt、mask、seed，并输出修复图。主要类是 `AncientPaintingInpainter`。 |
| `src/infer_sd_lora.py` | 命令行推理脚本。支持单张图片修复，也支持批量处理 `input_dir + mask_dir`。 |
| `src/tile_infer.py` | Stable Diffusion 瓦片推理脚本，用于适配长方形古画、长卷图像或高宽比例较大的输入图像。支持手动 mask 和自动 mask。 |
| `src/auto_mask.py` | 自动 mask 生成模块。只给一张破损图时，使用亮度异常、暗色污渍、白色划痕、局部边缘等图像处理规则估计需要修复的区域。 |
| `src/train_lora.py` | 古画风格 LoRA 训练脚本。使用 `data/lora_train/images` 和 `data/lora_train/captions`，让模型学习古画纹理、纸张质感、笔触风格。 |
| `src/train_inpaint_lora.py` | 成对修复微调脚本。使用 `data/train/clean`、`data/train/damaged`、`data/train/mask`。默认训练 LoRA；传入 `--train_mode full_unet` 时，可以微调整个 UNet。 |
| `src/train_common.py` | 训练公共工具。负责加载 SD inpainting 组件、添加 LoRA、保存 LoRA 权重、保存完整模型、tokenize caption、设置随机种子等。 |
| `src/datasets.py` | PyTorch 数据集定义。`CaptionImageDataset` 用于古画风格 LoRA 训练；`PairedInpaintDataset` 用于 clean/damaged/mask 成对微调。 |
| `src/image_utils.py` | 图像工具函数。负责加载 RGB 图、加载二值 mask、resize、裁剪、tensor 转换、根据 mask 合成最终结果、查找同名文件等。 |
| `src/make_damage.py` | 自动构造训练数据。从 clean 图生成划痕、污渍、缺失区域，同时保存 damaged 图和 mask。 |
| `src/metrics.py` | 评价脚本。对预测结果和 clean 原图计算 `PSNR`、`SSIM`；如果提供 mask，还会计算 masked PSNR。 |
| `src/build_refine_dataset.py` | 二阶段微调用。把第一阶段模型输出的修复图作为新的 damaged/input，再和 clean 原图、mask 组成 refine 数据集。 |
| `src/hesclip_lora.py` | 参考 HesClip 项目的 LoRA 低秩线性层实现，体现“冻结主模型，只训练 LoRA A/B 参数”的思想。实际 Diffusers 训练主要使用 PEFT/Diffusers 原生 LoRA 保存格式。 |
| `src/__init__.py` | 让 `src` 成为 Python 包，支持 `python -m src.xxx` 方式运行脚本。 |

### `src/pconv/` baseline 文件

| 文件 | 作用 |
|---|---|
| `src/pconv/partialconv2d.py` | Partial Convolution 基础层。 |
| `src/pconv/pconv_unet.py` | PConv-UNet 网络结构。 |
| `src/pconv/lora_conv.py` | 给卷积层加入 LoRA 分支，并冻结基础卷积权重。 |
| `src/pconv/datasets.py` | 读取 `clean / damaged / mask` 成对数据，并把项目 mask 转换成 PConv 所需的 valid mask。 |
| `src/pconv/train_pconv_lora.py` | PConv-LoRA 微调入口。 |
| `src/pconv/infer_pconv_lora.py` | PConv 或 PConv-LoRA 推理入口。 |
| `src/pconv/tile_pconv.py` | PConv-LoRA 瓦片推理入口，适合长方形古画、长卷图像或高宽比例较大的输入图像。 |
| `src/pconv/utils.py` | checkpoint、保存图像、loss、PSNR、设备选择等工具函数。 |

### 数据和输出目录

| 目录 | 作用 |
|---|---|
| `data/lora_train/images` | 用于训练古画风格 LoRA 的古画图片。 |
| `data/lora_train/captions` | LoRA 训练图片对应的 caption，文件名需要和图片同名，例如 `0001.png` 对应 `0001.txt`。 |
| `data/train/clean` | 训练集 clean 原图。 |
| `data/train/damaged` | 训练集受损图。 |
| `data/train/mask` | 训练集受损区域 mask。 |
| `data/val/clean` | 验证集 clean 原图，用于计算 PSNR / SSIM。 |
| `data/val/damaged` | 验证集受损图。 |
| `data/val/mask` | 验证集 mask。 |
| `checkpoints/` | 保存训练得到的 LoRA 权重或完整微调模型。 |
| `outputs/` | 保存推理结果、评价指标、对比图等输出。 |
| `test/` | 保存示例测试图和对应 mask，用于演示瓦片推理等流程。 |

## 安装依赖

建议先根据自己的 CUDA 版本安装合适的 PyTorch，然后再安装其他依赖。

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

如果只做 CPU 推理，也可以直接安装 `requirements.txt`，但 Stable Diffusion 在 CPU 上会非常慢。

## 数据准备

项目默认使用如下数据结构：

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

每组成对修复数据需要同名：

```text
data/train/clean/0001.png
data/train/damaged/0001.png
data/train/mask/0001.png
```

如果只有 clean 图，可以用脚本自动合成 damaged 和 mask：

```bash
python -m src.make_damage --clean_dir raw/train_clean --output_root data --split train --size 512
python -m src.make_damage --clean_dir raw/val_clean --output_root data --split val --size 512
```

## 推理

### 只有一张破损图时自动修复

如果没有 mask，推理脚本会先调用 `src.auto_mask` 自动估计受损区域，再把生成的 mask 送入 Stable Diffusion Inpainting。

```bash
python -m src.infer_sd_lora \
  --image damaged.png \
  --output outputs/sd_lora/damaged.png \
  --auto_mask_output outputs/auto_mask/damaged_mask.png \
  --lora_weights checkpoints/ancient_painting_lora
```

也可以只生成自动 mask，方便检查或手动修改：

```bash
python -m src.auto_mask \
  --image damaged.png \
  --output outputs/auto_mask/damaged_mask.png
```

自动 mask 适合明显的白色裂痕、划痕、暗色污渍和局部缺失。对复杂古画，手动修正 mask 通常会得到更稳定的修复效果。

### 单张图片修复

```bash
python -m src.infer_sd_lora \
  --image data/val/damaged/0001.png \
  --mask data/val/mask/0001.png \
  --output outputs/sd_lora/0001.png \
  --lora_weights checkpoints/ancient_painting_lora
```

如果暂时没有训练好的 LoRA，可以去掉 `--lora_weights`，脚本会直接使用 Hugging Face 上的基础 inpainting 模型。

### 批量修复

如果 `--mask_dir` 不提供，脚本会为每张输入图自动生成 mask：

```bash
python -m src.infer_sd_lora \
  --input_dir data/val/damaged \
  --output_dir outputs/sd_lora \
  --auto_mask_output_dir outputs/auto_mask \
  --lora_weights checkpoints/ancient_painting_lora
```

如果已经有 mask，使用下面的命令：

```bash
python -m src.infer_sd_lora \
  --input_dir data/val/damaged \
  --mask_dir data/val/mask \
  --output_dir outputs/sd_lora \
  --lora_weights checkpoints/ancient_painting_lora
```

默认 prompt 位于 `src/pipeline.py`：

```text
ancient Chinese painting, restore damaged areas, preserve original style,
natural texture, aged paper, traditional brush strokes
```

### Stable Diffusion 瓦片推理

本项目保留并改造了瓦片推理，用于适配长方形古画、长卷图像或高宽比例较大的输入图像。它不是因为模型只能处理正方形图片，也不是单纯为了节省显存，而是为了避免把长方形古画强行缩放到固定尺寸时造成比例变化、细节损失或局部纹理不自然。

瓦片推理的做法是：按照宽度和高度方向把原图和 mask 同步切分为多个带重叠区域的小块 patch，分别送入 Stable Diffusion Inpainting 修复，最后再按原始位置拼回完整图像，并在重叠区域做平滑融合。这样可以尽量保留原图比例和局部细节，更适合古画长卷、横幅图像等非正方形输入。

对于普通尺寸图片，可以直接使用 `src.infer_sd_lora`；对于长方形大图或长卷图像，推荐使用 `src.tile_infer`：

```bash
python -m src.tile_infer \
  --model_id runwayml/stable-diffusion-inpainting \
  --image test/img_inpainting.jpg \
  --mask test/img_inpainting_mask.png \
  --output outputs/sd_lora/tile_result.png \
  --patch_size 512 \
  --overlap 64
```

如果没有 mask，可以使用自动 mask：

```bash
python -m src.tile_infer \
  --model_id runwayml/stable-diffusion-inpainting \
  --image test/img_inpainting.jpg \
  --mask_mode auto \
  --auto_mask_output outputs/auto_mask/img_inpainting_mask.png \
  --output outputs/sd_lora/tile_result.png
```

## 训练古画风格 LoRA

该步骤只需要古画图片和 caption，不需要 damaged-clean 成对数据。

准备数据：

```text
data/lora_train/images/0001.png
data/lora_train/captions/0001.txt
```

caption 示例：

```text
ancient Chinese painting, traditional brush strokes, aged paper texture
```

训练命令：

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

训练完成后，`checkpoints/ancient_painting_lora` 可以作为 `--lora_weights` 被推理脚本加载。

## 使用成对数据微调修复能力

该步骤使用：

```text
data/train/clean
data/train/damaged
data/train/mask
```

默认训练 LoRA，只更新少量低秩参数：

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

如果需要微调原模型 UNet，可以使用：

```bash
accelerate launch -m src.train_inpaint_lora \
  --train_mode full_unet \
  --train_root data/train \
  --output_dir checkpoints/full_inpaint_model \
  --learning_rate 1e-6 \
  --max_train_steps 1000
```

注意：`full_unet` 比 LoRA 更耗显存，也更容易过拟合。课程实验中通常优先使用 LoRA。

## 根据输出结果继续 refine

如果已经用模型生成了一批修复结果，例如放在：

```text
outputs/sd_lora
```

可以用两种方式把这些结果用于继续微调。

### 方式一：训练时混入模型输出

```bash
accelerate launch -m src.train_inpaint_lora \
  --train_root data/train \
  --bootstrap_outputs_dir outputs/sd_lora \
  --bootstrap_probability 0.5 \
  --output_dir checkpoints/refined_inpaint_lora
```

含义：训练时有 50% 概率用 `outputs/sd_lora` 里的修复结果作为 conditioning image，再以 clean 原图为目标继续训练。

### 方式二：显式构造 refine 数据集

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

此时 `data/refine/damaged` 实际存放的是上一轮模型输出的修复图，`data/refine/clean` 仍然是原始 clean 图。

## PConv / PConv-LoRA baseline

PConv baseline 用于和 Stable Diffusion Inpainting / SD-LoRA 进行对比。它沿用本项目的数据格式：

```text
data/train/clean
data/train/damaged
data/train/mask
data/val/clean
data/val/damaged
data/val/mask
```

项目 mask 约定是：

```text
255：需要修复区域
0：正常区域
```

PConv 内部需要的是 valid mask：

```text
1：有效区域
0：缺失区域
```

代码中会自动转换：

```text
damage_mask = mask / 255
valid_mask = 1 - damage_mask
```

### 准备 PConv 预训练权重

PConv-LoRA 需要先准备 PConv / PConv-UNet 预训练权重，例如：

```text
models/pconv/pconv.pth
```

注意：PConv 权重不能和 Stable Diffusion 权重混用。如果 checkpoint 出现大量 missing/unexpected keys，说明权重结构和当前 `PConvUNet` 不匹配，需要做 key 映射或调整网络结构。

### PConv-LoRA smoke test

正式训练前建议先跑 1 个 epoch，确认数据、权重和流程可用：

```bash
python -m src.pconv.train_pconv_lora \
  --train_root data/train \
  --val_root data/val \
  --pretrained_pconv models/pconv/pconv.pth \
  --output_dir checkpoints/pconv_lora_smoke \
  --resolution 512 \
  --batch_size 1 \
  --epochs 1 \
  --rank 4 \
  --learning_rate 1e-4
```

### 正式训练 PConv-LoRA

```bash
python -m src.pconv.train_pconv_lora \
  --train_root data/train \
  --val_root data/val \
  --pretrained_pconv models/pconv/pconv.pth \
  --output_dir checkpoints/pconv_lora \
  --resolution 512 \
  --batch_size 1 \
  --epochs 10 \
  --rank 8 \
  --learning_rate 1e-4
```

输出：

```text
checkpoints/pconv_lora/
  best.pth
  latest.pth
  history.json
```

### PConv-LoRA 推理

```bash
python -m src.pconv.infer_pconv_lora \
  --input_dir data/val/damaged \
  --mask_dir data/val/mask \
  --checkpoint checkpoints/pconv_lora/best.pth \
  --output_dir outputs/pconv_lora \
  --resolution 512
```

如果只想运行未加 LoRA 的 PConv baseline，可以传入 `--pretrained_pconv`：

```bash
python -m src.pconv.infer_pconv_lora \
  --input_dir data/val/damaged \
  --mask_dir data/val/mask \
  --pretrained_pconv models/pconv/pconv.pth \
  --output_dir outputs/pconv \
  --resolution 512
```

### PConv 瓦片推理

PConv 瓦片推理与 Stable Diffusion 瓦片推理的定位类似，主要用于处理长方形古画、长卷图像或高宽比例较大的输入图像。它会把原图和 mask 同步切分成多个重叠 patch，分别使用 PConv/PConv-LoRA 修复，再融合回原图尺寸，从而减少直接缩放长方形图像带来的比例和细节损失。

```bash
python -m src.pconv.tile_pconv \
  --image test/img_inpainting.jpg \
  --mask test/img_inpainting_mask.png \
  --checkpoint checkpoints/pconv_lora/best.pth \
  --output outputs/pconv_lora/tile_result.png \
  --patch_size 512 \
  --overlap 64
```

也可以让 PConv 瓦片推理自动生成 mask：

```bash
python -m src.pconv.tile_pconv \
  --image test/img_inpainting.jpg \
  --mask_mode auto \
  --auto_mask_output outputs/auto_mask/img_inpainting_mask.png \
  --checkpoint checkpoints/pconv_lora/best.pth \
  --output outputs/pconv_lora/tile_result.png
```

## 评价指标

在有 clean 参考图的验证集上，可以计算 PSNR、SSIM、masked PSNR，并生成逐图 CSV：

```bash
python -m src.metrics \
  --pred_dir outputs/sd_lora \
  --clean_dir data/val/clean \
  --mask_dir data/val/mask \
  --output outputs/metrics/sd_lora_metrics.json
```

输出包括：

- `psnr`：整张图的 PSNR
- `ssim`：整张图的 SSIM
- `masked_psnr`：只在 mask 区域计算的 PSNR
- `mask_ratio`：mask 面积占比
- `*_detail.csv`：每张图的详细指标

如果测试图没有 clean 参考图，则不能严格计算 PSNR / SSIM，只能做视觉质量分析。

### 指标计算 + 样本可视化

```bash
python -m src.metrics \
  --pred_dir outputs/sd_lora \
  --clean_dir data/val/clean \
  --mask_dir data/val/mask \
  --damaged_dir data/val/damaged \
  --output outputs/metrics/sd_lora_metrics.json \
  --detail_csv outputs/metrics/sd_lora_metrics_detail.csv \
  --vis_dir outputs/vis/sd_lora \
  --top_k 5
```

输出目录中会保存 `best / middle / worst` 样本拼图。每张拼图包含：

```text
clean | damaged | mask | result | error
```

### 多方法对比

当已经得到 SD-LoRA 和 PConv-LoRA 的逐图 CSV 后，可以统一生成对比图：

```bash
python -m src.metrics \
  --compare_csv SD-LoRA=outputs/metrics/sd_lora_metrics_detail.csv \
  --compare_csv PConv-LoRA=outputs/metrics/pconv_lora_metrics_detail.csv \
  --compare_pred_dir SD-LoRA=outputs/sd_lora \
  --compare_pred_dir PConv-LoRA=outputs/pconv_lora \
  --clean_dir data/val/clean \
  --damaged_dir data/val/damaged \
  --mask_dir data/val/mask \
  --vis_dir outputs/vis/compare \
  --top_k 5
```

## Hugging Face Space 部署

将本仓库上传到 Hugging Face Gradio Space 后，可以设置环境变量：

```text
MODEL_ID=runwayml/stable-diffusion-inpainting
LORA_WEIGHTS=your-user/your-lora-repo
MAX_SIZE=768
TORCH_DTYPE=auto
```

其中：

- `MODEL_ID` 是基础 inpainting 模型
- `LORA_WEIGHTS` 是 LoRA 权重路径，可以是本地路径，也可以是 Hugging Face repo
- `MAX_SIZE` 控制输入图像最大边长，避免显存爆掉
- `TORCH_DTYPE` 可设为 `auto`、`fp16`、`bf16`、`fp32`

Space 会暴露 `/restore` API。客户端调用示例：

如果没有 mask，可以把 `mask` 设为 `None`，Space 会自动生成 mask。

```python
from gradio_client import Client, handle_file

client = Client("your-user/your-space")
result = client.predict(
    image=handle_file("damaged.png"),
    mask=None,
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

## 推荐实验流程

最小可运行流程：

1. 准备 clean 图像。
2. 运行 `src.make_damage` 生成 `damaged` 和 `mask`。
3. 准备 `data/lora_train/images` 和 caption。
4. 运行 `src.train_lora` 训练古画风格 LoRA。
5. 运行 `src.infer_sd_lora` 修复验证集或测试图。
6. 运行 `src.metrics` 在有 clean 参考的验证集上计算指标。
7. 使用 `src.train_inpaint_lora` 进一步做 paired inpainting 微调。
8. 可选：使用 `outputs/sd_lora` 和 clean 原图继续 refine。
9. 准备 PConv 预训练权重，运行 `src.pconv.train_pconv_lora` 和 `src.pconv.infer_pconv_lora` 得到 baseline。
10. 使用 `src.metrics` 对 SD-LoRA 和 PConv-LoRA 做统一指标与可视化对比。

## 说明

本项目没有从零训练 Stable Diffusion，而是在 Hugging Face 预训练 inpainting 模型的基础上进行 LoRA 适配和可选 UNet 微调。PConv baseline 是独立 CNN/U-Net 路线，不能使用 Stable Diffusion 权重。这样的设计更符合课程项目的计算资源限制，也能体现完整的深度学习训练、推理、baseline 对比和评估流程。
