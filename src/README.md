# Metrics 使用说明

`src/metrics.py` 用于图像修复结果的指标计算和样本可视化。

主要输出：

- 平均指标 JSON
- 逐图指标 CSV
- `best / middle / worst` 样本拼图

## 1. 数据格式

验证集目录：

```text
data/val/
  clean/
  damaged/
  mask/
```

模型输出目录示例：

```text
outputs/sd_lora/
outputs/pconv_lora/
```

要求输出图像与 `clean / damaged / mask` 中的图像尽量同名。

mask 约定：

```text
255：需要修复区域
0：正常区域
```

## 2. 单方法指标计算

以 `outputs/sd_lora` 为例：

```bash
python -m src.metrics \
  --pred_dir outputs/sd_lora \
  --clean_dir data/val/clean \
  --mask_dir data/val/mask \
  --output outputs/metrics/sd_lora_metrics.json
```

输出：

```text
outputs/metrics/sd_lora_metrics.json
outputs/metrics/sd_lora_metrics_detail.csv
```

其中：

- `sd_lora_metrics.json`：平均指标
- `sd_lora_metrics_detail.csv`：每张图像的详细指标

## 3. 单方法指标计算 + 样本可视化

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

输出：

```text
outputs/vis/sd_lora/
  best/
  middle/
  worst/
```

每张拼图格式为：

```text
clean | damaged | mask | result | error
```

其中：

- `clean`：干净原图
- `damaged`：受损输入图
- `mask`：修复区域
- `result`：模型修复结果
- `error`：修复结果与 clean 的误差图

## 4. 多方法样本对比

当已经得到多个方法的 CSV 后，可以统一生成样本对比：

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

输出：

```text
outputs/vis/compare/
  SD-LoRA/
    best/
    middle/
    worst/
  PConv-LoRA/
    best/
    middle/
    worst/
```

## 5. 参数说明

| 参数 | 作用 |
|---|---|
| `--pred_dir` | 单个模型的修复结果目录 |
| `--clean_dir` | 干净原图目录 |
| `--mask_dir` | mask 目录 |
| `--damaged_dir` | 受损图像目录，用于生成样本拼图 |
| `--output` | 平均指标 JSON 输出路径 |
| `--detail_csv` | 逐图指标 CSV 输出路径，可不填 |
| `--vis_dir` | 样本拼图保存目录 |
| `--top_k` | best / middle / worst 各保存的样本数量 |
| `--compare_csv` | 多方法对比时传入 CSV，格式为 `方法名=CSV路径` |
| `--compare_pred_dir` | 多方法对比时传入结果目录，格式为 `方法名=结果目录` |

## 6. CSV 字段

```text
filename,psnr,ssim,masked_psnr,mask_ratio
```

| 字段 | 含义 |
|---|---|
| `filename` | 图像文件名 |
| `psnr` | 整图 PSNR |
| `ssim` | 整图 SSIM |
| `masked_psnr` | 只在 mask 区域计算的 PSNR |
| `mask_ratio` | mask 区域面积占比 |
