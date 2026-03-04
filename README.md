# MoE-DiReF

PyTorch 实现：**MoE-DiReF: Hybrid Experts-Driven Diffusion with Rectified Flow and Dynamic Correction for Semantic-Aware Image Restoration**（图像修复 / Inpainting）。

---

## 安装

```bash
pip install -r requirements.txt
```

可选（评估用）：`pip install lpips`（LPIPS）、`pip install clean-fid`（FID 与论文一致）。

---

## 快速开始

**无数据时**用合成数据跑通全流程：

```bash
# 训练（2 epoch，小图）
python train.py --synthetic --max_epochs 2 --batch_size 4 --image_size 64 --save_dir ./checkpoints

# 批量评估 + 可视化
python inference.py --checkpoint ./checkpoints/moe_diref_ep2.pt --synthetic --max_batches 4 --save_vis ./vis

# 单张推理（无 mask 则自动 50% 随机 mask）
python scripts/infer_single.py --checkpoint ./checkpoints/moe_diref_ep2.pt --image your_image.png --output result.png
```

---

## 训练

```bash
python train.py [--config configs/default.yaml] [选项]
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--config` | 主配置 | `configs/default.yaml` |
| `--override` | 覆盖配置（如 `configs/dunhuang.yaml`） | - |
| `--synthetic` | 合成数据，不读磁盘图像 | 否 |
| `--max_epochs` | 最大 epoch | 配置内 `train.epochs` |
| `--batch_size` | 批大小 | 配置内 |
| `--image_size` | 图像边长 | 配置内 `data.image_size` |
| `--save_dir` | checkpoint 目录 | `./checkpoints` |
| `--device` | 设备 | 自动 |

数据路径在配置里填 `data.root`；目录不存在时请加 `--synthetic`。  
Checkpoint 每 `save_every` 个 epoch 存为 `{save_dir}/moe_diref_ep{N}.pt`。

---

## 推理

### 批量评估（测试集 + 指标）

```bash
python inference.py --checkpoint <path>.pt [选项]
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--checkpoint` | 权重路径（必填） | - |
| `--data_root` | 测试图目录 | 配置内 `data.root` |
| `--synthetic` | 合成测试集 | 否 |
| `--batch_size` | 评估批大小 | 4 |
| `--num_steps` | ODE 步数 | 10 |
| `--save_vis` | 保存前 20 张拼接图目录 | 不保存 |
| `--max_batches` | 最多跑多少 batch | 全部 |
| `--no_fid` / `--no_lpips` | 不算 FID/LPIPS | 算 |
| `--use_clean_fid` | 用 clean-fid 算 FID | 否 |

### 单张图片

```bash
python scripts/infer_single.py --checkpoint <path>.pt --image 输入.png [--mask mask.png] [--output 输出.png]
```

- 不写 `--mask` 则自动生成随机 mask（缺失率默认 50%，可用 `--mask_ratio` 改）。
- 不写 `--output` 则输出到同目录 `*_inpaint.png`。

---

## 数据

- **自建**：图片放同一目录，支持 `.jpg/.jpeg/.png/.bmp/.webp`；在配置中设 `data.root`，按 `data.train_split` 划分训练/验证。
- **CelebA**：解压后目录含 `img_align_celeba/`，配置里 `data.name: celeba`、`data.root` 指向该目录。
- **CelebA-HQ**：图片放一文件夹，`data.name: celeba_hq`、`data.root` 指向该文件夹。
- **合成**：加 `--synthetic`，不依赖任何真实数据。

---

## 配置

主配置：`configs/default.yaml`。常用项：

| 项 | 含义 |
|----|------|
| `data.root` | 图像根目录 |
| `data.image_size` | 输入边长 |
| `data.mask_range` | mask 缺失率范围，如 [0.2, 0.8] |
| `data.train_split` | 训练集比例 |
| `model.latent_dim`、`model.num_experts` | 潜在维、专家数 |
| `train.batch_size`、`train.epochs` | 批大小、epoch |
| `train.lambda_rec`、`lambda_perc`、`lambda_adv`、`lambda_path` | 损失权重 |
| `train.save_every` | 每 N epoch 存一次 ckpt |

覆盖：`--override configs/dunhuang.yaml` 会覆盖同名字段。

---

## 评估指标

- **PSNR / SSIM**：内置，↑ 越好。
- **LPIPS**：需 `pip install lpips`，↓ 越好；未装则跳过。
- **FID**：默认用本库 Inception 特征；加 `--use_clean_fid` 则用 clean-fid（需安装），↓ 越好。

---

## 验收（可选）

无需训练好的权重，检查数据/模型/训练/指标是否正常：

```bash
python scripts/verify.py data [--synthetic]   # 数据与 I_fus
python scripts/verify.py models               # 各模块前向
python scripts/verify.py train                # 训练若干 step
python scripts/verify.py metrics              # 指标与推理
python scripts/verify.py all [--synthetic]    # 以上全部
```

---

## 项目结构

```
lclw/
├── configs/          # default.yaml, dunhuang.yaml
├── data/             # dataset, mask, transforms
├── models/           # moe_diref, mkae, rectified_flow, reme, experts, attention
├── losses/           # L_rec, L_perc, WGAN-GP
├── utils/            # config, metrics (PSNR/SSIM/FID/LPIPS)
├── scripts/          # verify.py, infer_single.py
├── train.py          # 训练入口
├── inference.py      # 批量评估入口
└── requirements.txt
```

实现细节与模块对应见 `docs/IMPLEMENTATION_PLAN.md`。
