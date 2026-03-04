# MoE-DiReF 代码实现方案

基于论文《MoE-DiReF: Hybrid Experts-Driven Diffusion with Rectified Flow and Dynamic Correction for Semantic-Aware Image Restoration》的完整实现方案。

---

## 一、整体架构与目录规划

```
lclw/
├── configs/                    # 配置
│   ├── default.yaml            # 默认训练/推理配置
│   └── dunhuang.yaml           # 敦煌壁画场景配置（可选）
├── data/                       # 数据相关
│   ├── __init__.py
│   ├── dataset.py              # CelebA / CelebA-HQ / Dunhuang 数据集
│   ├── mask.py                 # 随机 mask 生成（20%–80% 缺失）
│   └── transforms.py          # 图像增强与归一化
├── models/                     # 模型主体
│   ├── __init__.py
│   ├── moe_diref.py            # 主框架：双分支 + REME 串联
│   ├── mkae.py                 # Multi-expert Knowledge Augmented Encoder
│   ├── rectified_flow.py       # Rectified Flow 扩散（ODE + 速度场）
│   ├── reme.py                 # REME：DoFe + UpFu + 三阶段校正
│   ├── experts.py              # 专家网络与门控
│   └── attention.py            # CnAM, AdaAttn, SAM, CAM, GAT
├── losses/                     # 损失函数
│   ├── __init__.py
│   ├── reconstruction.py       # L_rec (L2)
│   ├── adversarial.py          # L_adv (WGAN-GP)
│   └── perceptual.py          # L_perc (VGG)
├── train.py                    # 训练入口
├── inference.py                # 推理/修复入口
├── requirements.txt
└── README.md
```

---

## 二、核心模块实现要点

### 2.1 数据与 Mask（对应论文 IV-A）

- **数据集**：CelebA、CelebA-HQ、自建敦煌壁画；图像 resize 到统一分辨率（如 256/512）。
- **Mask**：二值 mask，1 表示损坏区域；随机生成 20%–80% 缺失率，形状可不规则（矩形/随机形状）。
- **输入构造**：`I_fus = I_in ⊙ (1−M) + M⊙ε`（ε 为高斯噪声），与论文 Eq.12 一致。

**实现文件**：`data/dataset.py`、`data/mask.py`。

---

### 2.2 MKAE（Multi-expert Knowledge Augmented Encoder）

对应论文 III-B，公式 (1)–(4)。

| 子模块 | 功能 | 实现要点 |
|--------|------|----------|
| 多尺度特征金字塔 | P = {φ_l(I), φ_m(I), φ_h(I)} | 三个编码分支：1×1、3×3、5×5 卷积，输出多分辨率特征并拼接/堆叠 |
| 门控网络 G(·) | 输入 I⊕ε，输出 N 维权重，和为 1 | 小 MLP 或轻量 CNN + Softmax，ε 为可选噪声 |
| N 个专家 | Experti(P) | 每个专家为 ResBlock 或小型 UNet 块，输入多尺度 P，输出特征 |
| 专家融合 | f_fusion = Σ w_i · Experti(P) | 按门控权重加权求和 |
| 内容–风格解耦 | z_c = CnAM(f), z_s = AdaAttn(f) | CnAM：内容感知注意力；AdaAttn：自适应注意力，输出 style 向量 |

**实现文件**：`models/mkae.py`、`models/experts.py`、`models/attention.py`。

---

### 2.3 Rectified Flow（潜在空间扩散）

对应论文 III-C，公式 (5)–(6)。

- **前向**：在潜在空间采样 (z_0, z_1)，z_t = (1−t)z_0 + t·z_1（线性插值）。
- **速度场**：网络 v_θ(z_t, t)，目标为 v_θ(z_t, t) ≈ z_1 − z_0。
- **Path consistency loss**：`L_path = E_{t~U(0,1)} [ ||v_θ(z_t, t) − (z_1 − z_0)||^2 ]`。
- **采样**：ODE 求解器（Euler/Heun），从 z_1 积分到 z_0，步数可设为 10（对应论文 80% 减步）。

潜在 z 可由 VAE encoder 得到，或简化为在像素/下采样空间做 Rectified Flow。

**实现文件**：`models/rectified_flow.py`。

---

### 2.4 REME（DoFe + UpFu + 三阶段校正）

对应论文 III-D。

**DoFe（下采样特征提取）**  
- 输入：I_fus、M。  
- 三支路：`f_stride = Conv_stride(I_fus)`，`f_max = MaxPool(I_fus)`，`f_avg = AvgPool(I_fus)`。  
- 门控融合：`f_DoFe = σ(M)⊙f_stride + (1−σ(M))⊙(f_max + f_avg)`（Eq.13–14）。

**UpFu（上采样融合单元）**  
- 输入：DoFe 高维特征 f_high，去噪后的低分辨率特征 f_denoised_low。  
- 对齐：DeformConv/STN 得到 Δp，`f_aligned = STN(f_high, Δp)`（Eq.7–8）。  
- 多注意力：`f_up_fusion = Conv(SAM(f_aligned); CAM(f_aligned); GAT(f_aligned))`（Eq.9–11）。  
- 约束低分辨率图：`x_constrained_low = x_denoised_low + γ·UpSample(f_up_fusion)`（Eq.10）。

**三阶段校正**  
1. 输入融合 + DoFe 得到 f_DoFe。  
2. f_DoFe 送入 MKAE，得到 z_c、z_s。  
3. 潜在约束：`z_c_t = z_t + λ·(z_c + z_s)`（Eq.15），用于扩散每步的 z_t。

**实现文件**：`models/reme.py`（含 DoFe、UpFu、三阶段逻辑），注意力在 `models/attention.py`。

---

### 2.5 总损失与训练策略（对应 III-E）

- **L_total = λ_rec·L_rec + λ_adv·L_adv + λ_perc·L_perc**
  - L_rec：`||I_gt − I_pred||_2^2`
  - L_adv：WGAN-GP（G 为 MoE-DiReF，D 为判别器）
  - L_perc：VGG 多层特征 L1
- **若做潜在扩散**：可加 L_path（Rectified Flow）到 L_total。
- **训练细节**：Adam (β1=0.9, β2=0.999)，batch_size=16，cosine 学习率 η_max=2e-4、η_min=5e-6，T=2000 epochs（论文 Eq.20）。

**实现文件**：`losses/` 下各损失；`train.py` 中组合 L_total 与优化器/调度器。

---

## 三、实现顺序建议

| 阶段 | 内容 | 产出 |
|------|------|------|
| 1 | 项目骨架、config、data/mask/dataset | 能跑通 dataloader 与 mask |
| 2 | attention（CnAM, AdaAttn, SAM, CAM, GAT） | 可单测注意力模块 |
| 3 | experts + 门控 + MKAE（多尺度 + 路由 + 内容风格解耦） | 单测 MKAE 前向 |
| 4 | Rectified Flow（速度场、L_path、ODE 采样） | 单测 RF 训练与采样 |
| 5 | DoFe + UpFu + REME 三阶段 | 与 MKAE/RF 对接 |
| 6 | 主模型 MoE-DiReF（双分支 + REME） | 端到端前向 |
| 7 | L_rec / L_perc / L_adv + 训练循环 | 小图训练可复现 |
| 8 | 推理脚本、指标（PSNR/SSIM/FID/LPIPS） | 复现论文表格与可视化 |

---

## 四、技术选型建议

- **框架**：PyTorch。
- **潜在空间**：若复现“潜在空间修复”，可接预训练 VAE（如 SD 的 VAE 或 LDM）；为简化可先用像素空间 Rectified Flow。
- **DIT**：论文提到 Diffusion Transformer (DIT)；可先用 U-Net 型扩散 backbone，再替换为 DiT 以更贴近原文。
- **DeformConv**：PyTorch 无官方 DeformConv，可用 `torchvision.ops.deform_conv2d` 或第三方实现。
- **GAT**：图注意力在图像上需先定义图结构（如 patch 或超像素），或简化为 channel-wise 注意力。

---

## 五、配置项示例（configs/default.yaml）

```yaml
data:
  name: celeba_hq
  root: /path/to/celeba_hq
  image_size: 256
  mask_range: [0.2, 0.8]

model:
  latent_dim: 64
  num_experts: 4
  rf_steps_train: 50
  rf_steps_infer: 10

reme:
  dofe_branches: [stride, maxpool, avgpool]
  upfu_use_stn: true

train:
  batch_size: 16
  epochs: 2000
  lr_max: 2e-4
  lr_min: 5e-6
  lambda_rec: 1.0
  lambda_adv: 0.1
  lambda_perc: 0.1
```

---

## 六、验收标准（与论文对齐）

1. **结构**：双分支（结构保持 + 生成）+ REME 三阶段校正，数据流与论文图 1 一致。  
2. **指标**：在 CelebA-HQ 上 SSIM≈0.985、FID≈2.30、LPIPS≈0.014（允许小幅波动）。  
3. **效率**：采样步数 10 步时与 50 步 DDPM 可比或更优。  
4. **消融**：可关闭 MKAE/REME/Rectified Flow 中某一项，复现 Table III/IV 趋势。

---

若确认该方案，下一步可从 **阶段 1（项目骨架 + 数据 + mask）** 开始写具体代码；你也可以指定先做某一模块（如先 MKAE 或先 Rectified Flow）。
