# 实验设计与布局：X-Drifting（Drift × Cross-Space）

> 论文写作与实验执行的统一蓝图。所有表格/图表编号与 `paper/main.md` 对齐。

## 1. 我们要证明什么（两条核心主张）

论文 `paper/main.md:17` 的命题：

> 一方面享受 latent 的**生成加速与压缩便利**（need proof），
> 一方面直接监督 pixel 避免重建损失且享受 pixel 的优化红利。

拆成两条**可证伪**的科学主张：

- **主张 A（latent 的红利）**：在**跨空间**设定下，从 latent 噪声出发训练生成器
  比从 pixel 噪声出发**更快收敛、更省算力、更稳**。
  *可证伪*：若 pixel 路径的收敛速度与稳定性相同，则 A 不成立。

- **主张 B（pixel 的红利）**：**直接用像素目标监督**（不做 latent 重建解码）
  在感知质量上优于"latent 监督 + 解码"，且避免了 VAE 解码引入的重建损失。
  *可证伪*：若 latent-监督 + 解码的 FID/pLPIPS 与 pixel-监督相当或更好，则 B 不成立。

**关键混淆变量**：这两个主张的"合成"就是 X-Drifting = latent 噪声 + pixel 监督。
实验必须**分轴**对照，才能把两者归因分开：

| 变体 | 噪声空间 | 监督空间 | 检验 |
|---|---|---|---|
| **P-P** | pixel | pixel | pixel 纯基线 |
| **L-L** | latent | latent（经 VAE 解码评估） | latent 纯基线（= 传统 LDM 单步） |
| **L-P (X-Drifting)** | latent | **pixel** | 我们的主张 |
| **P-L** | pixel | latent | 反向跨空间（理论可行，预期差） |

**只有 L-P 同时吃到 A 与 B**。P-P 吃不到 A，L-L 吃不到 B，P-L 都吃不到。
这张表是整个实验的骨架。

## 2. 训练目标（决定"监督空间"）

| 训练目标 | 监督空间 | 对应论文 | 我们的位置 |
|---|---|---|---|
| drift loss（Drift 论文） | 任意（本工作用 pixel） | Drift | **核心** |
| CrossFlow `L_CF`（含 `dF/dt` JVP） | pixel | CrossFlow | 对照 |
| 像素重建 + 感知 + GAN | pixel | VQGAN/RAE | 对照（CrossFlow 论文的"辅助损失"） |
| latent 重建（MSE in z） | latent | LDM/VAE | 对照（= L-L） |

**X-Drifting = drift loss + (latent 噪声 → pixel 输出)**。
这是我们相对 CrossFlow 的**核心差异**：CrossFlow 用 `dF/dt` 的 JVP 项（前向模式 AD，
2× 前向开销）+ 感知损失 + GAN；我们用 drift loss（只需反向）+ 可选的 pixel 特征器。
论文需要在 3.x 节明确这一点。

## 3. 与 CrossFlow 的对比（backbone 必须一致）

见 [docs/BACKBONE-COMPARISON.md](BACKBONE-COMPARISON.md)。

**结论**：我们当前 `CrossFlow.py` 的 backbone（vanilla ViT）与官方 `crossflowDiT`
（RMSNorm + SwiGLU + 2D RoPE + QK-norm + 零门控）**不一致**，参数量 +4%~+14%。
**必须修正**才能做公平对比。修正方案：复用 `models/generator.py` 的现代栈组件
（`LightningDiTBlock`/`SwiGLUFFN`/`RMSNorm`/`apply_rope`），只替换输入 patchify、
输出 unpatchify 与 in-context 前缀 token。

**修正后**参数量应回到：B 87.4M / L 305.6M / XL 450.3M（与官方一致）。

## 4. 数据集与尺度阶梯

按"MNIST 当 ImageNet 先验证，再逐级放大"的策略：

| 阶段 | 数据集 | 分辨率 | 类别 | 目的 | 产出 |
|---|---|---|---|---|---|
| **0. 单元/结构** | — | — | — | backbone 一致性、参数量对照 | `scripts/compare_backbone.py` |
| **1. 概念验证** | MNIST | 64px | 10 | 4×4 变体 × 2 目标在小尺度可跑通、可分胜负 | 表 T1（4 变体 × 2 目标） |
| **2. 规模验证** | CIFAR-10/100 | 32px | 10/100 | 标准 FID 基准（无 VAE 也能评） | 表 T2 |
| **3. 主结果** | **ImageNet-1k** | 256px | 1000 | 主表：与 CrossFlow/DiT/MeanFlow/iMF 对比 | 表 T3 |
| **4. 规模外推** | **ImageNet-10k** | 256px | 10000 | 类别数增加时的稳定性（主张 A 的外推） | 表 T4 |

### ImageNet-10k 结构

官方仓库 `download_imagenet.py` 只处理 ImageNet-1k，用 **ImageFolder 布局**
（`<root>/<class>/<image>.JPEG`，类名可以是 WNID 或 4 位 label index）。
ImageNet-10k（IN10K，10k 类）**没有公开的标准 split**，但社区惯例与 ImageNet-1k 同构：
按 ImageFolder 目录组织，每个类一个子目录，类数 10000。

**本项目对 ImageNet-10k 的约定**：

```
<IN10K_PATH>/train/<class_id>/<image>.JPEG    # 10000 个类目录
<IN10K_PATH>/val/<class_id>/<image>.JPEG
```

- `class_id` 用零填充 5 位整数（`00000`..`09999`）或原 WNID，**排序稳定**即可。
- `num_classes: 10000`（配置里设），`dataset.name: imagenet10k`。
- FID 参考统计需自己算（`scripts/compute_fid_stats.py`，复用 `utils/fid_util._compute_stats`）。
- 由于 ImageNet-10k 没有官方 FID 统计，**主张 A 的外推证据**主要用：
  - 训练损失收敛曲线（step vs. drift loss）的对比
  - IS / 精度（用 ImageNet-1k 的 1000 类子集做验证集代理）
  - 样本网格（视觉）
  - 若要报 FID，需同时给出 P-P / L-L / L-P / P-L 四个变体的 FID（同一参考统计）

**注意**：ImageNet-10k 与 ImageNet-1k 的类别定义不完全重叠，FID 统计不可混用。

## 5. 指标（每条主张对应哪些数字）

### 主张 A（latent 红利）需要的证据

| 指标 | 怎么测 | 期望 |
|---|---|---|
| **收敛速度** | drift loss @ 固定 step（如 10k/20k/50k）对比 L-P vs P-P | L-P 更低 |
| **算力效率** | 同 wall-clock 的 FID / 同 FID 的 wall-clock | L-P 更优 |
| **训练稳定性** | loss 方差、`g_norm`、发散次数 | L-P 更稳 |
| **参数效率** | 同 FID 下的模型参数量 | L-P 更小 |
| **latent 维度敏感度** | latent_channels ∈ {4, 32, 128}（SD-VAE / VA-VAE / FLUX.2） | 32 最优（CrossFlow 论文已证） |

### 主张 B（pixel 红利）需要的证据

| 指标 | 怎么测 | 期望 |
|---|---|---|
| **FID**（50k 样本） | 标准 FID，`cfg_list` 扫描取最优 | L-P ≈ P-P < L-L |
| **IS** | 标准 IS | L-P ≈ P-P |
| **pLPIPS / 感知距离** | 与真实样本配对的感知距离 | L-P < L-L |
| **重建损失敏感度** | 用**有损 VAE**（如 SD-VAE f8 4ch vs VA-VAE f16 32ch）训练 L-L，看 FID 退化 | L-L 对 VAE 重建质量敏感，L-P 不敏感 |
| **特征器消融** | drift loss 的特征器：MAE-pixel / MAE-latent / DINOv3 / f-loss | pixel 特征器最优（你 `main.md:31` 的提示） |

**"避免重建损失"的最强证据**（表 T5）：

> 用**同一个** L-P 模型，分别用"latent 监督 + 解码"与"pixel 监督"两种损失训练，
> 在**同一参考统计**下比较 FID。若 pixel 监督显著更优，则直接证明"解码引入重建损失"。

### 训练过程监控（离线，无人值守）

复用 `scripts/analyze_run.py`：

- `log/metrics.jsonl`：`loss`、各温度 `loss_*/`、`g_norm`、`lr`、`kimg`、`total_time`、`best_fid`、`eval/*_fid`
- `log/images/samples_train_step*.jpg`：固定种子样本网格（最直观的收敛证据）
- 训练结束自动生成 `log/report.png`（FID/loss/吞吐/样本演化四联图）+ `log/report.md`
- 批量对比：`scripts/analyze_all.py --root runs` → `comparison.png` + `index.md`

## 6. 实验矩阵与运行清单

### 矩阵（核心 8 组 = 4 变体 × 2 目标）

| # | 噪声 | 监督 | 目标 | 配置 |
|---|---|---|---|---|
| E1 | pixel | pixel | drift | `configs/exp/mnist_pp_drift.yaml` |
| E2 | latent | latent | drift | `configs/exp/mnist_ll_drift.yaml` |
| E3 | latent | **pixel** | **drift** | `configs/exp/mnist_lp_drift.yaml` ← X-Drifting |
| E4 | pixel | latent | drift | `configs/exp/mnist_pl_drift.yaml` |
| E5 | pixel | pixel | CrossFlow `L_CF` | `configs/exp/mnist_pp_crossflow.yaml` |
| E6 | latent | pixel | CrossFlow `L_CF` | `configs/exp/mnist_lp_crossflow.yaml` |
| E7 | latent | pixel | drift + DINOv3 特征 | `configs/exp/mnist_lp_drift_dino.yaml` |
| E8 | latent | pixel | drift + GAN | `configs/exp/mnist_lp_drift_gan.yaml` |

E1–E4 验证"跨空间分轴"；E5–E6 与 E3 对比"drift vs CrossFlow 目标"；
E7–E8 验证 pixel 红利的特征器/辅助损失消融。

### 阶段性产出

| 阶段 | 产出 | 表/图 |
|---|---|---|
| 1. 结构核对 | backbone 参数量对照 | 表 T0（`docs/BACKBONE-COMPARISON.md`） |
| 2. MNIST 8 组 | 收敛曲线 + 最终 FID/IS + 样本网格 | 表 T1、图 F1（`runs/comparison.png`） |
| 3. CIFAR-10/100 | 标准 FID 对比 | 表 T2 |
| 4. ImageNet-1k | 主结果 + 与 SOTA 对比 | 表 T3、图 F2（samples） |
| 5. ImageNet-10k | 类别外推稳定性 | 表 T4 |
| 6. 消融 | latent 维度、特征器、辅助损失 | 表 T5 |

## 7. 与官方 CrossFlow 的复现对齐

官方仓库已公开：`https://github.com/C-Fun/cross-flow.git`（克隆自 2026-10-10）。

**关键差异**（官方 README 明确）：

| | official | ours (X-Drifting) |
|---|---|---|
| 目标 | `L_CF` + JVP + 感知 + **无 GAN**（README:59） | **drift loss** + 可选感知/GAN |
| JVP 开销 | ~2× 前向（前向模式 AD） | **无**（纯反向） |
| 优化器 | AdamW 3e-4（论文用 Muon 8e-4） | AdamW（配置可调） |
| 训练长度 | 160 epochs, batch 2048 | 同 |
| backbone | `crossflowDiT`（RoPE+SwiGLU+vec-gate） | **待修正**为一致 |
| encoder | VA-VAE f16 32ch（默认）| 同（已下载权重） |

**论文中必须声明**：
1. backbone 与官方一致（参数量对照表 T0）；
2. 训练目标不同（drift vs CrossFlow `L_CF`）是**本文贡献**，不是实现偏差；
3. 我们的方法**不需要 JVP**（单次前向+反向），是**训练开销优势**；
4. 我们报告的 FID 用同一参考统计（JiT ImageNet-256，官方 README:128 同款）。

## 8. 论文结构建议

```
1. Introduction
   - latent vs pixel 之争（paper/main.md:2 的动机）
   - 现有方法的"同空间约束"（noise 空间 == 输出空间）
   - CrossFlow 首次打破该约束，但用 JVP + GAN 实现，训练贵
   - 我们提出 X-Drifting：drift loss 天然适配跨空间，单次前向+反向

2. Related Work
   - Latent diffusion (LDM, SD, SDXL)
   - Pixel diffusion (PixelDiT, EPG)
   - One-step: consistency / MeanFlow / iMF / CrossFlow
   - Drifting models (our base)

3. Method
   3.1 Cross-space generation 问题设定
   3.2 Drift loss 在跨空间的自然推广（核心）
   3.3 实现：X-Drifting 架构（与 CrossFlow backbone 对齐）
   3.4 训练目标：drift vs CrossFlow L_CF（对比）

4. Experiments
   4.1 概念验证（MNIST, 4 变体 × 2 目标）  ← 表 T1
   4.2 标准基准（CIFAR-10/100）            ← 表 T2
   4.3 主结果（ImageNet-1k）               ← 表 T3, 图 F2
   4.4 类别外推（ImageNet-10k）            ← 表 T4
   4.5 消融（latent 维度 / 特征器 / 辅助损失）← 表 T5

5. Discussion
   - 为什么 drift loss 天然适合跨空间
   - 训练开销对比（无 JVP）
   - 局限：像素监督的稳定性、大分辨率扩展

6. Conclusion
```

## 9. 当前待办（执行顺序）

1. ✅ 结构核对（T0）：`docs/BACKBONE-COMPARISON.md` + `scripts/compare_backbone.py`
2. ⬜ **修正 backbone** 为官方一致（复用 `models/generator.py` 现代栈）
3. ⬜ 实现 4 变体 × 2 目标的 8 组配置（`configs/exp/`）
4. ⬜ ImageNet-10k 数据管道（`dataset.name: imagenet10k`）
5. ⬜ MNIST 8 组跑通，产出 T1 + `runs/comparison.png`
6. ⬜ 论文 `paper/main.md` 按上述结构撰写
7. ⬜ 打包（conda-pack + 权重 bundle）上传集群

## 10. 立意回填到论文的语句（可直接用）

> We resolve the long-standing tension between latent and pixel generation with a
> single design: **X-Drifting** drifts a generator from a compact latent prior
> directly into pixel space, so the model enjoys both the *representation economy*
> of latent space and the *optimization fidelity* of pixel supervision. Unlike
> CrossFlow, which needs a forward-mode JVP along the latent trajectory plus
> adversarial and perceptual terms, X-Drifting inherits the drifting loss
> — an objective that natively accommodates mismatched input/output spaces —
> and trains with a single forward-and-backward pass. Empirically, X-Drifting
> converges faster and more stably than pixel-space baselines (Claim A) while
> matching or exceeding their perceptual quality without VAE reconstruction
> overhead (Claim B).
