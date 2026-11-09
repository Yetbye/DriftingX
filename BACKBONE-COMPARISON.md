# CrossFlow Backbone 对比：我们的实现 vs 官方仓库

来源：官方仓库 `https://github.com/C-Fun/cross-flow.git`（下称 **official**），
克隆自 2026-10-10，`models/crossflowDiT.py` 为主干实现。

## TL;DR

**参数量**：

| 变体 | official (`crossflowDiT_*_2`) | ours (`CrossFlow.py`) | Δ |
|---|---|---|---|
| B  (depth 12, hidden 768)  | **87.4M** | 99.4M | **+13.7%** |
| L  (depth 24, hidden 1024) | **305.6M** | 322.0M | **+5.4%** |
| XL (depth 28, hidden 1152) | **450.3M** | 468.7M | **+4.1%** |

depth / hidden / heads 都对齐了论文（12/24/28，768/1024/1152，12/16/16），但**内部组件不同**，
导致参数量偏大、表达力更强。**我们的结果目前与 official 不可直接对比**，需要修正。

## 结构差异（根因）

### 1. 模块栈：vanilla ViT vs 现代 ViT

| 组件 | official | ours | 影响 |
|---|---|---|---|
| 归一化 | **RMSNorm**（每 block 2 个 + final 1 个） | **LayerNorm** | 无 |
| 激活 | **SwiGLU**（w1/w3 门控，`mlp_ratio=8/3`） | **SiLU MLP**（`mlp_ratio=4.0`） | 参数 + 表�达力 |
| 位置编码 | **2D RoPE**（仅作用于图像 patch token，实数 cos/sin） | **固定 sin-cos 加法 PE** | 表达力 |
| 注意力归一 | **QK RMSNorm** | 无 QK-norm | 稳定性 |
| 残差 | **零初始化向量门控**（`attn_scale`/`mlp_scale`，`zeros(1,h)`） | 标准残差 | 收敛初期稳定性 |
| Patch embed | **BottleneckPatchEmbedder**（conv patch→PCA 128/256→1×1 conv） | 单层 `nn.Linear` | 参数 + 局部归纳偏置 |
| 输出头 | `RMSNorm + zero-init Linear` | `LayerNorm + Linear`（tanh） | 参数 |

### 2. 条件注入：token 数与方式不同

| | official | ours |
|---|---|---|
| class 条件 | 8 个可学习 class token（`num_class_tokens=8`）+ label embedding | 16 个 context token（`num_context_tokens=16`） |
| 时间条件 | 4 个可学习 time token + `r` embedding | 同一组 context token + 广播 time embedding |
| CFG 条件 | 无（label dropout + 像素空间 CFG） | 每样本 `cfg_scale` 标量经 MLP 注入 |
| 位置编码 | 可学习 `pos_embed`（`total_tokens = 256+8+4`） | 固定 sin-cos（仅 256 个空间 token） |
| 输出 | 丢弃前缀 token 后 unpatchify | 丢弃 context token 后 unpatchify |

### 3. 头数假设错误

official XL 的 `num_heads=16`（`models/crossflowDiT.py:380`），我们按 `head_dim=64` 推成 18。
这导致 XL 的 head_dim 变成 64（official 是 72），参数与注意力计算都不同。
L 的 16 和 B 的 12 与 official 一致，只有 XL 不一致。

### 4. Patch 嵌入维度（隐性参数开销）

official `BottleneckPatchEmbedder` 用两层 conv：`C_in·p²→128→hidden`（`pca_channels=128` for B/L，`256` for H/G）。
我们的单层 Linear 是 `C_in·p²→hidden`。对 `latent_channels=4, p=2`（输入侧），两者的 patch 维度都是 16，
所以输入侧参数接近；但 **输出侧** 我们是 `hidden→p²·3` 单层，official 是 `hidden→p²·3` 单层 + RMSNorm，差异很小。
真正的参数差距主要来自 SwiGLU（3 个矩阵 vs 2 个）与 MLP 比例。

## 与 drifting 论文的关系（重要）

**drifting 论文 `paper.tex:1236` 写明**：

> We adopt a DiT-style Transformer. Following yao2025reconstruction, we use **SwiGLU, RoPE, RMSNorm, and QK-Norm**.

这与 **official CrossFlow 完全一致**（现代 ViT 栈）。我们仓库的 `models/generator.py:LightningDiT`
**也实现了这套栈**（`use_swiglu/use_rope/use_rmsnorm/use_qknorm` 全部支持，含 RMSNorm、apply_rope、SwiGLUFFN）。

**只有 `CrossFlow.py` 这一个文件是 vanilla ViT**。所以这不是「Drift 的 backbone vs CrossFlow 的 backbone」
的差异，而是**我们把 CrossFlow 接进来时用了一个简化 backbone**，与两边论文描述的栈都不一致。

**修正方向（推荐）**：让 `CrossFlow.py` 复用 `models/generator.py` 的 `LightningDiTBlock` /
`SwiGLUFFN` / `RMSNorm` / `apply_rope`，只替换：
1. 输入侧：latent patchify（`BottleneckPatchEmbedder` 或等价 conv patch）
2. 输出侧：pixel unpatchify head
3. 条件：in-context prefix token（class + time），而非 adaLN-zero

这样 backbone 与 official 逐层对应，参数量回到 87.4M/305.6M/450.3M，**结果可与 official 与 drifting 论文公平对比**。

## 对实验设计的影响

- **必须修正**：否则论文里"与 CrossFlow 公平比较"的主张站不住。
- 修正后重跑 MNIST 的三条路径（latent / pixel / crossflow）作为 sanity，再上 ImageNet-1k/10k。
- 论文中需要**声明 backbone 一致**（引用 official 仓库的 `crossflowDiT`），并给出参数量对照表。
- 保留 vanilla-ViT 版本作为**消融**（vanilla vs modern stack），可单独论证现代栈的收益。

## 验证方法

```bash
# 跑这个脚本对比两边参数量与模块栈
python scripts/compare_backbone.py
```
