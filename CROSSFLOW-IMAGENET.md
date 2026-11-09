# CrossFlow on ImageNet-1k —— 复现与 Drift 框架适配

按 `CrossFlow/` 下的论文源码（`chapter/method.tex`、`chapter/appendix.tex`、
`table/hyper_parameter.tex`）复现 CrossFlow 的三种 backbone 尺寸，并像其他模型一样接入
drifting 训练框架。模型实现在 [`CrossFlow.py`](CrossFlow.py)。

## 论文规格对照

论文 `table/hyper_parameter.tex` 给出的三种 ViT 尺寸（patch size 均为 16）：

| 变体 | 层数 | hidden | head 数 | 论文参数量 | 本实现参数量 |
|------|------|--------|---------|-----------|-------------|
| CrossFlow-B  | 12 | 768  | 12 | 95M  | 99.3M |
| CrossFlow-L  | 24 | 1024 | 16 | 315M | 321.9M |
| CrossFlow-XL | 28 | 1152 | 18 | 461M | 468.6M |

- 层数与 hidden 直接取自论文；**head 数论文未给出**，按 ViT 惯例 `head_dim=64` 由
  hidden 推出（768/64=12、1024/64=16、1152/64=18）。
- 参数量每档高约 4-7M（~2%），差异来自论文未完全指定的条件嵌入模块
  （label $\times$ context-token 嵌入 + time + cfg）。既有的 latent/pixel DiT 基线存在同样偏差
  （论文 LightningDiT-XL 675M，本仓库同规格实现约 677M），因此不影响与基线的可比性。
- 尺寸通过 `size: B|L|XL` 选择，也可用 `hidden_size/depth/num_heads` 显式覆盖（显式值优先）。

## 架构要点（对齐论文）

- **latent → pixel 单步映射**：输入是 VAE latent 噪声 $z_t$，输出直接是 256×256 RGB。
- **token 数必须两侧一致**：`(latent_size/latent_patch_size)² == (img_size/patch_size)²`，
  ImageNet-256 上为 16×16 = 256（latent patch 2 → 16×16；输出 patch 16 → 16×16）。
  该约束在 `CrossFlowBackbone.__init__` 里有断言保护。
- **16 个 context token** 承载类别 + 时间条件，与 256 个空间 token 拼接后过 vanilla ViT。
- **固定 sin-cos 位置编码**（非可学习），输出层把每个 token 解码成 16×16×3 像素块。

## 与 drifting 框架的适配

- `CrossFlow` 封装暴露与 `DitGen` 相同的接口：
  `model(c=labels, cfg_scale=...)["samples"]` 返回 NHWC 像素，可直接进 drift loss。
- Drift 是单步生成、无扩散时间轴，故论文的时间条件 $r$ 固定为 0（`time_embed` 退化为
  可学习常数偏置，结构保持论文原样）；框架传入的逐样本 `cfg_scale` 通过独立的 cfg 嵌入注入。
- 训练时 `model.arch: crossflow` 触发 `train.py:main_gen` 选择 CrossFlow；
  artifact 的 `metadata.json` 记录 `arch`、`size`、全部尺寸，`build_generator_from_config`
  按 `arch` 分派，可严格往返加载（`load_state_dict(strict=True)`）。
- 大模型可开 `grad_checkpoint: true` 用梯度检查点降显存（XL 配置默认开）。

## ImageNet 配置

| 配置 | 用途 |
|------|------|
| `configs/gen/crossflow_sota_B.yaml`  | CrossFlow-B，batch 2048 |
| `configs/gen/crossflow_sota_L.yaml`  | CrossFlow-L，batch 2048 |
| `configs/gen/crossflow_sota_XL.yaml` | CrossFlow-XL，batch 1024，开梯度检查点 |

CrossFlow 输出像素，因此复用**像素空间**特征提取器（`hf://mae_pixel_640` + ConvNeXt），
与 `pixel_sota_*.yaml` 一致；数据管道用 `use_latent: false`（模型内部自行采 latent 噪声），
`use_aug: true`。

```bash
python main.py --gen --config configs/gen/crossflow_sota_B.yaml --workdir runs/gen_crossflow_B
```

依赖集群环境（`IMAGENET_PATH` / `IMAGENET_CACHE_PATH` / `HF_ROOT`，见 `utils/env.py`；
Slurm 用 `scripts/slurm/common.sh`）。

### 关于 VA-VAE 与 latent 通道

论文用 **VA-VAE 32 通道** latent（`latent_channels: 32`）。本仓库的 latent 缓存与
`hf://mae_latent_640` 基于 **SD-VAE 4 通道**（`latent_channels: 4`）。配置里默认给了 4，
若集群上提供 VA-VAE latent，把 `latent_channels` 改成 32 并保持
`latent_size`/`latent_patch_size` 使 token 数仍为 256 即可（`latent_size=32, latent_patch_size=2`
在 32 通道下依然成立）。

## 测试

```bash
pytest -q   # 15 项，含 tests/test_crossflow.py 的尺寸预设/接口/往返 7 项
```
