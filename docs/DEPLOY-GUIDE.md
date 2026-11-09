# 集群部署与打包指南（2026-10-10）

本指南把 `/mnt/afs/shawn/drifting-deploy/` + `/mnt/afs/shawn/offline_bundle/` +
conda-pack 环境包，整包上传到**离线集群**后如何跑起来。

## 1. 三个上传物

| 路径 | 大小 | 内容 |
|---|---|---|
| `/mnt/afs/shawn/drifting-deploy/` | ~39MB | **代码**（含 configs / docs / paper / tests / scripts） |
| `/mnt/afs/shawn/offline_bundle/` | ~15GB | **权重与统计**（SD-VAE / VA-VAE / MAE×3 / ConvNeXt / FID+IS Inception / FID 参考统计） |
| `/mnt/afs/shawn/dist/drift_env.tar.gz` | ~4.0GB | **conda-pack 环境**（torch 2.11.0+cu13 等） |

`offline_bundle/manifest.json` 记录每一项的下载状态。当前已完成：
- ✅ SD-VAE（320MB）
- ✅ VA-VAE `vavae-imagenet256-f16d32-dinov2.pt`（2.1GB，论文默认 32ch f16）
- ✅ MAE-latent-256（1.7GB）、MAE-latent-640（11GB）
- ✅ ConvNeXt-V2 base（678MB）
- ✅ FID Inception `pt_inception-2015-12-05-6726825d.pth`
- ✅ IS Inception `weights-inception-2015-12-05-6726825d.pth`
- ⏳ **MAE-pixel-640**（约 10GB，传输中断后重试中，`manifest.json` 里标 `ok:false`）

**注意**：`mae_pixel_640` 是 pixel/CrossFlow 配置的特征提取器（`feature.mae_path: hf://mae_pixel_640`）。
若上传前仍未完成，可在网络机上单独重跑：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=0 \
python scripts/fetch_offline_assets.py --dest /mnt/afs/shawn/offline_bundle \
    --skip stats sdvae convnext fid_weights
```

## 2. 集群侧部署

```bash
# (a) 环境
mkdir -p /scratch/$USER/envs
tar -xzf drift_env.tar.gz -C /scratch/$USER/envs/drifting
export PATH=/scratch/$USER/envs/drifting/bin:$PATH
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# (b) 代码
export DRIFT_REPO=/scratch/$USER/drifting
cp -r drifting-deploy "$DRIFT_REPO"

# (c) 权重（离线包）
export DRIFT_OFFLINE_BUNDLE=/scratch/$USER/offline_bundle
# 接好所有路径（utils/offline.py 会自动读这个变量）
export IMAGENET_PATH=/scratch/$USER/imagenet     # ImageFolder: train/ + val/
export DRIFT_RUNS_DIR=$DRIFT_REPO/runs
```

**验证环境**：
```bash
cd "$DRIFT_REPO"
python -c "import utils.env as env; import os; print('HF_ROOT', os.environ['HF_ROOT']); print('SDVAE', os.environ['SDVAE_PATH'])"
```

## 3. MNIST 概念验证实验（X-Drifting vs 像素基线）

> X-Drifting 语义：**特征提取空间恒为 pixel**（drift 场在像素空间），
> latent 只是生成器的噪声先验（crossflowDiT 内部采样），监督走 pixel-MAE。
> 因此真实样本一律用像素（`use_latent: false`），模型自己从 latent 噪声出像素。

```bash
# 1. MAE 特征提取器（drift loss 的特征来源，pixel-MAE）
python main.py --config configs/mae/mnist_real_pixel_64.yaml  --workdir runs/mae_pixel --no-stamp
ln -sfn <stamp>_mae_pixel runs/mnist_real_mae_pixel_latest

# 2. 生成器对比：像素基线 vs X-Drifting（identical pixel-MAE supervision）
python main.py --gen --config configs/exp/E1_P_pixel.yaml  --workdir runs/E1_P_pixel  --no-stamp   # pixel DiT 基线
python main.py --gen --config configs/exp/X_LP_drift.yaml  --workdir runs/X_LP_drift  --no-stamp   # X-Drifting (crossflowDiT)

# 3. 事后分析（离线，无需在线查看）
python scripts/analyze_all.py --root runs
python scripts/compare_backbone.py --official /path/to/cross-flow   # 验证 backbone 公平性
```

已跑通的结果（MNIST 64px，5000 步）：见 `paper/draft.md` §4.2。两者 drift loss 收敛到
~285（Δ<1.0），MNIST 上不可分——这是预期（MNIST 10 类 64px 太简单），
真正的 latent/pixel 分离需在 ImageNet-1k/10k 上测（`docs/EXPERIMENT-DESIGN.md`）。

**断点续训**：被抢占后原样重跑，把 `--no-stamp` 换成 `--resume`：
```bash
python main.py --gen --config ... --workdir runs/LP_drift --resume
```

## 4. ImageNet-1k / ImageNet-10k

### ImageNet-1k
数据放 `$IMAGENET_PATH/{train,val}`（ImageFolder），配置用 `configs/gen/crossflow_sota_{B,L,XL}.yaml`。
注意：`feature.mae_path` 需指向 `hf://mae_pixel_640`（pixel/CrossFlow）或 `hf://mae_latent_640`（latent）。

### ImageNet-10k
本项目约定（见 `docs/EXPERIMENT-DESIGN.md` §4）：

```
<IN10K_PATH>/train/<class_id>/<image>.JPEG     # 10000 个类目录
<IN10K_PATH>/val/<class_id>/<image>.JPEG
```

- `class_id` 零填充 5 位整数（`00000`..`09999`）或原 WNID，排序稳定即可。
- 配置里 `dataset.name: imagenet10k`，`num_classes: 10000`。
- **FID 参考统计需自算**：ImageNet-10k 没有公开统计，用 `scripts/compute_fid_stats.py`
  （复用 `utils/fid_util._compute_stats`）生成 `.npz`，再设 `IMAGENET_FID_NPZ`。
- 由于无标准 FID，主张 A 的外推证据用**收敛曲线 + IS + 样本网格**。

## 5. 待办（回集群前必须做）

1. ✅ **`mae_pixel_640` 下载完成**（已转为 torch 格式）
2. ✅ **backbone 已修正为官方一致**（参数量逐字节一致 87.4M/305.6M/450.3M，见 `docs/BACKBONE-COMPARISON.md`）
   B 87.4M / L 305.6M / XL 450.3M）
3. ✅ 写 `configs/exp/` MNIST 实验配置（E1_P_pixel, X_LP_drift）
4. ⬜ ImageNet-10k 数据管道（`dataset.name: imagenet10k` + `scripts/compute_fid_stats.py`）
5. ⬜ conda-pack 重建（含 `torch-fidelity`，`scripts/pack_env.sh`）→ `dist/drift_env.tar.gz`
6. ⬜ MNIST 8 组跑通 → 产出 `runs/comparison.png` + 表 T1

## 6. 与官方 CrossFlow 的对比基线

官方仓库：`https://github.com/C-Fun/cross-flow.git`（克隆自 2026-10-10）。

关键差异（详见 `docs/BACKBONE-COMPARISON.md` + `docs/EXPERIMENT-DESIGN.md` §7）：
- backbone：官方 `crossflowDiT`（RMSNorm+SwiGLU+2D RoPE+QK-norm+零门控）vs 我们 vanilla ViT
  → **需修正**
- 目标：官方 `L_CF`（JVP + 感知 + GAN）vs 我们 **drift loss**（无 JVP）
- encoder：官方 VA-VAE 32ch（已下载，2.1GB）vs 我们 SD-VAE 4ch

## 7. 磁盘预算

| 项 | 网络机 | 集群 |
|---|---|---|
| 代码 | 39MB | 39MB |
| 权重 bundle | 15GB | 15GB |
| conda-pack 环境 | 4GB | 4GB |
| ImageNet-1k 256px | — | ~150GB |
| ImageNet-10k 256px | — | ~1.5TB |
| runs/（MNIST 8 组） | — | ~5GB |
| runs/（ImageNet 主实验） | — | ~50GB |

总预算建议预留 **2TB**（含 ImageNet-10k 与其 latent 缓存）。
