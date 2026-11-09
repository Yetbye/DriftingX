# Drifting MNIST 实验指南（conda 环境 + 镜像安装）

本目录在 `lambertae-drifting` 的 PyTorch 移植基础上增加了 MNIST 支持
（`dataset/dataset.py` 中的 `dataset.name: mnist` 分支），用于对比三条生成路径：

- **latent(VAE) 模式**：图像经 SD-VAE（`stabilityai/sd-vae-ft-mse`，已下载到 `hf_cache/sdvae`）
  编码为 latent，DiT 在 latent 空间做 drift 生成，再解码回像素。
- **pixel-space 模式**：DiT 直接在像素空间做 drift 生成。
- **CrossFlow 模式**：CrossFlow 以 VAE latent 噪声为输入，直接回归出像素图
  （融合 latent 与 pixel）：`CrossFlow.py`。

三条路径各自需要先训好一个 MNIST 上的 MAE 特征提取器（drift loss 的特征来源）；
CrossFlow 输出像素，复用 pixel MAE 特征提取器。

## 环境

```bash
conda activate drifting   # python 3.12, torch 2.11.0+cu130（torch 栈走 conda 清华源）
cd ~/drifting
export SDVAE_PATH=$PWD/hf_cache/sdvae   # 本地 SD-VAE，运行时不访问 HuggingFace
export CUDA_VISIBLE_DEVICES=0
```

conda/pip 均已配置清华镜像；MNIST 来自 ossci 官方镜像源（`scripts/download_mnist.py`，
含 MD5 校验，落盘于 `data/mnist/MNIST/raw/`）。
环境重建脚本：`scripts/rebuild_conda_env.sh`。

## 训练命令

```bash
# 1. MAE 特征提取器（latent + pixel），各 5000 步
python main.py --config configs/mae/mnist_real_latent_64.yaml --workdir runs/mnist_real_mae_latent
python main.py --config configs/mae/mnist_real_pixel_64.yaml  --workdir runs/mnist_real_mae_pixel
ln -sfn <stamp>_mnist_real_mae_latent runs/mnist_real_mae_latent_latest
ln -sfn <stamp>_mnist_real_mae_pixel  runs/mnist_real_mae_pixel_latest

# 2. 生成器（drift loss），各 10000 步；--gen 切换
python main.py --gen --config configs/gen/mnist_real_latent_64.yaml   --workdir runs/mnist_real_gen_latent
python main.py --gen --config configs/gen/mnist_real_pixel_64.yaml    --workdir runs/mnist_real_gen_pixel
python main.py --gen --config configs/gen/mnist_real_crossflow_64.yaml --workdir runs/mnist_real_gen_crossflow
```

CrossFlow 的尺寸约束（配置里已满足）：latent token 数必须等于输出像素 token 数
`(latent_size / latent_patch_size)**2 == (img_size / patch_size)**2`。
MNIST 64px 图像 + 8×8 SD-VAE latent → 两侧都是 64 个 token。

## 结果对比（2026-09-13 / 09-16，均 10000 步）

| 生成路径 | 参数 | run 目录 | drift loss |
|----------|------|----------|------------|
| latent(VAE) | 39.2M | `runs/0913_1131_mnist_real_gen_latent` | 350.0 → 286.0 |
| pixel-space | 39.4M | `runs/0913_1131_mnist_real_gen_pixel` | 337.6 → 284.5 |
| CrossFlow | 26.1M | `runs/0916_0716_mnist_real_gen_crossflow` | 313.7 → 285.3 |

三条路径最终 loss 与样本质量相当（CrossFlow 参数量更小，从 latent 噪声一步回归像素）。
并排对比图（loss 曲线 + 样本网格）：`runs/compare_mnist.png`。

特征提取器：MAE(latent) recon 6.20 → 0.18；MAE(pixel) recon 187.2 → 3.63。

每个 run 的产物：`log/overview.png`（loss 曲线 + 样本网格）、`log/images/`（训练中样本）、
`log/metrics.jsonl`、`params_ema/`（EMA 权重，推理/评估用）。
`pytest -q`（8 项单元测试）全部通过。

## 可视化

```bash
python scripts/visualize_run.py runs/0913_1131_mnist_real_gen_latent   # 生成 log/overview.png
```

## 对上游代码的改动

1. `dataset/dataset.py`：新增 `_build_mnist_dataset`，`create_imagenet_split` 接受 `dataset_name`。
2. `utils/model_builder.py`：按 `config.dataset.name` 路由数据集，标签名动态。
3. `utils/env.py`：新增 `MNIST_PATH`（默认 `data/mnist`）。
4. `dataset/vae.py`：无本地缓存时默认走 `HF_ENDPOINT=https://hf-mirror.com`。
5. `models/mae_model.py`：修复非 1000 类 MAE 工件加载崩溃（从 `fc.weight` 推断 `num_classes`）。
6. `CrossFlow.py` + `models/generator.py:build_generator_from_config` + `train.py:main_gen`：
   新增 `model.arch: crossflow` 分支，接入 CrossFlow 生成器。
