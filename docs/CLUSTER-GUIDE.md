# 集群使用指南（离线 / 无交互终端）

目标：在**断网、无终端交互**的集群上跑 ImageNet-256 的 Drift / CrossFlow 正式实验，
所有数据与中间产物落盘保存，事后可完整复盘。

本指南按你实际要做的动作顺序排列：**准备离线包 → 上传 → 放数据 → 跑训练 → 中途与事后验证 → 续训 → 调超参**。

---

## 0. 一次性准备（在有网的机器上）

三件事：**打包环境**、**打包离线资源**、**拉代码**。

```bash
cd <repo>

# (a) 环境：优先 docker；没有 docker 权限就用 conda-pack 回退
bash scripts/build_docker.sh          # 产出 dist/drift__latest.tar（需可用 docker daemon）
bash scripts/pack_env.sh              # 产出 dist/drift_env.tar.gz（无需 docker/root）

# (b) 离线资源：VAE / MAE / ConvNeXt / FID-IS 权重
python scripts/fetch_offline_assets.py --dest ./offline_bundle \
    --fid-stats /path/to/imagenet_256_fid_stats.npz
```

> ⚠️ **本机当前状态（重要）**：这台机器能访问 GitHub 和清华 pypi，但**访问不了 HuggingFace**
> （`huggingface.co`/`hf-mirror.com` 均超时）。因此上面 (b) 里的 **MAE (`mae_latent_640` 等)
> 与 ConvNeXt 权重需要在一台能上 HF 的机器上补拉**；SD-VAE 与 FID/IS Inception 权重本机已可获取。
> `fetch_offline_assets.py` 会把每一项的成败写进 `offline_bundle/manifest.json`，按里面的
> `ok: false` 项补齐即可，不必重跑全部。

### 离线包目录结构（`fetch_offline_assets.py` 产出）

```
offline_bundle/
  hf_root/
    models/mae/pytorch/<name>/{metadata.json,ema_params.pt}   # MAE 特征提取器
    convnext/convnextv2-base-22k-224/                          # pixel/CrossFlow 额外特征
  sdvae/{config.json,diffusion_pytorch_model.safetensors}      # SD-VAE f8
  torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth   # FID
  torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth  # IS
  fid/imagenet_256_fid_stats.npz                               # FID 参考统计
  imnet_stats/imagenet_val_prc_arr0.npz                        # 可选：precision/recall
  manifest.json
```

上传：`dist/*.tar`、`dist/drift_env.tar.gz`、`offline_bundle/`、整个 repo。

---

## 1. 集群侧：环境怎么用

### 方式 A：docker（推荐，如果集群允许）

```bash
docker load -i drift__latest.tar
docker images | grep drift
```

镜像里只含**代码 + conda 环境**；数据、离线包、输出全部用 volume 挂进去（见第 2 节）。
容器入口是 `drift-train`，它负责：接好离线包路径 → 检查数据存在 → 自动选择单卡/分布式启动器 → 跑 `main.py`。

### 方式 B：conda-pack（无 docker 权限时）

```bash
mkdir -p /scratch/$USER/envs
tar -xzf drift_env.tar.gz -C /scratch/$USER/envs/drifting
source /scratch/$USER/envs/drifting/bin/activate     # 或直接用绝对路径的 python
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**非交互场景要点**：不要依赖 `conda activate`，直接用绝对路径调用解释器即可，
例如 `/scratch/$USER/envs/drifting/bin/python main.py ...`。

---

## 2. 数据放哪里

代码通过环境变量定位数据（见 `utils/env.py`），**不写死路径**：

| 变量 | 含义 | 必需性 |
|---|---|---|
| `IMAGENET_PATH` | ImageFolder 根，下面必须有 `train/` 与 `val/` 两个子目录 | 所有 ImageNet 实验必需 |
| `IMAGENET_CACHE_PATH` | 预编码 latent 缓存根（同样 `train/`、`val/`） | 仅 `use_cache: true` 的 latent 配置需要 |
| `DRIFT_OFFLINE_BUNDLE` | 离线资源包根目录 | 断网集群必需 |
| `DRIFT_RUNS_DIR` | 输出根 | 建议挂大容量卷 |

设了 `DRIFT_OFFLINE_BUNDLE` 之后，`HF_ROOT / SDVAE_PATH / DRIFT_CONVNEXT_PATH /
IMAGENET_FID_NPZ / TORCH_HOME` 会**自动**指向包内对应文件（`utils/offline.py`），
无需逐个设置；显式设置的单项变量优先，所以本地开发不受影响。

**latent 缓存构建**（若用 latent 配置，需先在能看到数据的地方生成）：

```bash
python -m dataset.latent \
  --data-path "$IMAGENET_PATH" --target-path "$IMAGENET_CACHE_PATH" \
  --local-batch-size 128 --num-workers 8 --pin-memory
```

> CrossFlow 与 pixel 配置用 `use_latent: false`（模型内部自采 latent 噪声），**不需要** latent 缓存。

---

## 3. 训练命令

### docker 一条命令（单机 8 卡）

```bash
docker run --rm --gpus all --ipc=host --shm-size=64g \
  -e NPROC_PER_NODE=8 \
  -e DRIFT_OFFLINE_BUNDLE=/bundle \
  -e IMAGENET_PATH=/data/imagenet \
  -v $BUNDLE:/bundle:ro \
  -v $IMAGENET:/data/imagenet:ro \
  -v $RUNS:/workspace/drifting/runs \
  drift:latest \
  drift-train --gen --config configs/gen/crossflow_sota_B.yaml \
             --workdir runs/crossflow_B
```

先做一次干跑检查（打印 torch/路径后退出）：

```bash
docker run --rm --gpus all -e DRIFT_DRY_CHECK=1 \
  -e DRIFT_OFFLINE_BUNDLE=/bundle -e IMAGENET_PATH=/data/imagenet \
  -v $BUNDLE:/bundle:ro -v $IMAGENET:/data/imagenet:ro -v $RUNS:/workspace/drifting/runs \
  drift:latest drift-train --config configs/gen/crossflow_sota_B.yaml --workdir runs/_check
```

### SLURM

用模板 `scripts/slurm/docker_gen_crossflow_B.sbatch`（内含容器启动）；无 docker 的
集群可参照现有 `scripts/slurm/gen_1node_smoke.sbatch`，把启动行换成绝对路径 python。

```bash
sbatch scripts/slurm/docker_gen_crossflow_B.sbatch
```

### 可用配置

| 配置 | 说明 |
|---|---|
| `configs/gen/crossflow_sota_{B,L,XL}.yaml` | CrossFlow B/L/XL（latent→pixel 单步） |
| `configs/gen/latent_sota_{B,L}.yaml` | DiT latent 空间 |
| `configs/gen/pixel_sota_{B,L}.yaml` | DiT 像素空间 |
| `configs/mae/{latent,pixel}_640.yaml` | MAE 特征提取器（drift loss 的特征来源） |

---

## 4. 中途怎么验证（无终端、无人值守）

**所有监控都落盘，不需要在线查看。** 训练过程中自动写入：

| 文件 | 内容 | 频率 |
|---|---|---|
| `log/metrics.jsonl` | 标量曲线：`loss`、各温度 `loss_*/`、`g_norm`、`lr`、`kimg`、`total_time`、`best_fid`、`eval/*_fid`、`eval/*_isc_mean` | 每 `logging.log_every_k` 步 |
| `log/images/samples_train_step*.jpg` | 固定种子的样本网格（EMA 模型），**最直观的"有没有学会"证据** | 每 `train.train_sample_per_step` 步 |
| `log/run.json` | 主机 / SLURM / 环境快照，用于对齐实验 | 启动一次 |
| `log/config.yaml` | 解析后的完整超参快照 | 启动一次 |
| `checkpoints/step_*.pt` | 完整训练态（model + EMA + optimizer + RNG） | 每 `save_per_step` 步 |
| `params_ema/` | 仅 EMA 权重的交付工件（含 `metadata.json`） | 同上 |

另外**训练结束（或崩溃）时会自动生成**：

- `log/report.png`：FID/IS 曲线、loss 曲线、吞吐、样本演化拼图
- `log/report.md`：最终指标、最佳 FID/cfg、逐步 FID 表、配置摘要

这套图/表的设计依据论文：Drift 论文看 FID(50k)+IS 与样本质量，CrossFlow 论文看
1-NFE 的 FID 轨迹与样本；两者的共同关键是**样本网格随训练演化的趋势**，所以样本拼图与
FID 曲线是必看项。

中途 `run_eval: true` 时还会每 `eval_per_step` 步跑一次 FID（`cfg_list` 里每个 CFG 各一次，
取最好），并按 `eval_samples` 采样。**注意 FID 跳步成本高**，长跑建议：
首个 checkpoint 后就设 `eval_on_step_one: true` 做一次 500 样本的 sanity FID，全量 50k 只在
关键节点跑。

---

## 5. 事后分析（集群上或拷回本地）

```bash
# 单个 run：生成 log/report.png + report.md + samples_evolution.png
python scripts/analyze_run.py runs/<stamp>_crossflow_B

# 批量：所有 run 的汇总对比图 + 索引表
python scripts/analyze_all.py --root runs
```

`analyze_all.py` 产出 `runs/comparison.png`（loss 曲线叠加 + best-FID 条形图）和
`runs/index.md`（每行一个 run：arch/size/空间/步数/最终 loss/最佳 FID/cfg），
适合多组实验横向对比。单 run 的 `report.md` 里有逐步 FID 表。

---

## 6. 断点续训（重点：离线长跑必须会）

**已支持，且做了防抢占加固。** 关键是让多次提交指向**同一个 run 目录**。默认行为会给
`--workdir` 打时间戳，所以续训必须显式关掉打戳：

```bash
# 首次运行：固定目录，不要时间戳
python main.py --gen --config configs/gen/crossflow_sota_B.yaml \
    --workdir runs/crossflow_B --no-stamp

# 被抢占/掉线后：同一目录 + --resume（目录里没有 checkpoint 会直接报错，避免误开新 run）
python main.py --gen --config configs/gen/crossflow_sota_B.yaml \
    --workdir runs/crossflow_B --resume
```

`train.py` 会自动从 `<workdir>/checkpoints/` 里最新的 checkpoint 恢复
model / EMA / optimizer / step（以及 RNG 状态）。

三个必须知道的点：

1. **不要**在续训时改动 `--workdir` 指向，否则会从零开始。`--resume` 会在目录无 checkpoint 时
   立刻失败，这正是为了拦住这种情况。
2. 已带时间戳的目录名（形如 `0917_0751_xxx`）会被原样复用，不会被二次打戳；但正规做法是用
   `--no-stamp` + `--resume` 显式控制。
3. 恢复时样本队列会按 `train.push_at_resume × push_per_step` 重新填充（Drift 的
   positive/negative memory bank 不在 checkpoint 里），这是论文实现的原生行为，前几十步 loss
   可能有小幅波动，属于正常现象。

### 本次为防抢占做的加固（`utils/ckpt_util.py` + `utils/misc.py`）

- **原子写**：checkpoint 与 EMA 工件先写 `.tmp.<pid>` 再 `os.replace`，中途被 kill 不会留下半截文件。
- **坏文件回退**：读取失败（截断/损坏）时自动跳到次新的可读 checkpoint，单个坏文件不会毁掉长跑。
- **RNG 状态**：checkpoint 额外保存 torch/cuda/numpy/python RNG，续训随机性更接近连续训练。
- **打戳不再重复**：`--no-stamp` / `--resume` 保证同一目录被复用，杜绝"续训变新 run"。
  （这个 bug 在验证时被实测抓到过：原实现会把 `runs/x` 变成 `runs/<新戳>_<旧戳>_x` 并从 step 0 重训。）

---

## 7. 速度与显存优化

已实现/可用的开关（按收益排序）：

| 手段 | 位置 | 效果 | 备注 |
|---|---|---|---|
| `hsdp_dim` | 配置顶层 | 分片数据并行，跨卡切参数/优化器态 | 8 卡设 8 |
| `compile: 0/1/2` | 配置顶层 | `2`=torch.compile(fullgraph) | 首次编译慢，长跑划算 |
| `use_bf16: true`, `attn_fp32: true` | DiT `model` | bf16 计算、注意力保 fp32 稳 | DiT 配置默认开 |
| **`use_remat: true`** | DiT `model` | 梯度检查点，**本次修复为真正生效** | 约 -30% 速度换大显存节省 |
| `grad_checkpoint: true` | CrossFlow `model` | 同上（CrossFlow 侧） | XL 配置默认开 |
| `num_workers` / `prefetch_factor` / `pin_memory` | `dataset.kwargs` | 数据加载吞吐 | pixel 配置建议 `pin_memory: true` |
| batch/`train_batch_size` / `gen_per_label` | `train` | 直接决定显存 | 先降 `gen_per_label` 再降 batch |

实测参考（A100 40G，256px）：

- DiT（h384/d8，256 token）：`use_remat=True` 显存 0.54 → **0.27 GB**，数值一致。
- CrossFlow-XL：`grad_checkpoint` 开时 batch 32 峰值约 **4.7 GB**，显存很宽裕；
  B/L 关掉即可。

> 修复说明：`use_remat` 此前在 `LightningDiT` 里是 `del use_remat` 的**空操作**，
> 配置写了也不起作用。现在真的对每个 block 做 `torch.utils.checkpoint`，
> 并只在训练+需要梯度时启用（推理/EMA 前向不额外开销）。

---

## 8. 超参数含义与管理

配置分五段，语义与论文的对应关系如下（完整示例见 `configs/gen/crossflow_sota_B.yaml`）。

### `dataset`
| 键 | 含义 |
|---|---|
| `resolution` | 图像边长，ImageNet 用 256 |
| `use_latent` / `use_cache` | 训练空间：latent / 预缓存 latent / 像素。CrossFlow 与 pixel 用 `false/false` |
| `use_aug` | 是否训练时增强（pixel 配置 `true`） |
| `num_classes` | 类别数（ImageNet 1000，MNIST 10） |
| `batch_size` / `eval_batch_size` | 数据加载批大小（会按 world size 均分） |
| `name` | 数据集路由（`imagenet` / `mnist`） |

### `model`（生成器）
DiT：`cond_dim`、`input_size`、`in_channels`、`patch_size`、`hidden_size`、`depth`、
`num_heads`、`mlp_ratio`、`use_*`、`n_cls_tokens`、`noise_classes/coords`；
CrossFlow：`arch: crossflow`、`size: B|L|XL`、`img_size`、`patch_size`、`latent_channels`、
`latent_size`、`latent_patch_size`、`num_context_tokens`、`grad_checkpoint`。
**约束**：`(latent_size/latent_patch_size)² == (img_size/patch_size)²`（两侧 token 数一致）。
B/L/XL 的 depth/hidden 严格按论文（12/768、24/1024、28/1152）。

### `optimizer`
`lr_schedule.{learning_rate,warmup_steps,lr_schedule,total_steps}`、`weight_decay`、
`adam_b1/adam_b2`。论文 Drift 用 AdamW(0.9,0.95)、latent 4e-4 / pixel 2e-4、warmup 10k；
CrossFlow 论文用 Muon 8e-4（本仓库为 AdamW，复现论文数值需自行换优化器）。

### `train`（Drift 核心超参）
| 键 | 含义 | 论文对应 |
|---|---|---|
| `push_per_step` | 每步往样本队列推入的真实样本数 | 论文 "push 64 new real samples" |
| `positive_bank_size` | 每类正样本队列长度 | 128 |
| `negative_bank_size` | 全局负样本队列长度 | 1000 |
| `pos_per_sample` / `neg_per_sample` | 每步从队列采样的正/负样本数 | — |
| `gen_per_label` | 每个标签生成的自我负样本数 `N_neg` | 64 |
| `loss_kwargs.R_list` | drift 核温度集合 τ | {0.02, 0.05, 0.2} |
| `cfg_min/cfg_max/neg_cfg_pw/no_cfg_frac` | CFG 采样分布 `α` | 训练时 α∈[1,4] |
| `ema_decay` | EMA 衰减 | 0.999/0.9995/0.9998/0.9999 中选 |
| `save_per_step` / `keep_last` | 存盘频率 / 保留个数 | — |
| `eval_per_step` / `eval_samples` / `cfg_list` | FID 频率 / 采样数 / 扫描的 CFG | — |
| `train_sample_per_step` / `train_sample_grid_size` | 样本网格频率 / 大小 | — |

### `feature`（drift loss 的特征提取器 —— **最关键**）
| 键 | 含义 |
|---|---|
| `use_mae` + `mae_path` | MAE 特征工件路径（`hf://mae_latent_640` 或本地目录） |
| `use_convnext` | 是否叠加 ConvNeXt-V2 特征（pixel/CrossFlow 配置为 `true`，论文里 pixel FID 9.35→3.70） |

论文明确指出：**没有任何特征编码器的模型在 ImageNet 上不 work**。latent 配置配 `mae_latent_*`，
pixel/CrossFlow 配 `mae_pixel_*` + ConvNeXt。

### 管理建议
- 每个实验用一份 yaml，`runs/<名字>` 对应固定目录，便于续训与对比。
- 覆盖个别超参不必改文件：复制配置改几行，或用 `--workdir`/`EXTRA_ARGS` 区分。
- 论文数值复现的三个已知差异（见 `CROSSFLOW-IMAGENET.md`）：本地用 SD-VAE 4 通道 vs 论文
  VA-VAE 32 通道；本地 AdamW vs 论文 Muon；CrossFlow 用 drift loss 训练而非论文的
  `dF/dt` 目标。要严格对齐论文数字需要逐项切换到论文设置。

---

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `newuidmap: executable file not found` | 无 rootless docker 权限，用 `scripts/pack_env.sh` 的 conda-pack 回退 |
| 启动即卡住/超时 | 断网下 HF 会重试；确保设了 `DRIFT_OFFLINE_BUNDLE` 或 `HF_HUB_OFFLINE=1` |
| `Only ImageNet is supported now` | FID 只支持 ImageNet（`utils/fid_util.py`）；非 ImageNet 名字会报错 |
| `We couldn't connect to huggingface.co` | ConvNeXt/MAE 未在离线包里；补 `DRIFT_CONVNEXT_PATH` 与 `feature.mae_path` |
| 续训后 loss 跳变 | workdir 指向了新的戳目录（见第 6 节）；或样本队列重新填充的短暂波动 |
| 显存不足 | 开 `use_remat`/`grad_checkpoint`，或降 `gen_per_label` / batch |
