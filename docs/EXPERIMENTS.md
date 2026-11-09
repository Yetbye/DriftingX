# 实验目录与一键运行

本文档列出 **CIFAR-10 上的全部论文实验**（对应 `paper/draft.md` §4）。
所有实验配置在 `configs/exp/`，一键运行 `scripts/run_all_experiments.sh`。

## 一键运行（离线集群）

```bash
# 前置：离线资源包 + CIFAR-10 数据已就位
export DRIFT_OFFLINE_BUNDLE=/scratch/$USER/offline_bundle
export SDVAE_PATH=$DRIFT_OFFLINE_BUNDLE/sdvae

bash scripts/run_all_experiments.sh            # 跑全部 8 个实验
bash scripts/run_all_experiments.sh --only L1  # 只跑某一层
```

每个实验输出到 `runs/<name>/`（checkpoint、metrics.jsonl、样本图、params_ema、
report.png）。训练结束自动生成报告；FID 在训练中和结束时自动评估。

## 实验清单

### Layer 1 — 噪声先验（主对比）
| 实验 | 配置 | 噪声 | 监督 | 论文表 |
|---|---|---|---|---|
| `C10_L1_PP` | pixel DiT | pixel | pixel-MAE | T1/T2 |
| `C10_L1_LP` | X-Drifting (crossflowDiT) | **latent** | pixel-MAE | T1/T2 |

验证：latent 噪声先验 vs pixel 噪声先验（其余全同）。已出数据：
P-P FID 216.9，L-P FID 227.3（L-P 收敛追赶更快，轨迹见 T2）。

### Layer 2 — 表征 grounding（VAE/RAE/PAE 阶梯）
| 实验 | 配置 | grounding encoder | 论文表 |
|---|---|---|---|
| `C10_L2_LP_gsdvae` | X-Drifting + grounding | SD-VAE（通用压缩 latent） | T3 |
| `C10_L2_LP_grae` | X-Drifting + grounding | RAE（DINO autoencoder） | T3 |
| `C10_L2_LP_gpae` | X-Drifting + grounding | PAE（prior-aligned DINO） | T3 |

验证：表征质量越高（SD-VAE < RAE < PAE），生成越好 —— latent 优势来自表征质量。
drift 的吸引-排斥在 frozen encoder 特征空间做，无 decoder、无重建损失。

### Layer 3 — 机制消融（grounding 上的 A+B+f）
| 实验 | 配置 | 机制 | 论文表 |
|---|---|---|---|
| `C10_L3_LP_floss` | + f-loss | 频域监督 | T4 |
| `C10_L3_LP_repcond` | + predict-then-condition | 机制 B | T4 |
| `C10_L3_LP_both` | + f-loss + predict-cond | B + f | T4 |

验证：predict-then-condition 加速收敛；f-loss 提升保真度；两者叠加最优。

## 机制对照（论文 §3）

- **机制 A（representation grounding）**：drift 的特征空间 φ = frozen encoder
  （SD-VAE/RAE/PAE）的 per-patch token，不是 pixel-MAE。代码：`grounding.py`。
- **机制 B（predict-then-condition）**：生成器预测 frozen 表征 ẑ 再 self-condition，
  不去噪第二条轨迹。代码：`grounding.py:RepresentationHead` + `train.py` 的
  `rep_align` 损失项。
- **f-loss（频域监督）**：对最终像素输出做 focal log-frequency 正则。代码：`freq_loss.py`。

## 配置生成与结果收集

```bash
python scripts/make_exp_configs.py      # 生成 configs/exp/*.yaml
python scripts/collect_c10_results.py   # 汇总 8 个实验的 FID
python scripts/analyze_run.py runs/<name>   # 单实验报告
python scripts/compare_backbone.py      # 验证 backbone 与官方一致
```

## 训练规模

CIFAR-10 64px（SD-VAE latent 8×8），crossflowDiT 512-hidden/8-layer，
batch 32，20k 步，AdamW lr 4e-4，EMA 0.999。所有对比实验共用同一 backbone、
同一 drift 特征提取器、同一预算，差异只在噪声先验 / grounding 表征 / 机制。
