# 复现说明

## 1. 三种复现层级

| 层级 | 内容 | 预计资源 |
|---|---|---|
| Core test | estimator、标签、预处理、表达式与 walk-forward 测试 | 普通电脑，数分钟 |
| Frozen-result rebuild | 从已冻结 CSV 重建统计量和图 | 普通电脑，约数秒 |
| Full research rerun | 全 A 数据、两路岛屿 GP、置换、CMI、32-fold LightGBM | 需要 CPU 集群，数十至数百 core-hours |

原始行情和数十 GB 中间 Parquet 不进入 Git。GitHub 仓库可以完整复核算法与最终小型证据包，但完全重跑需要自行从 BaoStock 获取数据。

## 2. 环境

推荐 Python 3.10--3.12。集群最终训练使用 LightGBM 3.3.5；本地兼容 3.3.5--4.6.0。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows 激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

先运行核心测试：

```bash
python -m pytest -q tests/test_core_*.py
```

## 3. 数据准备

### 3.1 下载与构造 V1.1 panel

```bash
python scripts/build_lightgbm_final_v11_data.py \
  --download-start 2015-01-01 \
  --signal-end 2026-08-21 \
  --execution-start 2024-01-02 \
  --oos-end 2026-08-19
```

正式输出：

```text
data/processed/clean_panel_v11.parquet
data/processed/execution_bars_v11.parquet
data/processed/trading_calendar_v11.csv
```

关键口径：

- `clean_panel_v11` 的行保留只使用信号日状态；
- `label_1d` 允许为空，训练时才过滤，OOS 预测仍保留；
- `can_buy_t1`、`can_sell_t2` 只供执行层使用；
- 行情使用不复权价格，以保持真实开盘成交语义；
- 行业映射必须是 Point-in-Time 月度快照。

## 4. 传统因子与统一预处理

LightGBM 的准备脚本同时生成 205 个传统因子、57 个冻结 GP 因子，并执行：

```text
MAD(5) -> industry/log-float-cap neutralization -> rank Gaussianization
```

```bash
python scripts/prepare_lightgbm_final_v1.py \
  --config configs/lightgbm_final_v11.json \
  --output-dir experiments/lightgbm_final_v11/data
```

这一步对全市场十余年面板需要较大内存。建议 128--160 GB RAM，预计约 1--3 小时，具体取决于磁盘吞吐。

## 5. 重新运行 GP 搜索

### 5.1 准备冻结搜索样本

```bash
python scripts/prepare_large_gp_panel.py \
  --mode prepare \
  --config configs/large_scale_gp_v2.json
```

配置冻结为：

- selection：2022 年前 240 个交易日；
- development validation：2023 年前 240 个交易日；
- 每日 500 只股票的确定性行业/市值分层样本；
- RankIC 与 MI 两个独立通道；
- 每通道 8 islands × 128 individuals × 24 generations；
- KSG $k=5$；
- 20 日 date-block permutation。

### 5.2 集群运行

```bash
sbatch cluster/submit_large_gp_v2.slurm
```

该任务超过 8 小时风险阈值，不建议在笔记本上运行。原实验中搜索、验证与 999 次 null 合计需要数十至数百 core-hours。

### 5.3 候选压缩与 CMI

先从 RankIC/MI 两路各取 100 个候选并集，再在 2022 selection 上用 $|\rho|\ge0.98$ 压缩为 57 个代表：

```bash
python scripts/run_v2_lowdim_cmi_selection.py \
  --mode run \
  --workers 16 \
  --allow-long-run
```

CMI forward selection 使用 99 次共享 20 日 block max-null。该步骤原耗时约 2.2 小时、22,208 次 estimator calls，仍推荐集群运行。

## 6. LightGBM walk-forward

配置文件：

```text
configs/lightgbm_final_v11.json
```

若集群使用 Slurm：

```bash
bash cluster/submit_lightgbm_final_v11_pipeline.sh
```

脚本先提交数据/特征准备作业，再以 `afterok` 依赖提交训练作业。单节点建议 20 CPU、160 GB RAM。原正式训练约 13 小时，峰值内存约 42 GB；准备阶段额外约 1.3 小时。

直接训练入口：

```bash
python scripts/run_lightgbm_final_v11.py \
  --config configs/lightgbm_final_v11.json
```

正式特征对照为：

1. Traditional；
2. Traditional + RankIC-GP；
3. Traditional + MI-GP；
4. Traditional + Union；
5. Traditional + CMI core；
6. Union only；
7. MI only。

不要在看到 OOS 结果后修改这七组、Top-100、720 日窗口或模型超参数，再把同一区间称作独立 OOS。

## 7. 重建最终图表

仓库已经包含冻结的小型结果文件。运行：

```bash
python scripts/build_final_report_assets.py
```

生成：

```text
results/final/derived_statistics.json
results/final/figures/model_overview.png
```

脚本会先确认七组模型各有 637 个不重复 OOS 日期，然后计算预先指定的配对收益差、Newey–West $t$ 和 20 日 moving-block bootstrap 区间。

## 8. 结果文件的最小证据链

| 结论 | 来源 |
|---|---|
| 七组总体表现 | `results/final/model_scoreboard.csv` |
| 年度稳定性 | `results/final/model_yearly.csv` |
| 每日组合与执行状态 | `results/final/model_daily.csv` |
| GP 57 个语义代表 | `results/final/gp_representatives_57.csv` |
| CMI forward path | `results/final/cmi_forward_selection.csv` |
| 两个核心因子重要性 | `results/final/cmi_core_feature_importance.csv` |
| 单因子十分位形状 | `results/final/cmi_core_deciles.csv` |
| 配对不确定性 | `results/final/derived_statistics.json` |

## 9. 随机性

- GP base seed：`20260923`；
- LightGBM seed：`20260925`；
- 报告 bootstrap seed：`20261009`；
- 数值库线程在多进程 GP 中强制为 1，避免线程过度订阅；
- 候选抽样使用日期、股票和 seed 的确定性优先级。

同一软件栈与 CPU 上应重现相同候选和近似相同浮点结果。不同 LightGBM、NumPy、BLAS 或 CPU 架构可能产生极小分裂差异，因此最终比较应允许浮点容差，而不是要求模型二进制完全相同。

## 10. 不应复现的历史分支

以下内容不属于最终主线：

- 将 CMI 放进 GP fitness 的早期实验；
- stability/turnover penalty 版 MI-GP；
- LambdaRank V2；
- 使用未来 `can_buy_t1/can_sell_t2` 预先删除训练样本的旧 panel；
- 失败的集群提交、wheel 兼容性诊断与重复归档包。

它们保留在本地历史仅用于说明研究如何证伪和修正，不应混入最终结果。
