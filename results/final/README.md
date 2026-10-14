# 冻结结果证据包

此目录只保存支撑最终报告所需的最小结果，不保存原始行情、完整预测矩阵、逐折模型或集群日志。

## 文件

| 文件 | 含义 |
|---|---|
| `model_scoreboard.csv` | 七组特征的 V1.1 总体 OOS 指标 |
| `model_yearly.csv` | 2024、2025、2026 YTD 分年结果 |
| `model_daily.csv` | 七组模型的日度组合收益、基准与执行状态 |
| `model_run_manifest.json` | 冻结运行口径 |
| `gp_representatives_57.csv` | 2022 selection 上 $|\rho|\ge0.98$ 压缩后的代表 |
| `cmi_forward_selection.csv` | CMI forward selection 的三步路径 |
| `cmi_selection_manifest.json` | CMI 参数与停止原因 |
| `cmi_core_feature_importance.csv` | Traditional + CMI core 的 fold 平均 importance |
| `cmi_core_deciles.csv` | C084/C182 的中性化、Rank-Gaussian 十分位收益 |
| `cmi_core_long_short.csv` | 原始方向 Q10-Q1 统计 |
| `cmi_core_rank_ic.csv` | Raw 与 neutral/normalized RankIC |
| `cmi_core_yearly.csv` | 两因子年度形状稳定性 |
| `derived_statistics.json` | 预先指定模型对的 NW 与 block-bootstrap 比较 |
| `figures/model_overview.png` | 模型总体图 |
| `figures/cmi_core_deciles.png` | 两个核心因子的十分位形状 |

## 口径

- OOS：2024-01-02 至 2026-08-19；
- 637 个交易日，32 个 walk-forward folds；
- Top-100 等权；
- 0 bps；
- 状态化处理未来开盘不可买、不可卖与停牌估值；
- 内部基准为当日合格股票截面 `label_1d` 的等权平均；
- V1.1 是数据契约修复性敏感性实验，不是新的 locked holdout。

## 完整性

`scripts/build_final_report_assets.py` 会验证：

1. scoreboard 恰好包含七组冻结特征集合；
2. 每组恰好 637 个不重复 OOS 日期；
3. 配对比较只在共同日期上进行；
4. 报告图和 `derived_statistics.json` 可从本目录独立重建。
