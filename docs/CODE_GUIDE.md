# 代码结构与公式映射

## 1. 设计原则

整理后的主代码只承担六件事：

1. 构造不含未来信息的股票—日期 panel；
2. 构造传统因子与统一预处理；
3. 估计 entropy、MI、CMI 及其显著性；
4. 用 RankIC-GP 与 MI-GP 搜索公式因子，再用 CMI 后置压缩；
5. 用固定滚动 walk-forward 训练 LightGBM；
6. 用 Top-100 等权与状态化交易规则评价结果。


## 2. 数据与标签

### 2.1 正式入口

```text
src/data/baostock.py
src/data/panel.py
scripts/build_lightgbm_final_v11_data.py
```

关键函数：

| 研究对象 | 函数 | 输入 | 输出 |
|---|---|---|---|
| 单股 PIT panel | `src.data.build_stock_frames()` | 未过滤日线、交易日历、上市日 | 信号 panel、执行行情 |
| 开盘可交易状态 | `add_open_execution_flags()` | OHLC、交易状态、ST 状态 | `can_buy_open`、`can_sell_open` |
| 分片合并 | `_stream_parquet()` | 单股 Parquet 分片 | 全市场 Parquet |

标签对应：

$$
y^{1d}_{i,t}=\frac{Open_{i,t+2}}{Open_{i,t+1}}-1.
$$

代码先在完整交易日历上对 `open` 做 `shift(-1)`、`shift(-2)`，再应用信号日过滤。

信号日资格只使用 $t$ 日可知状态；`can_buy_t1` 与 `can_sell_t2` 保存为未来执行结果，不参与信号日样本删除。

## 3. 因子与预处理

### 3.1 传统因子

```text
src/factors/traditional.py
```

`factor_specs()` 定义 205 个因子，覆盖 momentum、reversal、volatility、volume、turnover、liquidity、intraday structure、price-volume interaction、relative strength 与 distribution statistics。

| 研究对象 | 函数 |
|---|---|
| 因子目录与公式 | `build_factor_catalog()` |
| 完整 205 因子 | `generate_factor_panel()` |
| 指定子集 | `generate_selected_factor_panel()` |

所有滚动算子只使用当前及历史行。生成器按股票排序后运算，再恢复 `(date, stock)` 顺序。

### 3.2 横截面预处理

```text
src/preprocessing/cross_sectional.py
```

正式顺序是：

$$
Raw\rightarrow MAD(5)\rightarrow Neutralize\rightarrow RankGaussian.
$$

| 数学步骤 | 函数 |
|---|---|
| MAD 截尾 | `winsorize_mad_matrix()` |
| 百分位截尾（稳健性） | `winsorize_percentile_matrix()` |
| 行业与市值中性化 | `neutralize_matrix()` |
| z-score | `zscore_matrix()` |
| Rank Gaussian | `rank_gaussianize_matrix()` |
| 完整流水线 | `preprocess_matrix()` / `preprocess_cross_section()` |

中性化的回归是：

$$
f_{i,t}=\alpha_t+\beta_t\log(MCap_{i,t})+
\sum_k\gamma_{k,t}Industry_{i,k,t}+\varepsilon_{i,t}.
$$

代码输出残差 $\varepsilon_{i,t}$，随后才做 Rank Gaussian。非线性的 Rank Gaussian 可能重新引入很小的 Pearson 市值相关，但不会改变中性残差的横截面排序。

## 4. 信息论估计

```text
src/information/
├── entropy.py
├── mutual_information.py
├── conditional_mi.py
├── panel_mi.py
├── significance.py
├── multiple_testing.py
├── redundancy.py
└── selection.py
```

| 数学对象 | 核心函数 |
|---|---|
| 离散 entropy | `discrete_entropy()` |
| Histogram MI | `histogram_mutual_information()` |
| KL kNN differential entropy | `estimate_entropy_kl()` |
| KSG-1 MI | `estimate_mi_ksg()` |
| 离散 CMI | `estimate_discrete_cmi()` |
| Frenzel–Pompe CMI | `estimate_cmi_frenzel_pompe()` |
| 日横截面 MI | `daily_cross_sectional_mi()` |
| 滚动 pooled MI | `rolling_pooled_mi()` |
| 日期块 permutation | `panel_permutation_test_mi()` |
| BH-FDR | `benjamini_hochberg()` |
| Forward information selection | `forward_information_selection()` |

估计器保留有限样本负值，不做 `max(estimate, 0)`。理论 MI/CMI 非负，但轻微负估计包含有限样本误差信息；静默裁零会改变 null distribution。

## 5. Symbolic Alpha Mining

```text
src/symbolic/
├── expression.py
├── operators.py
├── gp.py
├── island_gp.py
├── fitness.py
└── large_scale.py
```

### 5.1 表达式引擎

| 对象 | 实现 |
|---|---|
| Terminal / Literal / Call | `expression.py` |
| canonical serialization | `canonical_expression()` |
| structural hash | `structural_hash()` |
| 最大历史窗口 | `required_history()` |
| 未来偏移检查 | `maximum_forward_offset()` |
| 递归执行 | `evaluate_expression()` |

### 5.2 受保护算子

`operators.py` 实现 `add`、`subtract`、`multiply`、`safe_div`、`delay`、`delta`、`ts_mean`、`ts_std`、`ts_min`、`ts_max`、`ts_rank`、`ts_corr`、`ts_cov` 与截面 `rank`。`safe_div`、完整滚动窗口、输出有效率 gate 和复杂度上限共同防止 NaN/inf 爆炸与表达式膨胀。

### 5.3 大规模搜索

正式入口：

```text
scripts/prepare_large_gp_panel.py
scripts/run_large_gp_cluster.py
scripts/run_large_gp_validation_null.py
scripts/run_v2_lowdim_cmi_selection.py
```

RankIC 与 MI 是两个独立岛群，不在同一个 fitness 中线性加权。两路各自保留 100 个候选后取并集，再做结构与语义去重。CMI 不放入 GP 内循环，原因是它的计算成本和条件集依赖过强；它只用于 57 个代表的后置 forward selection。

## 6. LightGBM 与 walk-forward

```text
src/models/cross_sectional_lightgbm.py
src/models/cross_sectional_lightgbm_final.py
src/models/features.py
scripts/prepare_lightgbm_final_v1.py
scripts/run_lightgbm_final_v11.py
```

| 步骤 | 函数 |
|---|---|
| 每日标签 Rank Gaussian | `rank_gaussian_target_by_date()` |
| 每日样本总权重归一为 1 | `date_equal_sample_weights()` |
| 多年度固定滚动 folds | `build_walk_forward_folds_range()` |
| 内部早停、重训、预测 | `fit_lightgbm_final()` |
| 七组冻结特征对照 | `build_group_matrix()` |

每个 fold 的流程是：

1. 最近 720 个交易日作为 inner train；
2. 留 2 日 gap；
3. 60 日 inner validation 选择最佳树数；
4. 测试日前最近 720 日重新训练；
5. 再留 2 日 outer gap；
6. 预测后续 20 个交易日；
7. 向前移动 20 日，不重复使用 OOS 结果调参。

OOS 测试集不参与早停。日期等权的 L2 让股票数量更多的后期交易日不会仅因行数更多而支配目标函数。

## 7. 组合与执行

```text
src/portfolio/execution_constraints.py
```

`simulate_topk_open_execution()` 在完整 OOS 截面中先选择 Top-100，再处理未来执行：

- $t+1$ 开盘不能买：该目标席位留现金；
- $t+2$ 或后续开盘不能卖：继续持有，直到可卖；
- 停牌期间：用最近有效开盘价估值；
- 被困持仓占用资本；
- 不使用杠杆。

这与“先删除未来不可交易股票再回测”在统计上完全不同。后者会系统性移除真实困难样本并造成选择偏差。

## 8. 因子评价指标

```text
src/validation/factor_evaluation.py
scripts/run_cmi_core_deciles_v11.py
```

| 指标 | 函数 |
|---|---|
| Pearson IC | `columnwise_pearson()` |
| Spearman RankIC | `columnwise_spearman()` |
| 十分位分组 | `assign_quantiles()` |
| 单边等权换手 | `equal_weight_turnover()` |
| Newey-West 均值标准误 | `newey_west_mean_se()` |
| CMI-core 非线性形状与年度稳定性 | `run_cmi_core_deciles_v11.py` |

## 9. 最终结果层

`results/final/` 只保留读者验证结论所需的最小证据：七组 scoreboard、日度收益、年度结果、CMI forward path、57 个代表、两个核心因子分层、特征重要性和两张图。

`scripts/build_final_report_assets.py` 只能读取这些冻结文件，不能重新训练或选择。它重新计算配对 Newey–West 统计量和 moving-block bootstrap 区间，并生成 `model_overview.png`。
