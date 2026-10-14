# 后续研究与策略迭代路线

## 1. 当前项目处于什么阶段

当前结果足以回答“方法是否值得继续”，但不足以直接实盘：

- RankIC 与 MI 双通道具有互补性；
- CMI 后置压缩优于把全部候选直接交给模型的点估计；
- CMI-core 的增量区间仍跨过零；
- 最大回撤约 37%；
- 因子在 2026 年出现衰减；
- V1.1 没有扣交易成本；
- 原 Final Holdout 已经打开，不能继续在同一区间上改规则并声称独立验证。

因此下一阶段的目标不是继续堆复杂模型，而是建立一个可持续的研究—冻结—forward-test—淘汰循环。

## 2. 第一优先级：建立新的未见样本

从当前数据截止日之后开始，把所有新交易日作为真正 forward-only 数据。规则在开始前冻结：

1. 因子表达式固定；
2. 中性化、Rank Gaussian 固定；
3. 720 日窗口、20 日重训频率固定；
4. LightGBM 参数固定；
5. Top-100 和执行规则固定；
6. 成本模型固定；
7. 不因前几个月表现差而临时替换因子。

至少积累 6--12 个月，再把它作为一次新的确认性检验。期间可以监控，但监控阈值本身应预先定义。

## 3. 第二优先级：把 0 bps 研究结果变成净收益结果

### 3.1 最小成本模型

先冻结一个透明版本：

$$
Cost_t=Turnover_t\times c_{roundtrip}.
$$

可从 10 bps 往返作为基准情景开始，但要明确它把佣金、印花税、规费和一部分滑点合并成常数。不要先用回测反推一个“刚好盈利”的成本。

### 3.2 下一步成本模型

随后再把成本拆为：

$$
Cost=Commission+Tax+Fees+Spread+MarketImpact.
$$

市场冲击可按订单金额占 ADV 比例做分段或平方根模型。需要报告容量曲线：100 万、1,000 万、5,000 万、1 亿资金下的净收益和冲击，而不是只报告一个账户规模。

## 4. 第三优先级：把预测排序变成风险受控组合

Top-100 等权是干净的研究映射，但不是最终组合优化。

### 4.1 建议的渐进顺序

1. Top-100 等权，作为不变基线；
2. 行业权重约束，使组合行业权重接近全市场或自定义基准；
3. 单股权重上限与 ADV 容量约束；
4. 组合 beta、Size、Liquidity、Volatility 暴露约束；
5. 加入换手惩罚的凸优化；
6. 最后才考虑风险模型与预期收益联合优化。

示例目标：

$$
\max_w\quad w^\top\hat\alpha
-\lambda_r w^\top\Sigma w
-\lambda_c\|w-w_{prev}\|_1,
$$

约束包括 $\sum_iw_i=1$、$0\le w_i\le w_{max}$、行业偏离、风格暴露与成交容量。

### 4.2 为什么不应立刻优化

如果同时改变因子、模型、持仓数、风险约束和成本参数，无法判断改善来自哪里。每次迭代只改变一个研究层，并保留当前 Top-100 等权作为控制组。

## 5. 因子库迭代

### 5.1 不继续盲目扩大同一语法

V2 已产生约 3 万个唯一结构，但有效 rank 仍很低。继续把代数翻倍，很可能生成更多“振幅—换手—收盘位置”的同主题变体。

更有效的做法：

- 为 terminal 设使用配额或分层采样，降低 `high_low_ratio` dominance；
- 用 semantic novelty archive，而不仅是结构 hash；
- 对表达式输出做 residual novelty：相对于现有 archive 的预测残差进行筛选；
- 先进行维度/单位约束，再允许组合；
- 对入选表达式做常数折叠、幂等算子消除和公共子表达式简化；
- 每一轮保留“新主题数量”，而不只保留 top fitness。

### 5.2 可增加但仍保持无量纲的 terminal

候选扩展：

- open/high/low/close 在当日 range 中的位置；
- 价格相对 20/60/120 日均线的偏离；
- 5/20/60 日相对成交量、成交额和换手；
- Parkinson、Garman–Klass 等区间波动率；
- 上下影线、实体/振幅；
- Amihud illiquidity、零收益比例、量价不对称；
- 相对行业或相对市场的残差信号。

每增加一类 terminal，都应做单独消融，确认它带来了新语义主题，而不是仅提高表达式数量。

## 6. MI 方法迭代

### 6.1 从“依赖”转向“可交易条件均值”

MI 可以因条件方差、尾部或分布形状变化而升高，但这些变化未必产生 $E[Y\mid X]$ 的可交易差异。下一步可把 MI 候选分成：

- mean-information：条件均值发生变化；
- variance-information：主要改变条件波动；
- tail-information：主要改变左尾或右尾概率；
- interaction-information：单因子弱、与传统因子交互后强。

可用条件收益曲线、GAM 辅助图、分位数回归和对 downside event 的分类诊断。它们是解释工具，不替换主模型。

### 6.2 稳健估计

- 在 $k\in\{3,5,10,20\}$ 上报告敏感性；
- 用不同 rolling window 比较 regime mixing；
- 报告 estimator 的 block-bootstrap 区间；
- 使用局部置换或 regime-stratified permutation，避免跨极端 regime 任意交换；
- 将 CMI 条件集维度严格限制在小范围；
- 把 estimator disagreement 当作不确定性，而不是挑选最有利结果。

## 7. 模型迭代

### 7.1 先做 Ridge 基线

当前冻结结果集中在 LightGBM。下一轮应补齐同一目标、同一 fold、同一因子组的 Ridge：

$$
\hat\beta=\arg\min_\beta
\sum_t\frac1{N_t}\sum_i(\tilde y_{i,t}-x_{i,t}^\top\beta)^2
+\lambda\|\beta\|_2^2.
$$

如果 MI 因子的增量只在 LightGBM 中出现，而 Ridge 没有出现，才更支持“信息是非单调/交互型”的解释。否则改善可能只是增加了另一个线性代理。

### 7.2 LightGBM 的稳健性

建议只做小型、预注册网格：

- `num_leaves`: 15 / 31 / 63；
- `min_data_in_leaf`: 250 / 500 / 1,000；
- learning rate 与树数成对调整；
- 固定 feature/bagging fraction；
- 用 nested walk-forward 选择，不接触新 holdout。

不要根据单次 2024--2026 曲线大规模调参。

### 7.3 状态化模型

当前 2026 衰减表明固定参数可能不够。低风险扩展不是 Transformer，而是：

- 720 日与 360 日两个固定窗口做预注册比较；
- 对近期样本使用温和指数衰减权重；
- 按市场宽度、波动率或流动性状态训练两个简单模型；
- 用 validation-only 规则在模型间切换或加权。

任何 regime 标签必须只使用当时信息，切换规则也必须 walk-forward。

## 8. 因子生命周期管理

建立月度监控表，每个因子记录：

- rolling 60/120/240 日 RankIC；
- rolling MI 与 block-null z-score；
- 与现有因子最大相关和最大 MI；
- 分组收益形状漂移；
- 覆盖率、换手、容量；
- LightGBM gain/split share；
- 年度与市场状态表现。

预先定义状态：

```text
research -> shadow -> active -> reduced -> retired
```

淘汰规则应基于连续多个窗口和不确定性，而不是一个月收益为负。退休因子保留历史记录，避免隔几个月以新名字重新挖出同一表达式。

## 9. 策略迭代的实验模板

每次迭代只填写一页：

| 字段 | 内容 |
|---|---|
| Hypothesis | 本次只验证什么 |
| Changed component | 数据、因子、筛选、模型、组合、成本中的哪一层 |
| Frozen controls | 哪些保持不变 |
| Selection period | 设计与筛选用哪段数据 |
| Validation period | 调参用哪段数据 |
| Locked period | 哪段完全不看 |
| Primary metric | 唯一主要指标 |
| Secondary metrics | 解释性指标 |
| Multiplicity family | 本轮一共比较多少方案 |
| Stop rule | 什么结果会拒绝假设 |
| Deployment gate | 达到什么标准才进入 shadow |

推荐的部署 gate：

1. 新 forward 样本中增量收益区间不再明显跨负值；
2. 成本与容量后仍有正超额；
3. 最大回撤和风险暴露符合预设约束；
4. 不依赖单一年份；
5. 不由单个异常月份或少数股票贡献；
6. 执行失败、停牌和涨跌停逻辑在仿真账户中一致。

## 10. 长期扩展方向

在核心研究循环稳定后，可以考虑：

- 基本面与分析师数据，但必须解决公告时点与修订历史；
- 指数成分、行业层级和市场状态的条件信息；
- 因子表达式的多目标 Pareto search（信息、复杂度、容量、稳定性）；
- Bayesian model averaging 或简单 ensemble；
- 更严格的 knockoff / conditional randomization 思路；
- 符号回归的预训练或 LLM 辅助表达式简化。

这些扩展都不应绕过同一原则：先证明新信息不是数据泄漏、已有因子代理或选择偏差，再讨论收益曲线。
