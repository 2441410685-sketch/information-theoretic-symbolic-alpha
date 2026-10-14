# 简历与面试表达

## 1. 推荐项目名称

**Information-Theoretic Symbolic Alpha Mining｜基于增量预测信息的 A 股非线性因子挖掘**

角色可写为：**独立量化研究项目｜Research Lead & Python Research Engineer**

## 2. 简历一行版

面向全 A 股构建信息论驱动的非线性因子研究框架，将传统 RankIC 筛选升级为“RankIC/MI 双通道符号搜索 + CMI 增量压缩 + 严格 walk-forward 验证”，从约 3 万个唯一表达式中提炼 2 个核心因子，并完成带 A 股交易约束的 Top-100 样本外检验。

## 3. 推荐简历版本

### Information-Theoretic Symbolic Alpha Mining

**项目简介：**面向全 A 股的非线性 Alpha 挖掘研究。以 Mutual Information 补充 RankIC 对非单调关系的识别不足，并通过 Conditional Mutual Information 衡量候选因子的增量预测信息，最终在严格样本外协议下检验其组合价值。

- **搭建信息论因子评价体系：**针对 U 型、阈值型等相关系数难以充分刻画的预测关系，自主实现 Histogram MI、KSG Mutual Information 与 Frenzel–Pompe Conditional MI，并用合成数据、日期块置换和有限样本校准检验估计偏差与显著性。
- **构建 RankIC-GP 与 MI-GP 双通道符号搜索：**分别从单调预测能力和非线性依赖出发挖掘候选因子；从约 3 万个唯一公式中筛选 200 个候选，经结构与语义去重、低维 CMI 前向选择，得到 57 个代表因子和 2 个增量核心因子。
- **建立全 A 股可交易研究协议：**统一 Point-in-Time 股票池与 $t+1$ 开盘买入、$t+2$ 开盘卖出的 Open-to-Open 标签；完成 MAD 截尾、行业/市值中性化、Rank Gaussian、固定滚动 walk-forward，并纳入涨跌停、停牌、T+1、卖出受阻和现金占位等执行约束。
- **完成七组因子对照实验：**在 2024-01-02 至 2026-08-19 的 637 个样本外交易日和 32 个滚动窗口上训练 LightGBM；在未计交易成本的研究口径下，Traditional + CMI-core 取得 34.75% 年化收益和 1.096 Sharpe，高于传统因子基线的 31.42% 和 1.010，显示增量筛选后的非线性因子能够补充传统因子信息。

## 4. 更适合一页简历的三条版

- 将横截面 Alpha 筛选从 RankIC 扩展至 KSG MI/CMI，提出“RankIC-GP + MI-GP 并行搜索 → 语义去重 → CMI 后置增量筛选”的两阶段框架，从约 3 万个通道内唯一公式中提炼 2 个核心因子。
- 构建全 A 股 Point-in-Time 研究流水线和 32-fold 固定滚动 OOS 协议，覆盖行业/市值中性化、A 股 T+1、涨跌停、停牌及卖出受阻；避免未来函数、随机切分和未来交易状态筛选。
- LightGBM 七组对照中，Traditional + CMI-core 的 OOS 年化/Sharpe 为 34.75%/1.096，传统基线为 31.42%/1.010，验证了增量筛选后的非线性因子对传统因子集合的补充价值。

## 5. 英文简历版

### Information-Theoretic Symbolic Alpha Mining | Independent Quant Research Project

- Developed an information-theoretic alpha discovery framework that complements RankIC with from-scratch KSG mutual information and Frenzel–Pompe conditional mutual information estimators, validated through synthetic benchmarks and date-block permutation tests.
- Designed dual-channel island genetic programming—RankIC-GP and MI-GP—covering 49,152 population slots and 15,403/14,910 channel-wise unique expressions; compressed 200 finalists into 57 semantic representatives and two incremental core factors via low-dimensional CMI forward selection.
- Built a point-in-time China A-share pipeline with executable $t+1$ open to $t+2$ open labels, daily size/industry neutralization, rank Gaussianization, 32-fold rolling walk-forward validation, and stateful handling of suspensions, price limits and T+1 constraints.
- On 637 OOS sessions from 2024 to 2026, Traditional + CMI-core achieved a 34.75% annualized return and 1.096 Sharpe before transaction costs, compared with 31.42% and 1.010 for the traditional-factor baseline.

## 6. 30 秒面试介绍

这个项目研究的不是“如何把回测做得更漂亮”，而是传统 RankIC 会不会漏掉非单调 Alpha。我从零实现 KSG MI 和 Frenzel–Pompe CMI，设计 RankIC-GP 与 MI-GP 两条并行符号搜索通道，再用语义去重和 CMI 只做后置增量筛选。最终在严格 Point-in-Time、固定滚动 walk-forward 和 A 股状态化执行下，对七组因子做了 637 天 OOS 对照。两个 CMI-core 因子的点估计优于传统基线，但统计区间仍包含零，所以最终结论是“发现了值得继续验证的增量信息”，而不是夸大为已经确认的 Alpha。

## 7. HR 最容易记住的标签

- 信息论 × 统计学习 × 符号回归；
- 不只会调用模型，能够实现并验证统计估计器；
- 同时理解研究创新、数据泄漏、交易可执行性和工程复现；
- 敢于报告失败结果和统计不确定性；
- 能把数学能力转化为完整量化研究 pipeline。

## 8. 使用时的边界

简历中可以写 34.75% 年化和 1.096 Sharpe，但必须保留“0 bps 研究口径”和“OOS 点估计”的限定。不要写“显著击败基线”“可直接实盘”或“MI 全面优于 RankIC”；这些表述不受现有证据支持。
