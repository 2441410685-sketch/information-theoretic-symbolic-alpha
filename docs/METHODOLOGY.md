# 方法、评价指标与有限样本估计

## 1. 预测问题与可交易时间轴

对股票 $i$ 和交易日 $t$，因子 $f_{i,t}$ 使用 $t$ 日收盘时可获得的信息。信号形成后，最早在 $t+1$ 开盘执行，因此主标签定义为：

$$
y_{i,t}^{1d}=\frac{Open_{i,t+2}}{Open_{i,t+1}}-1.
$$

这个标签对应 $t+1$ 开盘买入、$t+2$ 开盘卖出，满足 A 股 T+1。若直接使用 $Close_{t+1}/Close_t-1$，同时又让因子读取完整 $t$ 日收盘和成交量，就隐含了在因子计算完成之前以 $Close_t$ 成交。

股票池 $\mathcal U_t$ 只由 $t$ 日状态决定：非 ST、当日正常交易、上市至少 60 个交易日、价格成交字段有效。未来开盘能否买卖属于执行结果，不属于 $t$ 日样本资格。

## 2. 因子预处理

### 2.1 MAD 截尾

金融因子常有极端值、跳跃、分母接近零和成交量尖峰。均值与标准差会被少量异常点显著拉动，因此主流程使用中位数绝对偏差：

$$
MAD_t=\operatorname{median}_i\left|f_{i,t}-\operatorname{median}_j(f_{j,t})\right|.
$$

令 $m_t$ 为当日中位数，截尾区间为：

$$
[m_t-c\cdot 1.4826 MAD_t,\;m_t+c\cdot 1.4826 MAD_t],
$$

其中 $c=5$。常数 1.4826 使 Gaussian 样本下的 MAD 与标准差尺度一致。百分位截尾可作为稳健性方案，但固定百分位会在截面较小或横截面分布变化时截去不同比例的信息。

### 2.2 市值与行业中性化

每日横截面拟合：

$$
f_{i,t}^{win}=\alpha_t+\beta_t\log(MCap_{i,t})+
\sum_{k=1}^{K_t-1}\gamma_{k,t}\mathbb 1(Industry_{i,t}=k)+\varepsilon_{i,t}.
$$

取残差 $f_{i,t}^{neutral}=\varepsilon_{i,t}$。省略一个行业虚拟变量或等价地使用适当的最小二乘处理，避免完全共线。逐日回归而不是全样本回归，保证行业与市值暴露只使用当时截面。

### 2.3 Rank Gaussianization

对当日 $N_t$ 个有效样本：

$$
u_{i,t}=\frac{Rank(f^{neutral}_{i,t})-0.5}{N_t},
\qquad
z_{i,t}=\Phi^{-1}(u_{i,t}).
$$

理论 MI 对可逆变换不变。若 $g$ 单调可逆，Jacobian 项在联合 entropy 与边际 entropy 的组合中抵消，所以 $I(g(X);Y)=I(X;Y)$。但有限样本 KSG 依赖局部距离几何，尺度差异、重尾和局部稀疏仍会改变邻域，从而改变 $\widehat I$。Rank Gaussian 把每个交易日映射到稳定的边际尺度，降低极端值对距离的支配，并避免不同日期的因子量纲漂移。

## 3. 传统相关指标

### 3.1 Pearson IC

$$
IC_t=Corr(f_{\cdot,t},y_{\cdot,t}).
$$

它度量线性关系，对异常值敏感。

### 3.2 Spearman RankIC

$$
RankIC_t=Corr(Rank(f_{\cdot,t}),Rank(y_{\cdot,t})).
$$

它等价于秩上的 Pearson correlation，捕捉单调但未必线性的关系。U-shape、inverse-U 和对称 threshold 可能有显著预测结构而 RankIC 接近零。

### 3.3 时间汇总

$$
\overline{RankIC}=\frac{1}{T}\sum_t RankIC_t,
\qquad
ICIR=\frac{\overline{RankIC}}{s(RankIC_t)}.
$$

本项目同时报告正 IC 比例与 Newey–West 标准误。日度 IC 有序列相关时，$s/\sqrt T$ 的 IID 标准误会过小。

### 3.4 分组收益

每日按因子排序为 Q1--Q10，计算：

$$
\bar y_{q,t}=\frac{1}{|Q_{q,t}|}\sum_{i\in Q_{q,t}}y_{i,t}.
$$

Q10-Q1 适合方向性单调信号，却不是 MI 因子的通用终局指标。对 U-shape 因子，两个极端组可能同时优于中间组；对 threshold 因子，有效差异可能只出现在一个局部区间。

## 4. Entropy 与 Mutual Information

### 4.1 离散 entropy

$$
H(X)=-\sum_x p(x)\log p(x).
$$

它是编码一个随机变量所需的平均信息量。条件 entropy 为：

$$
H(Y\mid X)=-\sum_{x,y}p(x,y)\log p(y\mid x).
$$

### 4.2 Mutual Information

由 entropy 恒等式：

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
=H(Y)-H(Y\mid X).
$$

也可写为联合分布与独立分布的 KL divergence：

$$
I(X;Y)=D_{KL}\bigl(p(x,y)\Vert p(x)p(y)\bigr)
=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}.
$$

因此 $I(X;Y)\ge0$，并且当且仅当 $X,Y$ 独立时为零（在常规可测条件下）。MI 是对称、无方向的依赖度量。它回答“知道因子后，未来收益的不确定性减少多少”，不直接回答收益方向和交易收益。

## 5. Histogram MI 与有限样本偏差

把连续变量分别离散到 bins，使用频率：

$$
\hat p_{ab}=\frac{n_{ab}}{N},
\qquad
\widehat I_{hist}=\sum_{a,b:n_{ab}>0}\hat p_{ab}
\log\frac{\hat p_{ab}}{\hat p_{a\cdot}\hat p_{\cdot b}}.
$$

bin 太少会把局部依赖平均掉，产生高偏差；bin 太多会出现大量稀疏格子，独立样本也容易得到正 MI。样本熵的 plug-in estimator 通常向下偏，Miller–Madow 修正为：

$$
\widehat H_{MM}=\widehat H_{ML}+\frac{K_+-1}{2N},
$$

其中 $K_+$ 是非空状态数。MI 的修正可由 $H(X)+H(Y)-H(X,Y)$ 三项组合得到，但它只是一阶修正，不能解决 bin 选择、稀疏性或非平稳问题。

Histogram MI 的价值是透明和可视化，而不是作为本项目的最终连续变量估计器。

## 6. Differential Entropy 与 kNN Entropy

连续变量的 differential entropy：

$$
h(X)=-\int p(x)\log p(x)\,dx.
$$

它依赖坐标尺度，可以为负。例如区间长度小于 1 的均匀分布有 $h(X)=\log L<0$。MI 中尺度项抵消，所以 MI 仍非负且具有可逆变换不变性。

### 6.1 Kozachenko–Leonenko estimator

在 $d$ 维空间中，令 $\rho_{k,i}$ 为样本 $i$ 到第 $k$ 个邻居的距离，$c_d$ 为所选度量下单位球体积。局部密度近似：

$$
\hat p(x_i)\approx\frac{k}{N c_d\rho_{k,i}^{d}}.
$$

代入 $h(X)=-E\log p(X)$ 并使用 order statistic 的 digamma 修正，得到：

$$
\widehat h_{KL}(X)
=\psi(N)-\psi(k)+\log c_d
+\frac{d}{N}\sum_{i=1}^{N}\log\rho_{k,i}.
$$

$k$ 小时局部性强、方差高；$k$ 大时方差降低，但邻域跨越密度弯曲区，平滑偏差增大。维度升高时邻域半径迅速膨胀，这是后续 CMI 不能一次条件化数百因子的根本原因。

## 7. KSG Mutual Information estimator

直接用三个 KL entropy 相减，会让每一项的局部尺度误差不同，偏差不能充分抵消。KSG-1 在联合空间与边际空间共享同一局部尺度。

对每个样本在联合 $(X,Y)$ 空间使用 Chebyshev 距离找到第 $k$ 个邻居半径 $\epsilon_i$。再统计边际空间中严格满足距离 $<\epsilon_i$ 的邻居数 $n_{x,i},n_{y,i}$：

$$
\widehat I_{KSG1}(X;Y)
=\psi(k)+\psi(N)
-\frac1N\sum_{i=1}^{N}
\left[\psi(n_{x,i}+1)+\psi(n_{y,i}+1)\right].
$$

共享半径使 joint 与 marginal density 的主要局部体积误差相互抵消。代码使用 `np.nextafter(epsilon, 0)` 实现严格小于，避免把联合球边界点错误计入边际计数。

有限样本注意事项：

- 理论 MI 非负，但 $\widehat I_{KSG}$ 可轻微为负；本项目保留负值。
- ties 破坏连续分布假设；默认报错，只有显式指定时才加入固定 seed 的极小 jitter。
- $k=5$ 是局部性与稳定性的折中，不是普适最优值。
- pooled 样本量很大也不等于独立样本量很大；股票共振与日期依赖会降低有效样本量。

项目用 Independent、Linear、Quadratic、Sinusoidal、Threshold 和 Inverse-U 合成关系验证：非单调关系可能有 $Corr\approx0$、$RankIC\approx0$，但 $\widehat I>0$。

## 8. Panel data 中的 MI

金融样本索引为 $(i,t)$，不能把所有行无解释地 flatten 为 IID。

### 8.1 每日横截面 MI

$$
\widehat I_t=\widehat I(f_{\cdot,t};y_{\cdot,t}).
$$

优点是避免跨 regime 混合，并得到时间稳定性序列。缺点是单日样本通常只有几千只股票，KSG 方差较高，而且股票受共同市场与行业冲击，并非独立。

### 8.2 Rolling pooled MI

$$
\widehat I_{t-L:t}=\widehat I\left({f_{i,s}};\{y_{i,s}}\right),
\quad s\in[t-L,t].
$$

它增加名义样本量，但混合不同日期的边际分布、波动状态和股票组成。本项目先逐日 Rank Gaussian，再在固定日期窗口 pooled，并把结果解释为“该窗口混合分布下的有限样本依赖估计”，而不是不变的结构参数。

## 9. Null、Permutation 与 Block Permutation

### 9.1 Null hypothesis

Null 不是“MI 理论值必须等于零”的口号，而是一套具体的数据生成规则。例如：在保留因子矩阵和每个日期横截面结构的前提下，未来收益标签与因子没有预测对应关系。

Null distribution 是在该规则成立时，统计量 $T$ 会出现的有限样本分布：

$$
\{T^{\pi_1},T^{\pi_2},\ldots,T^{\pi_B}\}.
$$

Monte Carlo p-value 使用：

$$
p=\frac{1+\sum_{b=1}^{B}\mathbb 1(T^{\pi_b}\ge T^{obs})}{B+1}.
$$

加 1 避免有限次 permutation 产生不可能的 $p=0$。

### 9.2 为什么不用完全随机打乱所有行

金融数据存在时间自相关、波动聚集、市场 regime 和同日横截面相关。完全行级 permutation 会制造一个比真实无效因子更理想化的 null，导致 null 方差过小、p-value 过度乐观。

本项目按交易日共同移动全部股票标签，并以 20 个交易日为块置换日期块。这样保留：

- 每日横截面的共同冲击；
- 块内短期时间依赖；
- 因子自身的横截面结构；

同时破坏因子与正确未来收益日期之间的对应。

它仍不是精确 conditional randomization test，因为非平稳 regime 不能被任意交换。结果应称为 development screen，而不是绝对显著性证明。

## 10. Multiple testing

GP 会评估数万表达式。若每个无效候选以 5% 概率误报，选择历史最优值必然产生严重 selection bias。

Benjamini–Hochberg 将 p-value 排序 $p_{(1)}\le\cdots\le p_{(m)}$，寻找最大 $j$ 满足：

$$
p_{(j)}\le\frac{j}{m}q,
$$

并拒绝前 $j$ 个假设，从而在相应条件下控制 False Discovery Rate。项目还使用共享 permutation 的 maxT：每次置换取所有候选统计量最大值，用其 95% 分位数控制家族层面的极端选择。

这些校正覆盖同一轮冻结候选，但不能自动校正研究者在 V1、V2、多次阈值调整中的适应性。真正的解决方案仍是一次性新 holdout 或持续 forward test。

## 11. Conditional Mutual Information

### 11.1 定义

$$
I(X;Y\mid Z)=H(Y\mid Z)-H(Y\mid X,Z).
$$

等价形式：

$$
I(X;Y\mid Z)=
E_Z\left[D_{KL}\bigl(p(x,y\mid z)\Vert p(x\mid z)p(y\mid z)\bigr)\right].
$$

金融解释：已经知道因子集合 $Z$ 后，新因子 $X$ 还能减少多少关于未来收益 $Y$ 的不确定性。

### 11.2 Frenzel–Pompe estimator

在联合 $(X,Y,Z)$ 空间找到第 $k$ 个邻居半径 $\epsilon_i$，再用相同严格半径统计 $(X,Z)$、$(Y,Z)$ 与 $Z$ 空间的邻居数：$n_{xz,i},n_{yz,i},n_{z,i}$。本项目实现：

$$
\widehat I(X;Y\mid Z)
=\psi(k)+\frac1N\sum_{i=1}^{N}
\left[
\psi(n_{z,i}+1)-\psi(n_{xz,i}+1)-\psi(n_{yz,i}+1)
\right].
$$

当 $Z$ 维数增加时，联合空间维数为 $d_X+d_Y+d_Z$，局部邻域迅速变稀。一次把几百个传统因子塞进 $Z$ 会让估计几乎失去可解释性。

### 11.3 Forward information selection

从空集合开始：

$$
f^*=\arg\max_{f\notin S}\widehat I(f;Y\mid S),
\qquad S\leftarrow S\cup\{f^*\}.
$$

每一步只增加一个条件维度，并用共享 20 日 block max-null 决定停止。本项目从 57 个语义代表中选择：

1. `C084`：边际 MI 为 0.06537；
2. `C182`：在 `C084` 条件下 CMI 为 0.00830，超过该步 null 95% 阈值 0.00552；
3. 下一候选 `C102` 的 CMI 为 0.00364，低于 null 阈值 0.00464，停止。

CMI 适合做后置低维压缩，不适合放入每个 GP 个体的 fitness 内循环。

## 12. Genetic Programming

因子是表达式树。terminal 是无量纲 PIT 输入，primitive 是受保护的算术、滚动和截面算子。每代包括：

1. tournament selection；
2. elite 保留；
3. subtree crossover；
4. subtree mutation；
5. reproduction；
6. 岛间 ring migration；
7. 结构 hash 与语义 SimHash 去重。

正式 V2 使用 RankIC 和 MI 两个独立通道，各 8 个 islands、每岛 128 个体、24 代，共 $24{,}576$ 个 population slots/通道。搜索期不使用 stability 或 turnover soft penalty，以免事先压掉可能有信息但高换手的候选；这些指标在候选产生后诊断。

硬约束仍必须存在：最多 39 nodes、深度 9、历史 60 日、有效率至少 80%。它们不是经济偏好，而是避免数值爆炸、无限表达式和不可执行历史长度。

## 13. LightGBM 训练目标

原始收益在不同交易日波动尺度差异很大。监督目标先做每日横截面 Rank Gaussian：

$$
\tilde y_{i,t}=\Phi^{-1}\left(
\frac{Rank(y_{i,t})-0.5}{N_t}
\right).
$$

LightGBM 最小化日期等权的 L2：

$$
\mathcal L=
\sum_t\frac1{N_t}\sum_{i=1}^{N_t}
(\tilde y_{i,t}-\hat y_{i,t})^2.
$$

这让每个交易日总权重相同。早停仍使用内部验证集上的同一目标，不直接使用 RankIC 或 MI，原因是：

- RankIC 只奖励单调排序，对 MI 因子可能不公平；
- 验证 MI 作为每轮 early-stopping metric 计算量过高且方差大；
- 相同的连续目标便于比较所有特征组；
- LightGBM 的递归二元切分可用多个叶区间表示 U-shape、threshold 和非对称关系。

模型使用 `regression_l2`、learning rate 0.03、31 leaves、`min_data_in_leaf=500`、feature/bagging fraction 0.8、$L_1=0.1$、$L_2=5$、最多 1,000 轮和 100 轮早停耐心。

## 14. Walk-forward validation

固定滚动窗口比扩展窗口更适合本项目的非平稳 A 股数据。每个 fold 使用最近 720 个交易日，旧 regime 自然退出。标签依赖到 $t+2$，所以内部验证和 OOS 测试前都留 2 个交易日 gap。

最终 OOS 每次预测 20 个交易日，向前滚动 20 日。所有因子公式、候选、预处理、模型参数和 Top-100 规则在打开原 Final Holdout 前冻结。V1.1 后来修复了未来交易状态筛选，因此它只能称为修复性敏感性重算，不能重新称为未见 holdout。

## 15. 组合指标与执行

Top-100 等权目标权重为 1%。状态化执行遵循：买不到的目标席位留现金，卖不出的旧持仓继续占用资本，停牌用最近有效开盘价估值，不允许杠杆。

主要组合指标：

$$
AnnualReturn=(NAV_T/NAV_0)^{252/T}-1,
$$

$$
Sharpe=\frac{\bar r}{s(r)}\sqrt{252},
\qquad
MDD=\min_t\left(\frac{NAV_t}{\max_{s\le t}NAV_s}-1\right),
$$

$$
Calmar=\frac{AnnualReturn}{|MDD|}.
$$

内部基准是同日合格股票的 `label_1d` 等权平均。它适合比较选股信息，但不是可投资指数，也没有模拟自身换手。当前 V1.1 主结果为 0 bps；成本后结论必须在冻结持仓上另做预注册敏感性，而不能因成本不理想重调因子。

## 16. 参考文献

1. Kozachenko, L. F. and Leonenko, N. N. (1987), [Sample Estimate of the Entropy of a Random Vector](https://www.mathnet.ru/eng/ppi797).
2. Kraskov, A., Stögbauer, H. and Grassberger, P. (2004), [Estimating Mutual Information](https://doi.org/10.1103/PhysRevE.69.066138).
3. Frenzel, S. and Pompe, B. (2007), [Partial Mutual Information for Coupling Analysis of Multivariate Time Series](https://doi.org/10.1103/PhysRevLett.99.204101).
4. Benjamini, Y. and Hochberg, Y. (1995), [Controlling the False Discovery Rate](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
5. Ke, G. et al. (2017), [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html).
