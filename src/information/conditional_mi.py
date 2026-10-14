"""Day 16：离散 Conditional Mutual Information 与恒等式验证。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from scipy.special import digamma

from .entropy import Correction, TieStrategy, entropy_from_counts
from .mutual_information import mutual_information_from_counts


@dataclass(frozen=True)
class DiscreteCMIResult:
    """保存离散 CMI 的四种等价计算及信息分解诊断。"""

    estimate: float
    conditional_entropy_difference: float
    joint_entropy_identity: float
    chain_rule_increment: float
    stratum_weighted_mi: float
    h_y_given_z: float
    h_y_given_xz: float
    i_x_y: float
    i_z_y: float
    i_xz_y: float
    n_samples: int
    x_states: int
    y_states: int
    z_states: int
    occupied_xyz_states: int
    correction: Correction
    base: float

    @property
    def max_identity_error(self) -> float:
        """返回四种 CMI 恒等式之间的最大绝对数值误差。"""

        values = np.array(
            [
                self.conditional_entropy_difference,
                self.joint_entropy_identity,
                self.chain_rule_increment,
                self.stratum_weighted_mi,
            ],
            dtype=np.float64,
        )
        return float(np.max(np.abs(values - values[0])))

    def to_record(self) -> dict[str, float | int | str]:
        """转换为一行实验记录，便于核对链式法则。"""

        return {
            "cmi": self.estimate,
            "conditional_entropy_difference": self.conditional_entropy_difference,
            "joint_entropy_identity": self.joint_entropy_identity,
            "chain_rule_increment": self.chain_rule_increment,
            "stratum_weighted_mi": self.stratum_weighted_mi,
            "h_y_given_z": self.h_y_given_z,
            "h_y_given_xz": self.h_y_given_xz,
            "i_x_y": self.i_x_y,
            "i_z_y": self.i_z_y,
            "i_xz_y": self.i_xz_y,
            "n_samples": self.n_samples,
            "x_states": self.x_states,
            "y_states": self.y_states,
            "z_states": self.z_states,
            "occupied_xyz_states": self.occupied_xyz_states,
            "max_identity_error": self.max_identity_error,
            "correction": self.correction,
            "base": self.base,
        }


@dataclass(frozen=True)
class KNNCMIResult:
    """保存 Frenzel–Pompe 连续 CMI 估计及近邻几何诊断。"""

    estimate: float
    n_samples: int
    x_dimension: int
    y_dimension: int
    z_dimension: int
    joint_dimension: int
    k: int
    metric: str
    min_joint_radius: float
    median_joint_radius: float
    max_joint_radius: float
    mean_xz_neighbors: float
    mean_yz_neighbors: float
    mean_z_neighbors: float
    max_joint_boundary_ties: int
    jittered: bool

    def to_record(self) -> dict[str, float | int | str | bool]:
        """转换为实验表中的一行，避免只保存最终 CMI 数字。"""

        return {
            "cmi": self.estimate,
            "n_samples": self.n_samples,
            "x_dimension": self.x_dimension,
            "y_dimension": self.y_dimension,
            "z_dimension": self.z_dimension,
            "joint_dimension": self.joint_dimension,
            "k": self.k,
            "metric": self.metric,
            "min_joint_radius": self.min_joint_radius,
            "median_joint_radius": self.median_joint_radius,
            "max_joint_radius": self.max_joint_radius,
            "mean_xz_neighbors": self.mean_xz_neighbors,
            "mean_yz_neighbors": self.mean_yz_neighbors,
            "mean_z_neighbors": self.mean_z_neighbors,
            "max_joint_boundary_ties": self.max_joint_boundary_ties,
            "jittered": self.jittered,
        }


def _discrete_states(
    values: np.ndarray | pd.Series,
    name: str,
) -> tuple[pd.Series, np.ndarray]:
    """把一维标签或多列离散向量转换为可哈希状态与完整性掩码。"""

    array = np.asarray(values, dtype=object)
    if array.ndim == 1:
        series = pd.Series(array, dtype="object")
        return series, series.notna().to_numpy()
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(f"{name} 必须是一维标签或 N×d 离散状态矩阵")

    valid = ~pd.DataFrame(array).isna().any(axis=1).to_numpy()
    states = np.empty(len(array), dtype=object)
    states[:] = np.nan
    # 多个 conditioning variables 被组合成一个联合状态；这正对应 H(Z_1,...,Z_d)。
    for position in np.flatnonzero(valid):
        states[position] = tuple(array[position])
    return pd.Series(states, dtype="object"), valid


def contingency_tensor(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    z: np.ndarray | pd.Series,
) -> np.ndarray:
    """把完整样本编码为 ``x_state × y_state × z_state`` 三维频数表。

    任何变量缺失的行都按 complete-case 删除；缺失不会被伪装成一个经济状态。
    ``z`` 可以是多列离散状态矩阵，但维度增长会迅速制造稀疏单元格。
    """

    x_state, x_valid = _discrete_states(x, "x")
    y_state, y_valid = _discrete_states(y, "y")
    z_state, z_valid = _discrete_states(z, "z")
    lengths = {len(x_state), len(y_state), len(z_state)}
    if len(lengths) != 1:
        raise ValueError("x、y、z 的样本长度必须一致")
    valid = x_valid & y_valid & z_valid
    if not valid.any():
        raise ValueError("x、y、z 没有完整的联合样本")

    x_codes, x_levels = pd.factorize(x_state[valid], sort=False)
    y_codes, y_levels = pd.factorize(y_state[valid], sort=False)
    z_codes, z_levels = pd.factorize(z_state[valid], sort=False)
    shape = (len(x_levels), len(y_levels), len(z_levels))
    flat = (x_codes * shape[1] + y_codes) * shape[2] + z_codes
    return np.bincount(flat, minlength=np.prod(shape)).reshape(shape)


def _validate_tensor(counts_xyz: np.ndarray) -> np.ndarray:
    """检查三维频数表；允许概率权重，但不允许负数或空分布。"""

    tensor = np.asarray(counts_xyz, dtype=np.float64)
    if tensor.ndim != 3 or min(tensor.shape) < 1:
        raise ValueError("counts_xyz 必须是非空三维频数表")
    if np.any(~np.isfinite(tensor)) or np.any(tensor < 0.0):
        raise ValueError("counts_xyz 必须是有限非负数")
    if tensor.sum() <= 0.0:
        raise ValueError("联合频数总和必须大于0")
    return tensor


def conditional_mutual_information_from_counts(
    counts_xyz: np.ndarray,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """由三维频数表计算 ``I(X;Y|Z)``。

    使用联合熵恒等式
    ``H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z)``；原始估计不裁剪为零。
    """

    tensor = _validate_tensor(counts_xyz)
    xz = tensor.sum(axis=1)
    yz = tensor.sum(axis=0)
    z_counts = tensor.sum(axis=(0, 1))
    return float(
        entropy_from_counts(xz.ravel(), correction, base)
        + entropy_from_counts(yz.ravel(), correction, base)
        - entropy_from_counts(z_counts, correction, base)
        - entropy_from_counts(tensor.ravel(), correction, base)
    )


def _details_from_tensor(
    counts_xyz: np.ndarray,
    correction: Correction,
    base: float,
) -> DiscreteCMIResult:
    """在同一完整样本上计算 CMI 的四种等价表达式。"""

    tensor = _validate_tensor(counts_xyz)
    x_counts = tensor.sum(axis=(1, 2))
    y_counts = tensor.sum(axis=(0, 2))
    z_counts = tensor.sum(axis=(0, 1))
    xy = tensor.sum(axis=2)
    xz = tensor.sum(axis=1)
    yz = tensor.sum(axis=0)
    xyz = tensor.ravel()

    h_y = entropy_from_counts(y_counts, correction, base)
    h_z = entropy_from_counts(z_counts, correction, base)
    h_xz = entropy_from_counts(xz.ravel(), correction, base)
    h_yz = entropy_from_counts(yz.ravel(), correction, base)
    h_xyz = entropy_from_counts(xyz, correction, base)
    h_y_given_z = h_yz - h_z
    h_y_given_xz = h_xyz - h_xz
    conditional_difference = h_y_given_z - h_y_given_xz
    joint_identity = h_xz + h_yz - h_z - h_xyz

    i_x_y = mutual_information_from_counts(xy, correction, base)
    i_z_y = mutual_information_from_counts(yz.T, correction, base)
    # 把 (X,Z) 视作一个联合变量；转置确保 reshape 后列仍对应 Y。
    xz_by_y = tensor.transpose(0, 2, 1).reshape(
        tensor.shape[0] * tensor.shape[2], tensor.shape[1]
    )
    i_xz_y = mutual_information_from_counts(xz_by_y, correction, base)
    chain_increment = i_xz_y - i_z_y

    weighted = 0.0
    total = float(tensor.sum())
    for z_index, mass in enumerate(z_counts):
        if mass > 0.0:
            weighted += (mass / total) * mutual_information_from_counts(
                tensor[:, :, z_index], correction, base
            )

    # 频数来自样本时总和为整数；details API 专门面向样本，因此显式保存整数 N。
    n_samples = int(round(total))
    return DiscreteCMIResult(
        estimate=float(conditional_difference),
        conditional_entropy_difference=float(conditional_difference),
        joint_entropy_identity=float(joint_identity),
        chain_rule_increment=float(chain_increment),
        stratum_weighted_mi=float(weighted),
        h_y_given_z=float(h_y_given_z),
        h_y_given_xz=float(h_y_given_xz),
        i_x_y=float(i_x_y),
        i_z_y=float(i_z_y),
        i_xz_y=float(i_xz_y),
        n_samples=n_samples,
        x_states=tensor.shape[0],
        y_states=tensor.shape[1],
        z_states=tensor.shape[2],
        occupied_xyz_states=int(np.count_nonzero(tensor)),
        correction=correction,
        base=float(base),
    )


def conditional_mutual_information_details_from_counts(
    counts_xyz: np.ndarray,
    correction: Correction = "none",
    base: float = math.e,
) -> DiscreteCMIResult:
    """由三维频数表返回 CMI 的完整条件熵、链式法则与分层诊断。"""

    return _details_from_tensor(counts_xyz, correction, base)


def estimate_discrete_cmi_details(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    z: np.ndarray | pd.Series,
    correction: Correction = "none",
    base: float = math.e,
) -> DiscreteCMIResult:
    """估计离散 ``I(X;Y|Z)`` 并返回链式法则与条件熵诊断。"""

    tensor = contingency_tensor(x, y, z)
    return conditional_mutual_information_details_from_counts(
        tensor, correction, base
    )


def estimate_discrete_cmi(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    z: np.ndarray | pd.Series,
    correction: Correction = "none",
    base: float = math.e,
) -> float:
    """返回离散 CMI 的 plug-in 或 Miller–Madow 有限样本估计。"""

    return estimate_discrete_cmi_details(x, y, z, correction, base).estimate


def _continuous_cmi_matrix(
    values: np.ndarray | pd.Series,
    name: str,
) -> np.ndarray:
    """把连续变量整理为 ``N×d``，并拒绝非有限值与常数坐标。"""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or len(matrix) < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} 必须是至少2行的 N×d 数值矩阵")
    if np.any(~np.isfinite(matrix)):
        raise ValueError(f"Frenzel–Pompe 不接受 {name} 中的 NaN/inf")
    if np.any(np.ptp(matrix, axis=0) <= 1e-15):
        raise ValueError(f"{name} 包含常数坐标，不能使用连续 kNN CMI")
    return matrix


def _prepare_frenzel_pompe_inputs(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    z: np.ndarray | pd.Series,
    k: int,
    tie_strategy: TieStrategy,
    jitter_relative: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """验证连续输入，并仅在显式允许时对精确 ties 加可复现微扰。"""

    x_matrix = _continuous_cmi_matrix(x, "x").copy()
    y_matrix = _continuous_cmi_matrix(y, "y").copy()
    z_matrix = _continuous_cmi_matrix(z, "z").copy()
    if len({len(x_matrix), len(y_matrix), len(z_matrix)}) != 1:
        raise ValueError("x、y、z 的样本长度必须一致")
    n_samples = len(x_matrix)
    if not isinstance(k, (int, np.integer)) or not 1 <= int(k) < n_samples:
        raise ValueError("k 必须是满足 1 <= k < N 的整数")
    if tie_strategy not in ("raise", "jitter"):
        raise ValueError("tie_strategy 只能是 'raise' 或 'jitter'")

    has_ties = any(
        len(np.unique(matrix, axis=0)) < n_samples
        for matrix in (x_matrix, y_matrix, z_matrix)
    )
    jittered = False
    if has_ties:
        if tie_strategy == "raise":
            raise ValueError("x、y 或 z 包含精确 ties；请检查变量类型或显式使用 jitter")
        if not np.isfinite(jitter_relative) or jitter_relative <= 0.0:
            raise ValueError("jitter_relative 必须是有限正数")
        joint = np.column_stack([x_matrix, y_matrix, z_matrix])
        scale = np.std(joint, axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        rng = np.random.default_rng(random_state)
        joint += rng.normal(size=joint.shape) * scale * jitter_relative
        x_end = x_matrix.shape[1]
        y_end = x_end + y_matrix.shape[1]
        x_matrix = joint[:, :x_end]
        y_matrix = joint[:, x_end:y_end]
        z_matrix = joint[:, y_end:]
        jittered = True
    return x_matrix, y_matrix, z_matrix, jittered


def gaussian_conditional_mutual_information(
    covariance: np.ndarray,
    x_dimension: int = 1,
    y_dimension: int = 1,
) -> float:
    """返回联合 Gaussian ``I(X;Y|Z)`` 的解析值，单位为 nat。

    covariance 的坐标顺序固定为 ``[X, Y, Z]``。至少保留一个 Z 坐标，
    以免该接口和无条件 Gaussian MI 的语义混淆。
    """

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance 必须是方阵")
    dimension = matrix.shape[0]
    if (
        not isinstance(x_dimension, (int, np.integer))
        or not isinstance(y_dimension, (int, np.integer))
        or x_dimension < 1
        or y_dimension < 1
        or x_dimension + y_dimension >= dimension
    ):
        raise ValueError("X、Y 维数必须为正，且至少留下一个 Z 坐标")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("covariance 必须对称")

    x_index = np.arange(x_dimension)
    y_index = np.arange(x_dimension, x_dimension + y_dimension)
    z_index = np.arange(x_dimension + y_dimension, dimension)
    xz_index = np.r_[x_index, z_index]
    yz_index = np.r_[y_index, z_index]
    matrices = (
        matrix[np.ix_(xz_index, xz_index)],
        matrix[np.ix_(yz_index, yz_index)],
        matrix[np.ix_(z_index, z_index)],
        matrix,
    )
    signs_logs = [np.linalg.slogdet(part) for part in matrices]
    if any(sign <= 0.0 or not np.isfinite(logdet) for sign, logdet in signs_logs):
        raise ValueError("covariance 及所需边际协方差必须正定")
    log_xz, log_yz, log_z, log_xyz = [item[1] for item in signs_logs]
    return float(0.5 * (log_xz + log_yz - log_z - log_xyz))


def estimate_cmi_frenzel_pompe_details(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    z: np.ndarray | pd.Series,
    k: int = 5,
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260902,
) -> KNNCMIResult:
    """使用 Frenzel–Pompe k近邻方法估计连续 ``I(X;Y|Z)``。

    在联合 ``(X,Y,Z)`` 空间采用 Chebyshev 距离取得第 k 个邻居半径，
    再用同一严格半径分别统计 ``(X,Z)``、``(Y,Z)`` 与 ``Z`` 投影空间的
    邻居数。返回未经裁零的原始有限样本估计。
    """

    x_matrix, y_matrix, z_matrix, jittered = _prepare_frenzel_pompe_inputs(
        x,
        y,
        z,
        k,
        tie_strategy,
        jitter_relative,
        random_state,
    )
    n_samples = len(x_matrix)
    k = int(k)
    xz = np.column_stack([x_matrix, z_matrix])
    yz = np.column_stack([y_matrix, z_matrix])
    joint = np.column_stack([x_matrix, y_matrix, z_matrix])
    joint_tree = cKDTree(joint)

    # 第一列是样本自身，第 k 个其他邻居位于第 k+1 列。
    distances, _ = joint_tree.query(
        joint,
        k=k + 1,
        p=np.inf,
        workers=1,
    )
    radius = np.asarray(distances[:, k], dtype=np.float64)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("联合近邻半径非正；样本不满足连续 CMI 假设")

    # Frenzel–Pompe 与 KSG 一样要求投影空间严格使用距离 < epsilon_i。
    strict_radius = np.nextafter(radius, 0.0)

    def projected_counts(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            cKDTree(values).query_ball_point(
                values,
                r=strict_radius,
                p=np.inf,
                workers=1,
                return_length=True,
            ),
            dtype=np.int64,
        ) - 1

    xz_neighbors = projected_counts(xz)
    yz_neighbors = projected_counts(yz)
    z_neighbors = projected_counts(z_matrix)
    if np.any(xz_neighbors < 0) or np.any(yz_neighbors < 0) or np.any(z_neighbors < 0):
        raise RuntimeError("投影空间近邻计数出现负数")

    estimate = float(
        digamma(k)
        + np.mean(
            digamma(z_neighbors + 1)
            - digamma(xz_neighbors + 1)
            - digamma(yz_neighbors + 1)
        )
    )

    inclusive_joint = np.asarray(
        joint_tree.query_ball_point(
            joint,
            r=radius,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1
    strict_joint = np.asarray(
        joint_tree.query_ball_point(
            joint,
            r=strict_radius,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1

    return KNNCMIResult(
        estimate=estimate,
        n_samples=n_samples,
        x_dimension=x_matrix.shape[1],
        y_dimension=y_matrix.shape[1],
        z_dimension=z_matrix.shape[1],
        joint_dimension=joint.shape[1],
        k=k,
        metric="chebyshev",
        min_joint_radius=float(np.min(radius)),
        median_joint_radius=float(np.median(radius)),
        max_joint_radius=float(np.max(radius)),
        mean_xz_neighbors=float(np.mean(xz_neighbors)),
        mean_yz_neighbors=float(np.mean(yz_neighbors)),
        mean_z_neighbors=float(np.mean(z_neighbors)),
        max_joint_boundary_ties=int(np.max(inclusive_joint - strict_joint)),
        jittered=jittered,
    )


def estimate_cmi_frenzel_pompe(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    z: np.ndarray | pd.Series,
    k: int = 5,
    tie_strategy: TieStrategy = "raise",
    jitter_relative: float = 1e-10,
    random_state: int = 20260902,
) -> float:
    """返回 Frenzel–Pompe CMI 的原始有限样本估计，单位为 nat。"""

    return estimate_cmi_frenzel_pompe_details(
        x,
        y,
        z,
        k=k,
        tie_strategy=tie_strategy,
        jitter_relative=jitter_relative,
        random_state=random_state,
    ).estimate
