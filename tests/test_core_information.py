"""KSG MI 与 Frenzel--Pompe CMI 的合成校准。"""

from __future__ import annotations

import math

import numpy as np

from src.information import estimate_cmi_frenzel_pompe, estimate_mi_ksg


def test_ksg_matches_correlated_gaussian_truth() -> None:
    """相关 Gaussian 的 KSG 估计应接近解析互信息。"""

    rng = np.random.default_rng(11)
    rho = 0.7
    x = rng.normal(size=5_000)
    y = rho * x + math.sqrt(1.0 - rho**2) * rng.normal(size=len(x))
    estimate = estimate_mi_ksg(x, y, k=5)
    truth = -0.5 * math.log(1.0 - rho**2)
    assert abs(estimate - truth) < 0.06


def test_ksg_detects_rankic_blind_quadratic_dependence() -> None:
    """对称二次关系可以线性相关接近零、互信息明显为正。"""

    rng = np.random.default_rng(12)
    x = rng.uniform(-2.0, 2.0, size=4_000)
    y = x**2 + rng.normal(scale=0.25, size=len(x))
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.06
    assert estimate_mi_ksg(x, y, k=5) > 0.8


def test_cmi_removes_common_cause_but_keeps_increment() -> None:
    """给定共同原因后独立残差应近零，条件相关残差仍有增量信息。"""

    rng = np.random.default_rng(13)
    n = 4_000
    z = rng.normal(size=n)
    ex = rng.normal(size=n)
    ey = rng.normal(size=n)
    x = 0.8 * z + ex
    y_null = 0.8 * z + ey
    rho = 0.65
    y_signal = 0.8 * z + rho * ex + math.sqrt(1.0 - rho**2) * ey
    null = estimate_cmi_frenzel_pompe(x, y_null, z, k=5)
    signal = estimate_cmi_frenzel_pompe(x, y_signal, z, k=5)
    assert abs(null) < 0.05
    assert signal > 0.20
