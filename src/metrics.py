# -*- coding: utf-8 -*-
"""指标唯一实现（metric_spec.md 的代码对应）

⚠️ 本项目所有步骤必须调用本模块，不得各自重写。
若需改动，先改 metric_spec.md 并升高版本号。
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

# ── 全局口径（唯一来源） ──
H = 60                    # 标签期限（交易日）
COST_ONE_WAY = 0.001      # 单边成本 10bp
MIN_N = 50                # 单期最少股票数
NW_LAGS = H               # Newey-West 滞后阶数

SIGNS_FULL = {"gm_vol": -1, "dgm_q": +1, "gm": +1}
SIGNS_MFG = {"transfer_intensity": +1, "dep_rate": +1}


# ═══════════════════════════════════════════
# 1. 相关系数
# ═══════════════════════════════════════════

def spear(x, y):
    """Spearman 秩相关（不依赖 scipy.stats.spearmanr）"""
    rx = pd.Series(x).rank().to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank().to_numpy(dtype=float, copy=True)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


# ═══════════════════════════════════════════
# 2. 合成分数（唯一口径）
# ═══════════════════════════════════════════

def composite_score(g, factors, signs, weight=None):
    """
    ⚠️⚠️ 危险：本函数对传入的 g 做 rank。若 g 是整个面板（多个期），
           rank 会跨期计算 —— 这是错的！

    正确用法：只在「单个期」的切片上调用，或改用 composite_score_panel(df, ...)

    ⚠️ E2/E3 曾误用本函数处理整个面板，导致分数含跨期 rank，
       已登记为 superseded（见 metric_spec.md）
    """
    if weight is None:
        weight = {f: 1.0 for f in factors}
    num = pd.Series(0.0, index=g.index)
    den = pd.Series(0.0, index=g.index)
    for f in factors:
        r = g[f].rank(pct=True)
        ok = r.notna()
        w = abs(weight.get(f, 1.0))
        num[ok] += r[ok] * signs[f] * w
        den[ok] += w
    return (num / den.replace(0, np.nan)).rename("composite")


def composite_score_panel(df, factors, signs, date_col="accper_dt", weight=None):
    """
    ✅ 正确口径：期内 rank(pct) 后按方向加权平均

    这是本项目唯一的合成分数算法。任何步骤都应用本函数，
    不得在整面板上直接调 composite_score。
    """
    parts = []
    for _, g in df.groupby(date_col, sort=False):
        parts.append(composite_score(g, factors, signs, weight))
    return pd.concat(parts).reindex(df.index)


# ═══════════════════════════════════════════
# 3. IC 序列
# ═══════════════════════════════════════════

def ic_series(d, score, label="label", date_col="accper_dt",
              min_n=MIN_N, dropna_all=None):
    """
    每期截面算一次 Spearman，返回 IC 序列。

    参数 dropna_all：若给定因子列表，则要求这些因子全部非空（完全覆盖口径）
                    None = 部分覆盖（仅要求 score 与 label 非空）
    """
    dd = d if dropna_all is None else d.dropna(subset=list(dropna_all))
    ics = []
    for _, g in dd.groupby(date_col):
        s = pd.DataFrame({"sc": score.loc[g.index], "y": g[label]}).dropna()
        if len(s) < min_n:
            continue
        v = spear(s["sc"].values, s["y"].values)
        if np.isfinite(v):
            ics.append(v)
    return np.array(ics)


def ic_series_single(d, col, label="label", date_col="accper_dt", min_n=MIN_N):
    """单因子 IC 序列"""
    ics = []
    for _, g in d.groupby(date_col):
        s = g[[col, label]].dropna()
        if len(s) < min_n:
            continue
        v = spear(s[col].values, s[label].values)
        if np.isfinite(v):
            ics.append(v)
    return np.array(ics)


# ═══════════════════════════════════════════
# 4. HAC 标准误与 t 值
# ═══════════════════════════════════════════

def newey_west_se(x, lags=NW_LAGS):
    """
    Newey-West HAC 标准误（Bartlett 核）
    SE² = [γ0 + 2Σ_{j=1..L}(1 − j/(L+1))γ_j] / n
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    xd = x - x.mean()
    se2 = float(np.dot(xd, xd) / n)
    L = min(int(lags), n - 2)
    for j in range(1, L + 1):
        gj = float(np.dot(xd[j:], xd[:-j]) / n)
        se2 += 2.0 * (1.0 - j / (lags + 1.0)) * gj
    return float(np.sqrt(max(se2, 1e-12) / n))


def hac_stats(ics, lags=NW_LAGS, alpha=0.05):
    """IC 序列的一站式统计：均值 / ICIR / t_naive / t_hac / CI / 分位"""
    ics = np.asarray(ics, dtype=float)
    ics = ics[np.isfinite(ics)]
    if len(ics) < 3:
        return {}
    m = float(ics.mean())
    sd = float(ics.std(ddof=1))
    se_n = sd / np.sqrt(len(ics))
    se_h = newey_west_se(ics, lags)
    z = norm.ppf(1 - alpha / 2)
    return {
        "mean": m, "std": sd,
        "icir": float(m / sd) if sd > 0 else np.nan,
        "t_naive": float(m / se_n) if se_n > 0 else np.nan,
        "t_hac": float(m / se_h) if se_h and se_h > 0 else np.nan,
        "se_naive": float(se_n), "se_hac": float(se_h),
        "inflation": float(se_h / se_n) if se_n > 0 else np.nan,
        "lo": m - z * se_h, "hi": m + z * se_h,
        "n": int(len(ics)),
        "median": float(np.median(ics)),
        "iqr": float(np.percentile(ics, 75) - np.percentile(ics, 25)),
        "worst": float(ics.min()), "best": float(ics.max()),
        "sign_consistency": float((np.sign(ics) == np.sign(m)).mean()),
        "pos_ratio": float((ics > 0).mean()),
    }


# ═══════════════════════════════════════════
# 5. 配对差检验（第二把尺）
# ═══════════════════════════════════════════

def paired_ic_diff(d, score_a, score_b, label="label",
                   date_col="accper_dt", min_n=MIN_N):
    """同一期内算两个分数的 IC，返回差值序列（抵消共同波动）"""
    out = []
    for _, g in d.groupby(date_col):
        s = pd.DataFrame({"a": score_a.loc[g.index],
                          "b": score_b.loc[g.index],
                          "y": g[label]}).dropna()
        if len(s) < min_n:
            continue
        ia = spear(s["a"].values, s["y"].values)
        ib = spear(s["b"].values, s["y"].values)
        if np.isfinite(ia) and np.isfinite(ib):
            out.append(ia - ib)
    return np.array(out)


def paired_test(diffs, lags=NW_LAGS, alpha=0.05):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 3:
        return {}
    se = newey_west_se(diffs, lags)
    m = float(diffs.mean())
    z = norm.ppf(1 - alpha / 2)
    return {"mean_diff": m, "se_hac": se,
            "t_hac": float(m / se) if se and se > 0 else np.nan,
            "lo": m - z * se, "hi": m + z * se,
            "excludes_zero": bool(m - z * se > 0 or m + z * se < 0),
            "n": int(len(diffs))}


# ═══════════════════════════════════════════
# 6. 十分位与 long-only 分离
# ═══════════════════════════════════════════

def decile_analysis(d, score, label="label", date_col="accper_dt",
                    q=10, min_n=MIN_N):
    """
    A 股专用：必须把「多头超额」与「空头贡献」分开报

    多头超额 = Q10 − 全样本均值   （long-only 实际能拿到的）
    空头贡献 = 全样本均值 − Q1     （不能做空，拿不到）
    """
    rows, full = [], []
    for _, g in d.groupby(date_col):
        s = pd.DataFrame({"sc": score.loc[g.index], "y": g[label]}).dropna()
        if len(s) < min_n:
            continue
        full.append(float(s["y"].mean()))
        s = s.assign(b=pd.qcut(s["sc"].rank(method="first"), q,
                               labels=False, duplicates="drop"))
        rows.append(s.groupby("b")["y"].mean())
    if not rows:
        return None
    dec = pd.concat(rows, axis=1).mean(axis=1)
    fm = float(np.mean(full))
    top, bot = float(dec.iloc[-1]), float(dec.iloc[0])
    diffs = np.diff(dec.values)
    diffs = diffs[np.isfinite(diffs)]
    return {"deciles": {int(k): float(v) for k, v in dec.items()},
            "full_mean": fm,
            "spread": top - bot,
            "long_excess": top - fm,
            "short_contrib": fm - bot,
            "short_share": (fm - bot) / (top - bot) if (top - bot) != 0 else np.nan,
            "monotonic": float((np.sign(diffs) == np.sign(diffs[0])).mean())
                         if len(diffs) else np.nan}


# ═══════════════════════════════════════════
# 7. 多重检验门槛
# ═══════════════════════════════════════════

def thresholds(n_trials):
    return {
        "classic_5pct": 1.96,
        "new_factor": 3.0,
        "bonferroni": float(norm.ppf(1 - 0.05 / (2 * max(n_trials, 2)))),
    }


# ═══════════════════════════════════════════
# 8. DSR（注意 SR_0 里的 1/√(T−1)）
# ═══════════════════════════════════════════

def dsr(sharpe, n_trials, T, skew=0.0, kurt=3.0, var_sr=None):
    """
    Deflated Sharpe Ratio（Bailey & López de Prado 2014）

    ⚠️ SR_0 必须含 1/√(T−1) 缩放，漏掉会把门槛抬到 2+，任何策略都过不了
    """
    gamma = 0.5772
    if var_sr is None:
        var_sr = 1.0 / max(T - 1, 1)
    if n_trials < 2:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = float(np.sqrt(var_sr) * ((1 - gamma) * z1 + gamma * z2))
    denom = np.sqrt(max(1 - skew * sharpe + ((kurt - 1) / 4) * sharpe ** 2, 1e-9))
    zz = (sharpe - sr0) * np.sqrt(max(T - 1, 1)) / denom
    return {"sr0": sr0, "z": float(zz), "dsr": float(norm.cdf(zz))}


# ═══════════════════════════════════════════
# 9. PSI
# ═══════════════════════════════════════════

def psi(base, cur, bins=10, eps=1e-4):
    """Population Stability Index（用基准期分位点切桶）"""
    base = pd.Series(base).dropna()
    cur = pd.Series(cur).dropna()
    if len(base) < 100 or len(cur) < 100:
        return np.nan
    qs = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return np.nan
    qs[0], qs[-1] = -np.inf, np.inf
    b = np.clip(np.histogram(base, bins=qs)[0] / len(base), eps, None)
    c = np.clip(np.histogram(cur, bins=qs)[0] / len(cur), eps, None)
    return float(np.sum((c - b) * np.log(c / b)))
