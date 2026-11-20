# -*- coding: utf-8 -*-
"""E7 统计推断与风险：SR的SE / DSR(真实净值) / 尾部 / 回撤 / 压力八项 / regime / 泄漏五查"""
import sys, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

S = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test\SCRIPT")
sys.path.insert(0, str(S))
import metrics as M

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
PPY = 4
N_TRIALS = 15
GRID_BP = (0, 5, 10, 15, 20, 30, 40, 60, 80)
BASE_COST_BP = 20.9

e4 = json.loads((CLEAN / "meta" / "e4_backtest.json").read_text(encoding="utf-8"))
e5 = json.loads((CLEAN / "meta" / "e5_cost.json").read_text(encoding="utf-8"))
e6 = json.loads((CLEAN / "meta" / "e6_turnover.json").read_text(encoding="utf-8"))

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def sharpe_se(sr, n_years):
    return float(np.sqrt((1 + 0.5 * sr ** 2) / n_years))


def years_needed(sr, t=2.0):
    return float(4 * (1 + 0.5 * sr ** 2) / max(sr ** 2, 1e-9))


def search_budget(n, sr):
    return float(2 * np.log(max(n, 2)) / max(sr ** 2, 1e-9))


def cvar(rets, alpha=0.05):
    r = np.sort(np.asarray(rets, dtype=float))
    k = max(int(np.ceil(alpha * len(r))), 1)
    return float(-r[:k].mean())


def varr(rets, alpha=0.05):
    r = np.sort(np.asarray(rets, dtype=float))
    k = max(int(np.ceil(alpha * len(r))), 1)
    return float(-r[k - 1])


def drawdown_stats(nav):
    nav = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1
    mdd = float(dd.min())
    trough = int(dd.argmin())
    rec = None
    for i in range(trough, len(nav)):
        if nav[i] >= peak[trough]:
            rec = i - trough
            break
    return {"mdd": mdd, "trough": trough, "recovery_periods": rec,
            "underwater": (len(nav) - 1 - trough) if rec is None else 0,
            "cond_dd_below_10": float(dd[dd <= -0.10].mean()) if (dd <= -0.10).any() else None,
            "dd_series": dd.tolist()}


def dsr_real(net, n_trials, ppy=PPY):
    r = np.asarray(net, dtype=float)
    T_years = len(r) / ppy
    sd = r.std(ddof=1)
    sr_pp = r.mean() / sd if sd > 0 else np.nan
    sr_ann = sr_pp * np.sqrt(ppy)
    se = sharpe_se(sr_ann, T_years)
    g = 0.5772
    if n_trials < 2:
        e_max = 0.0
    else:
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
        e_max = se * ((1 - g) * z1 + g * z2)
    z = (sr_ann - e_max) / se if se > 0 else np.nan
    return {"sr_ann": float(sr_ann), "T_years": float(T_years), "se": float(se),
            "e_max": float(e_max), "z": float(z), "dsr": float(norm.cdf(z))}


def main():
    log("读 E4/E5/E6 …")
    out = {}

    for name in e4:
        bt = pd.DataFrame(e4[name]["bt_records"])
        bt["date"] = pd.to_datetime(bt["date"])
        g = bt["gross"].values; b = bt["bench"].values; turn = bt["turn"].values
        t_mean = float(turn.mean())
        cost = BASE_COST_BP / 10000.0
        net = (1 + g) * (1 - t_mean * 2 * cost) - 1
        excess = net - b
        nav = np.cumprod(1 + net)
        ann_bench = (1 + b).prod() ** (PPY / len(b)) - 1
        ann_net = (1 + net).prod() ** (PPY / len(net)) - 1
        ann_excess = ann_net - ann_bench
        sr_ex = excess.mean() / excess.std(ddof=1) * np.sqrt(PPY)
        n_years = len(net) / PPY

        log(f"\n=== {name} ===")
        se = sharpe_se(sr_ex, n_years)
        z95 = 1.96
        log(f"  SR(超额) {sr_ex:+.3f}  T={n_years:.1f}年  SE {se:.3f}  "
            f"95%CI [{sr_ex-z95*se:+.3f}, {sr_ex+z95*se:+.3f}]")
        log(f"  达到 t=2 所需年数 {years_needed(sr_ex):.1f} 年（实际 {n_years:.1f} 年）")
        sb = search_budget(N_TRIALS, sr_ex)
        log(f"  搜索预算：N={N_TRIALS} 时需 {sb:.1f} 年（实际 {n_years:.1f} 年）"
            f" → {'✅ 足够' if n_years >= sb else '❌ 不足'}")

        ds_abs = dsr_real(net, N_TRIALS)          # 绝对收益口径（含 beta）
        ds_ex = dsr_real(excess, N_TRIALS)        # 超额口径（alpha）
        log(f"  DSR 绝对口径：SR {ds_abs['sr_ann']:+.3f}  E[max] {ds_abs['e_max']:.3f}  "
            f"DSR {ds_abs['dsr']:.4f} → {'✅' if ds_abs['dsr']>0.95 else '❌'}")
        log(f"  DSR 超额口径：SR {ds_ex['sr_ann']:+.3f}  E[max] {ds_ex['e_max']:.3f}  "
            f"DSR {ds_ex['dsr']:.4f} → {'✅' if ds_ex['dsr']>0.95 else '❌'}")
        ds = ds_ex

        cv = cvar(excess); va = varr(excess)
        log(f"  尾部：VaR95 {va*100:.2f}%  **CVaR95 {cv*100:.2f}%**  （CVaR/VaR = {cv/va:.2f}×）")

        dd = drawdown_stats(nav)
        log(f"  回撤：MDD {dd['mdd']*100:.1f}%  谷底期 {dd['trough']}  "
            f"恢复 {dd['recovery_periods'] if dd['recovery_periods'] is not None else '未恢复'} 期  "
            f"水下 {dd['underwater']} 期")

        # 压力矩阵
        stress = {}
        stress["S1_cost_x2"] = float(_run_stress(g, b, t_mean, BASE_COST_BP * 2, PPY))
        stress["S2_cost_x3"] = float(_run_stress(g, b, t_mean, BASE_COST_BP * 3, PPY))
        # S3 执行延迟：收益序列后移一期（保守近似）
        g_d = np.r_[np.nan, g[:-1]]; b_d = np.r_[np.nan, b[:-1]]
        m = np.isfinite(g_d)
        stress["S3_delay_1period"] = float(_run_stress(g_d[m], b_d[m], t_mean, BASE_COST_BP, PPY))
        stress["S4_slip_30bp"] = float(_run_stress(g, b, t_mean, BASE_COST_BP + 30, PPY))
        # S5 universe 收缩：只能近似（用 half 换手假设不变，实为放大成本）
        stress["S5_universe_shrink(近似)"] = float(_run_stress(g, b, t_mean * 1.3, BASE_COST_BP, PPY))
        # S6 参数扰动：用 E1 的 timing k1（滞后1期）近似
        stress["S6_param_drift(k1近似)"] = float(_run_stress(g_d[m], b_d[m], t_mean, BASE_COST_BP * 1.2, PPY))
        # S7 数据扰动 1% 缺失：近似为毛收益打 99% 折扣
        stress["S7_data_drop1pct"] = float(_run_stress(g * 0.99, b, t_mean, BASE_COST_BP, PPY))
        # S8 regime：分年最差年
        bt["yr"] = bt["date"].dt.year
        yr_ex = []
        for y, gg in bt.groupby("yr"):
            nn = (1 + gg["gross"].values) * (1 - gg["turn"].mean() * 2 * cost) - 1
            yr_ex.append(float(nn.mean() - gg["bench"].mean()))
        stress["S8_regime_worst_year"] = float(min(yr_ex))

        log(f"  压力八项（净超额年化）：")
        for k, v in stress.items():
            log(f"    {k:32s} {v*100:+7.2f}%  {'✅' if v > 0 else '❌'}")

        n_pos = sum(1 for v in stress.values() if v > 0)
        log(f"  → 八项中 {n_pos}/8 为正")

        out[name] = {"sharpe": {"sr_ann": float(sr_ex), "se": float(se), "n_years": n_years,
                                "lo": float(sr_ex - z95 * se), "hi": float(sr_ex + z95 * se),
                                "years_needed": years_needed(sr_ex)},
                     "search_budget": {"n_trials": N_TRIALS, "needed_years": sb,
                                       "actual_years": n_years, "sufficient": bool(n_years >= sb)},
                     "dsr": ds, "dsr_absolute": ds_abs, "dsr_excess": ds_ex,
                     "tail": {"var_95": va, "cvar_95": cv},
                     "regime": _regime_slice(bt, cost),
                     "drawdown": {k: v for k, v in dd.items() if k != "dd_series"},
                     "stress": stress, "stress_positive": n_pos,
                     "yearly_excess": yr_ex}

    # 泄漏五查：扫脚本找全样本 scaler
    log("\n=== 泄漏五查（第2条：扫脚本）===")
    bad = []
    for f in S.glob("*.py"):
        txt = f.read_text(encoding="utf-8")
        for m in re.finditer(r"(StandardScaler|MinMaxScaler)\(\)\.fit_transform\(", txt):
            bad.append({"file": f.name, "pattern": m.group(0)})
    log(f"  全样本 fit_transform 命中 {len(bad)} 处")
    for x in bad:
        log(f"    ⚠️ {x['file']}: {x['pattern']}")

    out["leakage_scan"] = bad
    (CLEAN / "meta" / "e7_stat.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("\n  → e7_stat.json")


def _regime_slice(bt, cost):
    """用扩张窗口中位数分四态（严格点时刻）"""
    ex = bt["net"].values * 0 + ((1 + bt["gross"].values) * (1 - bt["turn"].values * 2 * cost) - 1) - bt["bench"].values
    s = pd.Series(ex, index=pd.to_datetime(bt["date"]))
    vol = s.rolling(4, min_periods=4).std()
    trend = s.rolling(4, min_periods=4).mean()
    vol_med = vol.expanding(min_periods=6).median()
    trend_med = trend.expanding(min_periods=6).median()
    hi_vol = (vol > vol_med).fillna(False)
    up = (trend > trend_med).fillna(False)
    lab = np.select([up & ~hi_vol, up & hi_vol, ~up & ~hi_vol, ~up & hi_vol],
                    ["宽松上涨", "谨慎上涨", "复苏震荡", "危机下跌"], default="未定")
    out = {}
    for k in ["宽松上涨", "谨慎上涨", "复苏震荡", "危机下跌"]:
        m = lab == k
        if m.sum() >= 3:
            ex_k = s[m].values
            out[k] = {"n": int(m.sum()), "mean_excess": float(ex_k.mean()),
                      "std": float(ex_k.std(ddof=1)) if len(ex_k) > 1 else np.nan,
                      "sharpe_ann": float(ex_k.mean() / ex_k.std(ddof=1) * np.sqrt(PPY))
                                    if len(ex_k) > 1 and ex_k.std(ddof=1) > 0 else np.nan,
                      "cvar_95": cvar(ex_k) if len(ex_k) >= 5 else np.nan}
    return out


def _run_stress(g, b, turn, cost_bp, ppy):
    cost = cost_bp / 10000.0
    net = (1 + g) * (1 - turn * 2 * cost) - 1
    ann_net = (1 + net).prod() ** (ppy / len(net)) - 1
    ann_b = (1 + b).prod() ** (ppy / len(b)) - 1
    return ann_net - ann_b


if __name__ == "__main__":
    main()
