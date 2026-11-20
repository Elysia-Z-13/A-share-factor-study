# -*- coding: utf-8 -*-
"""M-V 模型检验：五道门 + 置换检验 + 过拟合度 + DSR"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
SEED = 42

GROUP_FULL = ["gm_vol", "dgm_q", "gm"]
SIGN_FULL = {"gm_vol": -1, "dgm_q": +1, "gm": +1}
GROUP_MFG = ["transfer_intensity", "dep_rate"]
SIGN_MFG = {"transfer_intensity": +1, "dep_rate": +1}

COST_ONE_WAY = 0.001      # 10bp
N_TRIALS = 14             # 来自 trial 账本

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def spear(x, y):
    rx = pd.Series(x).rank().to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank().to_numpy(dtype=float, copy=True)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float((rx*ry).sum()/den) if den > 0 else np.nan


def legal_deadline(a):
    y, m = a.year, a.month
    if m == 12: return pd.Timestamp(y + 1, 4, 30)
    if m == 3:  return pd.Timestamp(y, 4, 30)
    if m == 6:  return pd.Timestamp(y, 8, 31)
    if m == 9:  return pd.Timestamp(y, 10, 31)
    return a + pd.Timedelta(days=120)


def load():
    fp = pd.read_csv(CLEAN / "fin" / "factor_panel.csv",
                     encoding="utf-8-sig", dtype={"code": str})
    fp["accper_dt"] = pd.to_datetime(fp["accper"], errors="coerce")
    fp = fp.dropna(subset=["accper_dt"]).sort_values(["code", "accper_dt"])
    denom = fp["fa_cost_begin"].replace(0, np.nan)
    fp["transfer_intensity"] = fp["transfer_from_cip"] / denom
    fp["dep_rate"] = fp["dep_add"] / denom
    fp["gm_vol"] = fp.groupby("code")["gm"].transform(
        lambda s: s.rolling(4, min_periods=3).std())
    fp["dgm_q"] = fp.groupby("code")["gm"].diff()
    fp["deadline"] = fp["accper_dt"].apply(legal_deadline)
    ann = pd.to_datetime(fp.get("declare"), errors="coerce")
    fp["available_at"] = ann.where(ann.notna(), fp["deadline"])

    parts = []
    for y in range(2010, 2025):
        f = CLEAN / "labels" / f"label_{y}.csv"
        if f.exists():
            parts.append(pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                                     usecols=["code", "date", "label"]))
    lab = pd.concat(parts, ignore_index=True)
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab.dropna(subset=["label"]).drop_duplicates(["code", "date"])
    lab_by = {c: g.sort_values("date")[["date", "label"]].reset_index(drop=True)
              for c, g in lab.groupby("code", sort=False)}
    out = []
    for code, g in fp.groupby("code", sort=False):
        g = g.sort_values("available_at").reset_index(drop=True)
        ll = lab_by.get(code)
        if ll is None or ll.empty:
            out.append(g.assign(label=np.nan)); continue
        m = pd.merge_asof(g, ll, left_on="available_at", right_on="date",
                          direction="forward", tolerance=pd.Timedelta(days=15))
        out.append(m.drop(columns=["date"]))
    return pd.concat(out, ignore_index=True).sort_values(
        ["accper_dt", "code"]).reset_index(drop=True)


def eq_score(d, factors, signs):
    return sum(d[f].rank(pct=True) * signs[f] for f in factors) / len(factors)


def ic_series(d, score, date_col="accper_dt", min_n=50):
    ics = []
    for _, g in d.groupby(date_col):
        s = pd.DataFrame({"sc": score.loc[g.index], "y": g["label"]}).dropna()
        if len(s) < min_n:
            continue
        v = spear(s["sc"].values, s["y"].values)
        if np.isfinite(v):
            ics.append(v)
    return np.array(ics)


def perm_test(d, factors, signs, n_perm=200, seed=SEED, date_col="accper_dt"):
    """同日内打乱标签的置换检验"""
    rng = np.random.default_rng(seed)
    score = eq_score(d, factors, signs)
    real = float(np.mean(ic_series(d, score)))

    null = []
    for _ in range(n_perm):
        dd = d.copy()
        dd["_y"] = dd.groupby(date_col)["label"].transform(
            lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        s2 = pd.DataFrame({"sc": score.loc[dd.index], "y": dd["_y"]}).dropna()
        ics = []
        for _, g in s2.groupby(dd[date_col]):
            if len(g) < 50:
                continue
            v = spear(g["sc"].values, g["y"].values)
            if np.isfinite(v):
                ics.append(v)
        if ics:
            null.append(float(np.mean(ics)))
    null = np.array(null)
    return {"real_ic": real, "null_mean": float(null.mean()),
            "null_std": float(null.std()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float((null >= real).mean()), "n_perm": len(null)}


def overfit_check(df, factors, signs, date_col="accper_dt"):
    """按时间切 70/30，算训练与测试的 IC，得 过拟合度"""
    ud = np.array(sorted(df[date_col].unique()))
    cut = int(len(ud) * 0.7)
    tr = df[df[date_col].isin(set(ud[:cut]))]
    te = df[df[date_col].isin(set(ud[cut:]))]
    score_tr = eq_score(tr, factors, signs)
    score_te = eq_score(te, factors, signs)
    ic_tr = float(np.mean(ic_series(tr, score_tr)))
    ic_te = float(np.mean(ic_series(te, score_te)))
    overfit = (ic_tr - ic_te) / ic_tr if ic_tr != 0 else np.nan
    return {"ic_train": ic_tr, "ic_test": ic_te, "overfit_ratio": overfit}


def spread_check(df, factors, signs, date_col="accper_dt", q=10, min_n=50):
    """十分位 Q10 − Q1 的平均 label"""
    rows = []
    for _, g in df.groupby(date_col):
        s = g.copy()
        s["_sc"] = eq_score(s, factors, signs)
        s = s.dropna(subset=["_sc", "label"])
        if len(s) < min_n:
            continue
        s["b"] = pd.qcut(s["_sc"].rank(method="first"), q, labels=False, duplicates="drop")
        m = s.groupby("b")["label"].mean()
        if len(m) >= q:
            rows.append(float(m.iloc[-1] - m.iloc[0]))
    return float(np.mean(rows)) if rows else np.nan


def sharpe_from_ic(ic_mean, n_periods, H=60):
    """粗略估算：把 IC 转成年化 Sharpe（等权多空近似）"""
    # E[ret] ≈ IC × σ_cross × 常数；这里用保守近似
    ann = np.sqrt(252 / H)
    return float(ic_mean / 0.12 * np.sqrt(n_periods)) / np.sqrt(n_periods) * ann


def dsr(sharpe, n_trials, T, skew=0.0, kurt=3.0, var_sr=None):
    """
    Deflated Sharpe Ratio（Bailey & López de Prado 2014）

    SR_0 = sqrt(V_SR) × [ (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)) ]
      V_SR = 各试验 Sharpe 的方差；未知时用零假设下的 1/(T−1)
    DSR  = Φ[ (SR − SR_0) × sqrt(T−1) / sqrt(1 − γ3·SR + ((γ4−1)/4)·SR²) ]

    ⚠️ 常见错误：把 SR_0 算成 sqrt(2 ln N) 而忘了除以 sqrt(T)，
       会把门槛抬到 2.0 以上，任何策略都过不了。
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
    z = (sharpe - sr0) * np.sqrt(max(T - 1, 1)) / denom
    return {"sr0": float(sr0), "dsr": float(norm.cdf(z)), "z": float(z),
            "var_sr": float(var_sr)}


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,}")

    out = {}
    for gname, factors, signs in [("全市场组", GROUP_FULL, SIGN_FULL),
                                  ("制造业组", GROUP_MFG, SIGN_MFG)]:
        log(f"\n{'='*50}\n=== {gname} ===\n{'='*50}")
        d = df.dropna(subset=factors + ["label"]).copy()
        log(f"  可用样本 {len(d):,} 行 / {d['accper_dt'].nunique()} 期")

        # G1 单变量：spread
        sp = spread_check(d, factors, signs)
        cost_rt = 2 * COST_ONE_WAY
        g1 = bool(np.isfinite(sp) and sp >= 2 * cost_rt)
        log(f"  G1 spread {sp*100:+.3f}% vs 2×成本 {2*cost_rt*100:.2f}%  → {'✅' if g1 else '❌'}")

        # G2 模型：与基线比（我们用的就是基线，故记为"基线最优，不升级"）
        ov = overfit_check(d, factors, signs)
        log(f"  G2 IC_train {ov['ic_train']:+.4f} / IC_test {ov['ic_test']:+.4f} 过拟合度 {ov['overfit_ratio']:+.3f}")

        # G3 信号：ICIR
        score = eq_score(d, factors, signs)
        ics = ic_series(d, score)
        icir = float(ics.mean() / ics.std()) if ics.std() > 0 else np.nan
        t_stat = float(ics.mean() / ics.std() * np.sqrt(len(ics))) if ics.std() > 0 else np.nan
        g3 = bool(np.isfinite(icir) and abs(icir) > 0.1)
        log(f"  G3 IC {ics.mean():+.4f} ICIR {icir:+.3f} t={t_stat:+.2f} n={len(ics)} → {'✅' if g3 else '❌'}")

        # 置换检验
        log("  置换检验（同日内打乱，200 次）…")
        perm = perm_test(d, factors, signs, n_perm=200)
        log(f"    真实 IC {perm['real_ic']:+.4f}  零分布 mean {perm['null_mean']:+.4f} "
            f"std {perm['null_std']:.4f}  p95 {perm['null_p95']:+.4f}  p={perm['p_value']:.3f}")

        # 分年 IC（关键：看是否失效）
        yearly = {}
        dd = d.copy()
        dd["_sc"] = score
        dd["_y"] = dd["label"]
        dd["yr"] = dd["accper_dt"].dt.year
        for yr, g in dd.groupby("yr"):
            ics_y = []
            for _, gg in g.groupby("accper_dt"):
                s2 = gg[["_sc", "_y"]].dropna()
                if len(s2) < 50:
                    continue
                v = spear(s2["_sc"].values, s2["_y"].values)
                if np.isfinite(v):
                    ics_y.append(v)
            if ics_y:
                yearly[int(yr)] = float(np.mean(ics_y))
        log(f"    分年 IC: " + " ".join(f"{y}:{v:+.3f}" for y, v in sorted(yearly.items())))
        neg_years = sum(1 for v in yearly.values() if v < 0)
        log(f"    负 IC 年数: {neg_years}/{len(yearly)}")

        # 估算 Sharpe 并算 DSR
        sh = sharpe_from_ic(ics.mean(), len(ics))
        ds = dsr(sh, N_TRIALS, len(ics))
        log(f"  估算 Sharpe {sh:+.3f}  SR_0 {ds['sr0']:.3f}  DSR {ds['dsr']:.4f} "
            f"→ {'✅ >0.95' if ds['dsr'] > 0.95 else '❌ <0.95'}")

        out[gname] = {"n_rows": len(d), "n_periods": int(d["accper_dt"].nunique()),
                      "g1_spread": sp, "g1_pass": g1,
                      "overfit": ov, "ic_mean": float(ics.mean()),
                      "icir": icir, "t_stat": t_stat, "n_ic": len(ics),
                      "g3_pass": g3, "perm": perm,
                      "sharpe_est": sh, "dsr": ds,
                      "dsr_pass": bool(ds["dsr"] > 0.95),
                      "yearly_ic": yearly, "neg_years": neg_years}

    (CLEAN / "meta" / "mv_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n  → mv_validation.json")
    print("SUMMARY=" + json.dumps({k: {"ic": v["ic_mean"], "icir": v["icir"],
                                       "p": v["perm"]["p_value"],
                                       "dsr": v["dsr"]["dsr"],
                                       "spread": v["g1_spread"]}
                                    for k, v in out.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
