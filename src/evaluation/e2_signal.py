# -*- coding: utf-8 -*-
"""E2 信号评估：HAC t + 折级汇总 + 十分位 + long-only 分离 + 流动性分桶 + 行业代理"""
import os
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
DATA = ROOT / "data"
H = 60
COST_ONE_WAY = 0.001
N_TRIALS = 15

GROUP_FULL = ["gm_vol", "dgm_q", "gm"]
SIGN_FULL = {"gm_vol": -1, "dgm_q": +1, "gm": +1}

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
    # 行业代理：是否制造业（transfer_intensity 有值 = 披露固定资产附注）
    fp["is_manufacturing"] = fp["transfer_intensity"].notna().astype(int)

    # 标签
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

    # 成交额
    amt_parts = []
    for y in range(2010, 2025):
        f = CLEAN / "quotes" / f"quote_{y}.csv"
        if f.exists():
            amt_parts.append(pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                                         usecols=["code", "date", "amount"]))
    amt = pd.concat(amt_parts, ignore_index=True)
    amt["date"] = pd.to_datetime(amt["date"])
    amt = amt.dropna(subset=["amount"]).drop_duplicates(["code", "date"])
    amt_by = {c: g.sort_values("date")[["date", "amount"]].reset_index(drop=True)
              for c, g in amt.groupby("code", sort=False)}

    out = []
    for code, g in fp.groupby("code", sort=False):
        g = g.sort_values("available_at").reset_index(drop=True)
        ll, aa = lab_by.get(code), amt_by.get(code)
        if ll is None or ll.empty:
            out.append(g.assign(label=np.nan, amount=np.nan)); continue
        m = pd.merge_asof(g, ll, left_on="available_at", right_on="date",
                          direction="forward", tolerance=pd.Timedelta(days=15))
        m = m.drop(columns=["date"])
        if aa is not None and not aa.empty:
            m = pd.merge_asof(m, aa, left_on="available_at", right_on="date",
                              direction="backward", tolerance=pd.Timedelta(days=15))
            m = m.drop(columns=["date"])
        else:
            m["amount"] = np.nan
        out.append(m)
    return pd.concat(out, ignore_index=True)


def newey_west_se(x, lags):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 3: return np.nan
    xd = x - x.mean()
    se2 = float(np.dot(xd, xd) / n)
    L = min(lags, n - 2)
    for j in range(1, L + 1):
        gj = float(np.dot(xd[j:], xd[:-j]) / n)
        se2 += 2.0 * (1.0 - j / (L + 1.0)) * gj
    return float(np.sqrt(max(se2, 1e-12) / n))


def hac_stats(ics, lags=H):
    ics = np.asarray(ics, dtype=float); ics = ics[np.isfinite(ics)]
    if len(ics) < 3: return {}
    se_n = ics.std(ddof=1) / np.sqrt(len(ics))
    se_h = newey_west_se(ics, lags)
    m = ics.mean()
    return {"mean": float(m), "std": float(ics.std(ddof=1)),
            "icir": float(m / ics.std(ddof=1)) if ics.std(ddof=1) > 0 else np.nan,
            "t_naive": float(m / se_n) if se_n > 0 else np.nan,
            "t_hac": float(m / se_h) if se_h and se_h > 0 else np.nan,
            "se_naive": float(se_n), "se_hac": float(se_h),
            "inflation": float(se_h / se_n) if se_n > 0 else np.nan,
            "n": int(len(ics)), "median": float(np.median(ics)),
            "iqr": float(np.percentile(ics, 75) - np.percentile(ics, 25)),
            "worst": float(ics.min()), "best": float(ics.max()),
            "sign_consistency": float((np.sign(ics) == np.sign(m)).mean()),
            "pos_ratio": float((ics > 0).mean())}


def ic_series(d, col, label="label", date_col="accper_dt", min_n=50):
    ics = []
    for _, g in d.groupby(date_col):
        s = g[[col, label]].dropna()
        if len(s) < min_n: continue
        v = spear(s[col].values, s[label].values)
        if np.isfinite(v): ics.append(v)
    return np.array(ics)


def fold_ic(ics, n_folds=5):
    n = len(ics)
    if n < n_folds * 2: return []
    folds = np.array_split(np.arange(n), n_folds)
    return [{"fold": i+1, "mean": float(ics[f].mean()), "n": int(len(f))}
            for i, f in enumerate(folds)]


def decile_analysis(d, col, label="label", date_col="accper_dt", q=10, min_n=50):
    rows, full = [], []
    for _, g in d.groupby(date_col):
        s = g[[col, label]].dropna()
        if len(s) < min_n: continue
        full.append(float(s[label].mean()))
        s = s.assign(b=pd.qcut(s[col].rank(method="first"), q,
                               labels=False, duplicates="drop"))
        rows.append(s.groupby("b")[label].mean())
    if not rows: return None
    dec = pd.concat(rows, axis=1).mean(axis=1)
    fm = float(np.mean(full)); top, bot = float(dec.iloc[-1]), float(dec.iloc[0])
    diffs = np.diff(dec.values); diffs = diffs[np.isfinite(diffs)]
    return {"deciles": {int(k): float(v) for k, v in dec.items()},
            "full_mean": fm, "spread": top - bot, "long_excess": top - fm,
            "short_contrib": fm - bot,
            "short_share": (fm - bot) / (top - bot) if (top - bot) != 0 else np.nan,
            "monotonic": float((np.sign(diffs) == np.sign(diffs[0])).mean()) if len(diffs) else np.nan}


def liquidity_buckets(d, col, label="label", amount_col="amount",
                      date_col="accper_dt", n_b=3, min_n=40):
    d = d.dropna(subset=[amount_col])
    if len(d) < 1000: return {}
    out = {}
    for b in range(n_b):
        ics, sps = [], []
        for _, g in d.groupby(date_col):
            s = g[[col, label, amount_col]].dropna()
            if len(s) < min_n * n_b: continue
            s = s.assign(lb=pd.qcut(s[amount_col].rank(method="first"), n_b,
                                    labels=False, duplicates="drop"))
            gg = s[s["lb"] == b]
            if len(gg) < min_n: continue
            v = spear(gg[col].values, gg[label].values)
            if np.isfinite(v): ics.append(v)
            ss = gg.assign(q=pd.qcut(gg[col].rank(method="first"), 10,
                                     labels=False, duplicates="drop"))
            qm = ss.groupby("q")[label].mean()
            if len(qm) >= 10: sps.append(float(qm.iloc[-1] - qm.iloc[0]))
        out[f"bucket_{b+1}_{['低','中','高'][b]}成交额"] = {
            "ic": float(np.mean(ics)) if ics else np.nan,
            "spread": float(np.mean(sps)) if sps else np.nan,
            "n_ic": len(ics)}
    return out


def turnover_proxy(d, col, date_col="accper_dt", top_pct=0.3):
    prev, turns = None, []
    for _, g in d.groupby(date_col):
        s = g[[col, "code"]].dropna()
        if len(s) < 100: continue
        k = max(int(len(s) * top_pct), 1)
        cur = set(s.nlargest(k, col)["code"])
        if prev is not None:
            turns.append(len(cur - prev) / max(len(cur), 1))
        prev = cur
    return float(np.mean(turns)) if turns else np.nan


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,} / 有成交额 {df['amount'].notna().sum():,}")

    # 构造合成分数
    # ⚠️ 修正 1：只对「可用因子」求平均（不固定除以 3）
    # ⚠️ 修正 2（更严重）：rank 必须在「期内」做，不能跨期
    #    原先 `df[f].rank(pct=True)` 在整面板上 rank → 分数含时间趋势
    if "composite_score_panel" not in dir():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from metrics import composite_score_panel
    df["score_full"] = composite_score_panel(df, GROUP_FULL, SIGN_FULL, "accper_dt")

    targets = [
        ("dgm_q（E1 唯一 PROCEED）", "dgm_q", None),
        ("全市场组合成（3 因子）", "score_full", GROUP_FULL),
        ("gm（E1 STOP，作对照）", "gm", None),
        ("gm_vol（E1 STOP，应归 state）", "gm_vol", None),
        ("transfer_intensity（E1 STOP）", "transfer_intensity", None),
        ("dep_rate（E1 STOP）", "dep_rate", None),
    ]

    res = {}
    for name, col, need in targets:
        sub = df.dropna(subset=[label] if (label := "label") else []) if False else df
        d = df[[c for c in set([col, "label", "code", "accper_dt", "amount",
                                "is_manufacturing"] + (need or []))
                if c in df.columns]].dropna(subset=[col, "label"])
        if len(d) < 500:
            log(f"  {name}: 样本不足 ({len(d)})"); continue
        log(f"\n=== {name} ===")
        log(f"  样本 {len(d):,} 行 / {d['accper_dt'].nunique()} 期")

        ics = ic_series(d, col)
        st = hac_stats(ics, lags=H)
        log(f"  IC {st['mean']:+.4f}  ICIR {st['icir']:+.3f}  n={st['n']}")
        log(f"  t_naive {st['t_naive']:+.2f} → t_hac {st['t_hac']:+.2f}  "
            f"(SE 膨胀 {st['inflation']:.2f}×)")
        log(f"  中位 {st['median']:+.4f}  IQR {st['iqr']:.4f}  最差折 {st['worst']:+.4f}  同号 {st['sign_consistency']*100:.0f}%")

        folds = fold_ic(ics)
        log(f"  分折: " + " ".join(f"折{f['fold']}:{f['mean']:+.3f}" for f in folds))

        dec = decile_analysis(d, col)
        if dec:
            log(f"  十分位: " + " ".join(f"{k}:{v*100:+.1f}%" for k, v in dec["deciles"].items()))
            log(f"  多空价差 {dec['spread']*100:+.3f}%  多头超额 {dec['long_excess']*100:+.3f}%  "
                f"空头贡献 {dec['short_contrib']*100:+.3f}%  空头占比 {dec['short_share']*100:.0f}%")
            log(f"  单调性 {dec['monotonic']*100:.0f}% 同号")

        liq = liquidity_buckets(d, col)
        if liq:
            log(f"  流动性分桶: " + " ".join(f"{k.split('_',2)[-1]}:{v['ic']:+.4f}" for k, v in liq.items()))

        # 行业内 IC（用制造业代理）
        ind = {}
        for g_val, g_name in [(0, "非制造业"), (1, "制造业")]:
            dd = d[d["is_manufacturing"] == g_val]
            if len(dd) < 300: continue
            ics_ind = ic_series(dd, col)
            if len(ics_ind):
                ind[g_name] = float(ics_ind.mean())
        if ind:
            log(f"  行业代理分组 IC: " + " ".join(f"{k}:{v:+.4f}" for k, v in ind.items()))

        res[name] = {"column": col, "n_rows": len(d),
                     "n_periods": int(d["accper_dt"].nunique()),
                     "hac": st, "folds": folds,
                     "decile": dec, "liquidity": liq, "industry": ind,
                     "turnover": turnover_proxy(d, col)}

    # 多重检验门槛
    thr = {"classic_5pct": 1.96, "new_factor": 3.0,
           "bonferroni_N15": float(norm.ppf(1 - 0.05 / (2 * N_TRIALS)))}
    log(f"\n=== 多重检验门槛（N={N_TRIALS}）===")
    for k, v in thr.items():
        log(f"  {k}: |t| > {v:.3f}")

    (CLEAN / "meta" / "e2_signal.json").write_text(
        json.dumps({"results": res, "thresholds": thr, "H": H,
                    "cost_one_way": COST_ONE_WAY},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("\n  → e2_signal.json")


if __name__ == "__main__":
    main()
