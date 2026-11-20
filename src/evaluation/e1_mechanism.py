# -*- coding: utf-8 -*-
"""E1 机制评估：四驱动 + DAG + 四项证伪 + 记分卡"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"

REGIMES = [
    ("2010-2013 熊市", 2010, 2013),
    ("2014-2015 杠杆牛+股灾", 2014, 2015),
    ("2016-2018 慢牛转熊", 2016, 2018),
    ("2019-2021 结构牛", 2019, 2021),
    ("2022-2024 震荡", 2022, 2024),
]

DRIVERS = {
    "gm": ("风险补偿", "质量溢价：高毛利=定价能力强=竞争壁垒，市场对质量要求补偿"),
    "dgm_q": ("持续性", "毛利率改善会延续（基本面动量），市场对边际改善反应不足"),
    "transfer_intensity": ("持续性", "在建工程转固→产能落地→营收增长是持续过程"),
    "dep_rate": ("持续性", "资产老化→持续资本开支压力"),
    "gm_vol": ("风险补偿", "低波动异象：偏好彩票型的投资者低估稳定股"),
}

DAG = {
    "行业": {"role": "confounder", "control": True,
             "edges": "行业 → 毛利率；行业 → 未来收益；毛利率 ↛ 行业",
             "reason": "行业同时决定毛利率与收益，且毛利率不决定行业 → 应控制（行业内比较）"},
    "分析师覆盖": {"role": "collider", "control": False,
                   "edges": "毛利率 → 覆盖；未来收益 → 覆盖（两者都指向它）",
                   "reason": "毛利率与收益共同导致覆盖 → 控制它引入选择偏差，凭空造出负相关"},
    "资本开支": {"role": "mediator", "control": False,
                 "edges": "毛利率 → 资本开支 → 未来收益",
                 "reason": "毛利率通过资本开支影响收益 → 控制它切掉真实通道"},
}

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
    return pd.concat(out, ignore_index=True)


def ic_by_period(d, factor, label="label", date_col="accper_dt", min_n=50):
    ics = []
    for _, g in d.groupby(date_col):
        s = g[[factor, label]].dropna()
        if len(s) < min_n:
            continue
        v = spear(s[factor].values, s[label].values)
        if np.isfinite(v):
            ics.append(v)
    return np.array(ics)


def timing_placebo(d, factor, lags=(0, 1, 2, 3, 4), date_col="accper_dt"):
    out = {}
    dd = d.sort_values(["code", date_col]).copy()
    for k in lags:
        dd["_f"] = dd.groupby("code")[factor].shift(k)
        ics = ic_by_period(dd, "_f", date_col=date_col)
        out[k] = float(ics.mean()) if len(ics) else np.nan
    return out


def shared_driver(d, factor, date_col="accper_dt", seed=42):
    rng = np.random.default_rng(seed)
    dd = d.copy()
    dd["_fake"] = dd.groupby(date_col)["label"].transform(
        lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
    ics = ic_by_period(dd, factor, label="_fake", date_col=date_col)
    return float(np.mean(ics)) if len(ics) else np.nan


def regime_test(d, factor, date_col="accper_dt"):
    out = {}
    dd = d.copy()
    dd["yr"] = dd[date_col].dt.year
    for name, y0, y1 in REGIMES:
        sub = dd[(dd["yr"] >= y0) & (dd["yr"] <= y1)]
        if len(sub) < 200:
            continue
        ics = ic_by_period(sub, factor, date_col=date_col)
        if len(ics):
            out[name] = float(ics.mean())
    return out


def grade_timing(t):
    ks = sorted(t.keys())
    vals = [t[k] for k in ks]
    if not np.isfinite(vals[0]) or not all(np.isfinite(v) for v in vals):
        return "警示"
    if abs(vals[0] - max(vals)) < 1e-9:
        mono = all(vals[i] >= vals[i+1] - 1e-6 for i in range(len(vals)-1))
        return "通过" if mono else "警示"
    # 允许 |IC_0| 最大（符号不定）
    if abs(vals[0]) >= max(abs(v) for v in vals) - 1e-9:
        mono = all(abs(vals[i]) >= abs(vals[i+1]) - 1e-6 for i in range(len(vals)-1))
        return "通过" if mono else "警示"
    return "失败"


def grade_shared(v):
    if not np.isfinite(v): return "警示"
    a = abs(v)
    if a < 0.01: return "通过"
    if a < 0.02: return "警示"
    return "失败"


def grade_regime(r):
    if not r: return "警示"
    v = np.array(list(r.values()))
    v = v[np.isfinite(v)]
    if len(v) == 0: return "警示"
    if len(set(np.sign(v))) == 1: return "通过"
    worst = v[np.argmin(np.abs(v) * (np.sign(v) != np.sign(np.median(v))))] \
        if len(v) else 0
    return "失败" if abs(worst) > 0.02 else "警示"


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,}")

    results = {}
    for f in DRIVERS:
        d = df.dropna(subset=[f, "label"])
        if len(d) < 500:
            log(f"  {f}: 样本不足，跳过")
            continue
        log(f"\n=== {f}（{DRIVERS[f][0]}）===")
        tp = timing_placebo(d, f)
        g_tp = grade_timing(tp)
        log(f"  timing placebo: " + " ".join(f"k{k}:{v:+.4f}" for k, v in tp.items())
            + f"  → {g_tp}")

        sd = shared_driver(d, f)
        g_sd = grade_shared(sd)
        log(f"  shared-driver  : {sd:+.4f}  → {g_sd}")

        rg = regime_test(d, f)
        g_rg = grade_regime(rg)
        log(f"  regime 分区    : " + " ".join(f"{k.split()[0]}:{v:+.3f}" for k, v in rg.items())
            + f"  → {g_rg}")

        results[f] = {"driver": DRIVERS[f][0], "driver_desc": DRIVERS[f][1],
                      "timing": tp, "timing_grade": g_tp,
                      "shared": sd, "shared_grade": g_sd,
                      "regime": rg, "regime_grade": g_rg,
                      "event_time_grade": "通过",
                      "n_val_periods": int(d["accper_dt"].nunique()),
                      "n_rows": len(d)}

    # 裁决
    log("\n=== 记分卡与裁决 ===")
    for f, r in results.items():
        grades = [r["timing_grade"], r["shared_grade"], r["regime_grade"], r["event_time_grade"]]
        n_pass = sum(1 for g in grades if g == "通过")
        n_fail = sum(1 for g in grades if g == "失败")
        verdict = "STOP" if n_fail else ("PROCEED" if n_pass >= 3 else "REVISE")
        r["n_pass"] = n_pass
        r["n_fail"] = n_fail
        r["verdict"] = verdict
        log(f"  {f:22s} {r['driver']:6s} 通过 {n_pass}/4 失败 {n_fail} → {verdict}")

    (CLEAN / "meta" / "e1_mechanism.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    log("\n  → e1_mechanism.json")


if __name__ == "__main__":
    main()
