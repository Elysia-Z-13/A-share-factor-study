# -*- coding: utf-8 -*-
"""E4 策略评估（A 股 long-only）"""
import os
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

# metrics.py 所在目录（仓库的 src/）
S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
import metrics as M

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
H = M.H
COST = M.COST_ONE_WAY
TOP_PCT = 0.30
HOLD = 3        # 60日持有 / 20日调仓 ≈ 3 个 cohort

COMBOS = {
    "2因子(gm_vol+dgm_q)": (["gm_vol", "dgm_q"], {"gm_vol": -1, "dgm_q": +1}),
    "3因子(+gm)":          (["gm_vol", "dgm_q", "gm"],
                            {"gm_vol": -1, "dgm_q": +1, "gm": +1}),
}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


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

    flags = pd.read_csv(CLEAN / "meta" / "universe_flags.csv",
                        encoding="utf-8-sig", dtype={"code": str})
    flags["date"] = pd.to_datetime(flags["date"])
    fl_by = {c: g.sort_values("date")[["date", "tradable", "hit_limit_up"]].reset_index(drop=True)
             for c, g in flags.groupby("code", sort=False)}

    out = []
    for code, g in fp.groupby("code", sort=False):
        g = g.sort_values("available_at").reset_index(drop=True)
        ll, ff = lab_by.get(code), fl_by.get(code)
        if ll is None or ll.empty:
            out.append(g.assign(label=np.nan, tradable=np.nan, hit_limit_up=np.nan))
            continue
        m = pd.merge_asof(g, ll, left_on="available_at", right_on="date",
                          direction="forward", tolerance=pd.Timedelta(days=15))
        m = m.drop(columns=["date"])
        if ff is not None and not ff.empty:
            m = pd.merge_asof(m, ff, left_on="available_at", right_on="date",
                              direction="backward", tolerance=pd.Timedelta(days=15))
            m = m.drop(columns=["date"], errors="ignore")
        out.append(m)
    return pd.concat(out, ignore_index=True)


def run_cohorts(d, factors, signs, date_col="accper_dt"):
    d = d.copy()
    d["score"] = M.composite_score_panel(d, factors, signs, date_col)
    cohorts, prev = [], None
    for t, g in d.groupby(date_col):
        s = g.dropna(subset=["score", "label"])
        if "tradable" in s.columns:
            s = s[s["tradable"].fillna(False).astype(bool)]
        if "hit_limit_up" in s.columns:
            s = s[~s["hit_limit_up"].fillna(False).astype(bool)]
        if len(s) < 100:
            continue
        k = max(int(len(s) * TOP_PCT), 1)
        picks = s.nlargest(k, "score")
        cur = set(picks["code"])
        turn = 1.0 if prev is None else 0.5 * (
            len(cur - prev) / max(len(cur), 1) + len(prev - cur) / max(len(prev), 1))
        cohorts.append({"date": t, "k": k, "n_univ": len(s), "turn": float(turn),
                        "gross": float(picks["label"].mean()),
                        "bench": float(s["label"].mean())})
        prev = cur
    return pd.DataFrame(cohorts)


def combine(coh, hold=HOLD):
    rows = []
    for i in range(hold - 1, len(coh)):
        w = coh.iloc[i - hold + 1: i + 1]
        g = float(w["gross"].mean()); b = float(w["bench"].mean())
        turn = float(w["turn"].mean()); cost = turn * 2 * COST
        net = (1 + g) * (1 - cost) - 1
        rows.append({"date": coh.iloc[i]["date"], "gross": g, "net": net,
                     "bench": b, "turn": turn, "cost": cost})
    return pd.DataFrame(rows)


def metrics_block(df, ppy=4, label=""):
    r = df["net"].values; b = df["bench"].values
    gr = df["gross"].values
    ex = r - b
    n = len(r)
    ann = (1 + r).prod() ** (ppy / n) - 1
    ann_g = (1 + gr).prod() ** (ppy / n) - 1
    ann_b = (1 + b).prod() ** (ppy / n) - 1
    nav = np.cumprod(1 + r); peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    return {"label": label, "n": n,
            "ann_gross": float(ann_g), "ann_net": float(ann), "ann_bench": float(ann_b),
            "ann_excess_net": float(ann - ann_b),
            "excess_mean_per_period": float(ex.mean()),
            "vol_excess": float(ex.std(ddof=1) * np.sqrt(ppy)),
            "sharpe_excess": float(ex.mean() / ex.std(ddof=1) * np.sqrt(ppy))
                             if ex.std(ddof=1) > 0 else np.nan,
            "mdd_net": mdd,
            "calmar": float(ann / abs(mdd)) if mdd != 0 else np.nan,
            "win_rate_excess": float((ex > 0).mean()),
            "avg_turnover": float(df["turn"].mean()),
            "ann_cost": float(df["cost"].mean() * ppy)}


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,} / tradable {df['tradable'].fillna(False).sum():,.0f}")

    results = {}
    for name, (factors, signs) in COMBOS.items():
        log(f"\n=== {name} ===")
        coh = run_cohorts(df, factors, signs)
        log(f"  cohorts {len(coh)}  平均选中 {coh['k'].mean():.0f} 只  平均换手 {coh['turn'].mean()*100:.1f}%")
        bt = combine(coh)
        mb = metrics_block(bt, label=name)
        log(f"  年化毛 {mb['ann_gross']*100:+.2f}%  年化净 {mb['ann_net']*100:+.2f}%  "
            f"基准 {mb['ann_bench']*100:+.2f}%")
        log(f"  净超额 {mb['ann_excess_net']*100:+.2f}%  Sharpe(excess) {mb['sharpe_excess']:+.2f}  "
            f"MDD {mb['mdd_net']*100:+.1f}%")
        log(f"  换手 {mb['avg_turnover']*100:.1f}%  年成本 {mb['ann_cost']*100:.2f}%  "
            f"胜率(excess) {mb['win_rate_excess']*100:.0f}%")

        # 分年度
        bt2 = bt.copy(); bt2["year"] = bt2["date"].dt.year
        yr = bt2.groupby("year").agg(net=("net", "mean"), bench=("bench", "mean"),
                                     gross=("gross", "mean"))
        yr["excess"] = yr["net"] - yr["bench"]
        log(f"  分年度净超额: " + " ".join(f"{y}:{v*100:+.1f}%" for y, v in yr["excess"].items()))

        # 压力：成本 ×2/×3
        stress = {}
        for mult in (1, 2, 3):
            c = bt["turn"].mean() * 2 * COST * mult
            n2 = (1 + bt["gross"]) * (1 - c) - 1
            stress[f"cost_x{mult}"] = float(
                ((1 + n2).prod() ** (4 / len(n2)) - 1) - mb["ann_bench"])
        log(f"  成本压力: " + " ".join(f"×{k[-1]}:{v*100:+.2f}%" for k, v in stress.items()))

        results[name] = {"metrics": mb, "yearly": yr.to_dict(),
                         "stress_cost": stress,
                         "avg_k": float(coh["k"].mean()),
                         "n_cohorts": len(coh),
                         "bt_records": bt.assign(date=bt["date"].astype(str)).to_dict("records")}

    (CLEAN / "meta" / "e4_backtest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("\n  → e4_backtest.json")


if __name__ == "__main__":
    main()
