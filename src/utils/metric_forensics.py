# -*- coding: utf-8 -*-
"""口径差异取证：把 M-S 的 0.0402 与 E3 的 0.0512 之间的 0.0110 差距分解"""
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
FACTORS = ["gm_vol", "dgm_q", "gm"]
SIGNS = M.SIGNS_FULL

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


def score_fixed_denominator(g, factors, signs):
    """M-S 的老口径：固定除以 len(factors)"""
    return sum(g[f].rank(pct=True) * signs[f] for f in factors) / len(factors)


def score_available(g, factors, signs):
    """新口径：除以可用因子数"""
    return M.composite_score(g, factors, signs)


def ic_full(d, scorer, dropna_all=None):
    """全样本 IC（不按折）"""
    dd = d if dropna_all is None else d.dropna(subset=list(dropna_all))
    ics = []
    for _, g in dd.groupby("accper_dt"):
        sc = scorer(g, FACTORS, SIGNS)
        s = pd.DataFrame({"sc": sc, "y": g["label"]}).dropna()
        if len(s) < M.MIN_N:
            continue
        v = M.spear(s["sc"].values, s["y"].values)
        if np.isfinite(v):
            ics.append(v)
    return np.array(ics)


def ic_foldwise(d, scorer, n_folds=5, dropna_all=None):
    """M-S 的口径：按 5 折算，只在测试折上算"""
    dd = d if dropna_all is None else d.dropna(subset=list(dropna_all))
    ud = np.array(sorted(dd["accper_dt"].unique()))
    folds = np.array_split(ud, n_folds)
    ics = []
    for fold in folds:
        te = dd[dd["accper_dt"].isin(set(fold))]
        for _, g in te.groupby("accper_dt"):
            sc = scorer(g, FACTORS, SIGNS)
            s = pd.DataFrame({"sc": sc, "y": g["label"]}).dropna()
            if len(s) < M.MIN_N:
                continue
            v = M.spear(s["sc"].values, s["y"].values)
            if np.isfinite(v):
                ics.append(v)
    return np.array(ics)


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行")

    # 四种组合：{固定分母, 可用数分母} × {完全覆盖, 部分覆盖}
    variants = {
        "A 固定分母 + 完全覆盖（M-S 老口径）":      (score_fixed_denominator, FACTORS),
        "B 固定分母 + 部分覆盖":                    (score_fixed_denominator, None),
        "C 可用数分母 + 完全覆盖":                  (score_available, FACTORS),
        "D 可用数分母 + 部分覆盖（新口径）":         (score_available, None),
    }
    log("\n=== 全样本口径（不按折）===")
    full_res = {}
    for name, (scorer, dropna_all) in variants.items():
        ics = ic_full(df, scorer, dropna_all)
        st = M.hac_stats(ics)
        full_res[name] = st
        log(f"  {name:34s} IC {st['mean']:+.4f}  n={st['n']}")

    log("\n=== 折内口径（按 5 折，M-S 的做法）===")
    fold_res = {}
    for name, (scorer, dropna_all) in variants.items():
        ics = ic_foldwise(df, scorer, dropna_all=dropna_all)
        st = M.hac_stats(ics)
        fold_res[name] = st
        log(f"  {name:34s} IC {st['mean']:+.4f}  n={st['n']}")

    # 分解
    log("\n=== 差距分解 ===")
    a = full_res["A 固定分母 + 完全覆盖（M-S 老口径）"]["mean"]
    b = full_res["B 固定分母 + 部分覆盖"]["mean"]
    c = full_res["C 可用数分母 + 完全覆盖"]["mean"]
    dd_ = full_res["D 可用数分母 + 部分覆盖（新口径）"]["mean"]
    log(f"  A(固定+完全) = {a:+.4f}")
    log(f"  B(固定+部分) = {b:+.4f}   ← 换样本范围：{b-a:+.4f}")
    log(f"  C(可用+完全) = {c:+.4f}   ← 换分母：{c-a:+.4f}")
    log(f"  D(可用+部分) = {dd_:+.4f}   ← 两者都换：{dd_-a:+.4f}")
    log(f"  （交叉项 = {dd_-a - (b-a) - (c-a):+.4f}）")

    log("\n=== 折内 vs 全样本 ===")
    for name in variants:
        f_ = full_res[name]["mean"]
        w_ = fold_res[name]["mean"]
        log(f"  {name:34s} 全样本 {f_:+.4f}  折内 {w_:+.4f}  差 {w_-f_:+.4f}")

    out = {"full_sample": {k: v for k, v in full_res.items()},
           "foldwise": {k: v for k, v in fold_res.items()},
           "decomposition": {"A": a, "B": b, "C": c, "D": dd_,
                             "sample_effect": b - a,
                             "denominator_effect": c - a,
                             "combined": dd_ - a,
                             "interaction": dd_ - a - (b - a) - (c - a)}}
    (CLEAN / "meta" / "metric_forensics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("\n  → metric_forensics.json")


if __name__ == "__main__":
    main()
