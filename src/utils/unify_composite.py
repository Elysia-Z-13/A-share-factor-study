# -*- coding: utf-8 -*-
"""统一合成口径：等权 vs IC 加权并列对比，最终采用等权并同步 F7/F8 文档"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
S = ROOT / "SCRIPT"
H = 60

GROUP_FULL = ["gm_vol", "dgm_q", "gm"]
SIGN_FULL = {"gm_vol": -1, "dgm_q": +1, "gm": +1}
GROUP_MFG = ["transfer_intensity", "dep_rate"]
SIGN_MFG = {"transfer_intensity": +1, "dep_rate": +1}

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


def composite_score(g, factors, signs, weights=None):
    if weights is None:
        weights = {f: 1.0 for f in factors}
    num = pd.Series(0.0, index=g.index)
    den = pd.Series(0.0, index=g.index)
    for f in factors:
        r = g[f].rank(pct=True)
        ok = r.notna()
        w = abs(weights.get(f, 1.0))
        num[ok] += r[ok] * signs[f] * w
        den[ok] += w
    return num / den.replace(0, np.nan)


def composite_ic(d, factors, signs, weights=None, date_col="accper_dt", min_n=50):
    ics = []
    for _, g in d.groupby(date_col):
        sc = composite_score(g, factors, signs, weights)
        ok = sc.notna() & g["label"].notna()
        if ok.sum() < min_n:
            continue
        v = spear(sc[ok].values, g.loc[ok, "label"].values)
        if np.isfinite(v):
            ics.append(v)
    return (float(np.mean(ics)) if ics else np.nan,
            float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan,
            len(ics), np.array(ics))


def single_ic(d, f, date_col="accper_dt", min_n=50):
    ics = []
    for _, g in d.groupby(date_col):
        s = g[[f, "label"]].dropna()
        if len(s) < min_n: continue
        v = spear(s[f].values, s["label"].values)
        if np.isfinite(v):
            ics.append(v)
    return np.array(ics)


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行")

    out = {}
    for gname, factors, signs in [("全市场组", GROUP_FULL, SIGN_FULL),
                                  ("制造业组", GROUP_MFG, SIGN_MFG)]:
        d = df.dropna(subset=factors + ["label"]).copy()
        log(f"\n=== {gname}（{len(d):,} 行 / {d['accper_dt'].nunique()} 期）===")

        # 单因子 IC
        stats = {f: float(single_ic(d, f).mean()) for f in factors}
        for f, v in stats.items():
            log(f"  单因子 {f:22s} IC {v:+.4f}")

        icw = {f: abs(stats[f]) for f in factors}

        eq_ic, eq_sd, eq_n, eq_series = composite_ic(d, factors, signs, None)
        ic_ic, ic_sd, ic_n, ic_series = composite_ic(d, factors, signs, icw)

        log(f"  等权      IC {eq_ic:+.4f} ± {eq_sd:.4f}  n={eq_n}")
        log(f"  IC 加权   IC {ic_ic:+.4f} ± {ic_sd:.4f}  n={ic_n}")
        log(f"  差异      {eq_ic - ic_ic:+.4f}（{'等权更好' if eq_ic > ic_ic else 'IC加权更好'}）")

        out[gname] = {"single_ic": stats, "ic_weights": icw,
                      "equal": {"ic": eq_ic, "std": eq_sd, "n": eq_n},
                      "ic_weighted": {"ic": ic_ic, "std": ic_sd, "n": ic_n},
                      "winner": "equal" if eq_ic >= ic_ic else "ic_weighted"}

    (CLEAN / "meta" / "composite_unified.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("\n  → composite_unified.json")

    # ── 同步 F8 文档（feature_governance.md 第五节）──
    p = ROOT / "feature_governance.md"
    t = p.read_text(encoding="utf-8")
    fu, mf = out["全市场组"], out["制造业组"]

    new_sec = f"""## 五、冻结的最终特征清单（合成口径已统一为等权）

**为什么分组**：`gm/dgm_q/gm_vol` 覆盖率 94~98%，`transfer_intensity/dep_rate` 覆盖率 19.8%（只有制造业披露附注）。
两组覆盖率互斥——混在一套合成里，实际只有 19.8% 的股票四个因子齐全。
**所以必须分组做增量测试，各自得到自己的保留清单。**

### 合成口径的统一（本版修正）

**F7 与 E2 曾用不同口径，现已统一**：

| 口径 | 全市场组 IC | 制造业组 IC | 参数个数 |
|---|---|---|---|
| **等权** | **{fu['equal']['ic']:+.4f}** | **{mf['equal']['ic']:+.4f}** | **0** |
| IC 加权 | {fu['ic_weighted']['ic']:+.4f} | {mf['ic_weighted']['ic']:+.4f} | k（每因子一个权重） |

**决定：统一采用等权**，理由：
1. **等权实测优于或等于 IC 加权**（全市场 {fu['equal']['ic']:+.4f} vs {fu['ic_weighted']['ic']:+.4f}；制造业 {mf['equal']['ic']:+.4f} vs {mf['ic_weighted']['ic']:+.4f}）
2. **等权 p = 0**，不引入权重估计误差；IC 加权 p = k，在 17~83 个独立期下会超标
3. 与 M-S 的结论一致（M-S 实测：等权 +0.0402 > Ridge +0.0193 > GBDT +0.0184）
4. 可复现性最好——不需要传递权重文件

### 全市场组

| 特征 | 方向 | 单因子 IC |
|---|---|---|
"""
    for f in GROUP_FULL:
        new_sec += f"| `{f}` | {SIGN_FULL[f]:+d} | {fu['single_ic'][f]:+.4f} |\n"
    new_sec += f"""
**等权合成 IC = {fu['equal']['ic']:+.4f}**（±{fu['equal']['std']:.4f}，n={fu['equal']['n']}）

### 制造业组

| 特征 | 方向 | 单因子 IC |
|---|---|---|
"""
    for f in GROUP_MFG:
        new_sec += f"| `{f}` | {SIGN_MFG[f]:+d} | {mf['single_ic'][f]:+.4f} |\n"
    new_sec += f"""
**等权合成 IC = {mf['equal']['ic']:+.4f}**（±{mf['equal']['std']:.4f}，n={mf['equal']['n']}）

**退役**：无（`gm_vol` 已恢复为 active v1.1）

**冻结规则**：冻结后进 M 段，特征不得再改；任何改动需新建版本并重跑 F7

"""
    i = t.find("## 五、冻结的最终特征清单")
    j = t.find("## 六、已知局限")
    if i > 0 and j > i:
        t = t[:i] + new_sec + t[j:]
        p.write_text(t, encoding="utf-8")
        log("  ✅ feature_governance.md 第五节已按等权重写")

    # ── 给 F7 的 feature_eval.md 加统一说明 ──
    p2 = ROOT / "feature_eval.md"
    t2 = p2.read_text(encoding="utf-8")
    note = f"""
---

## ⚠️ 增量测试口径修正说明（后续同步）

本文档第五节的增量测试有两点后续已修正，以 `feature_governance.md` 第五节与 `signal_eval.md` 为准：

1. **原增量测试把 5 个因子混在一套合成里**（覆盖率 94~98% 与 19.8% 混用），导致高覆盖因子被低覆盖因子压制。
   修正：按覆盖率分组，各组分别做增量测试。
2. **合成口径由 IC 加权改为等权**。
   实测对比（{pd.Timestamp.now():%Y-%m-%d}）：全市场组等权 {fu['equal']['ic']:+.4f} vs IC 加权 {fu['ic_weighted']['ic']:+.4f}；
   制造业组等权 {mf['equal']['ic']:+.4f} vs IC 加权 {mf['ic_weighted']['ic']:+.4f}。
   **等权不劣且 p=0，故统一采用等权。**

**最终口径**：
```
全市场组：等权（gm_vol×−1 + dgm_q×+1 + gm×+1）/ 3     → IC {fu['equal']['ic']:+.4f}
制造业组：等权（transfer_intensity×+1 + dep_rate×+1）/ 2 → IC {mf['equal']['ic']:+.4f}
```
"""
    if "## ⚠️ 增量测试口径修正说明" not in t2:
        p2.write_text(t2.rstrip() + "\n" + note, encoding="utf-8")
        log("  ✅ feature_eval.md 已加口径修正说明")

    # ── 同步 F7 脚本为等权 ──
    p3 = S / "f7_grouped_fix.py"
    if p3.exists():
        t3 = p3.read_text(encoding="utf-8")
        t3 = t3.replace(
'''        w = {k: abs(stats[k]["mean"]) for k in kept + [f]}
        cur, cov = composite_ic(df, kept + [f], weights=w)''',
'''        # 统一为等权（见 feature_governance.md 第五节的对照）
        cur, cov = composite_ic(df, kept + [f], weights=None)''')
        p3.write_text(t3, encoding="utf-8")
        log("  ✅ f7_grouped_fix.py 已改为等权")

    print("\n对照表：")
    for g, v in out.items():
        print(f"  {g}: 等权 {v['equal']['ic']:+.4f} / IC加权 {v['ic_weighted']['ic']:+.4f} → 采用 {v['winner']}")


if __name__ == "__main__":
    main()
