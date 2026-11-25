# -*- coding: utf-8 -*-
"""F7 修正：按样本组分别做增量测试（覆盖率互斥的因子不能混在一套合成里）"""
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
H = 60
SPLIT = pd.Timestamp("2025-01-01")

GROUPS = {
    "全市场组（覆盖率≥90%）": ["gm", "dgm_q", "gm_vol"],
    "制造业组（覆盖率≈19.8%）": ["transfer_intensity", "dep_rate"],
}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def spear(x, y):
    rx = pd.Series(x).rank().to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank().to_numpy(dtype=float, copy=True)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float((rx*ry).sum()/den) if den > 0 else np.nan


def legal_deadline(accper):
    y, m = accper.year, accper.month
    if m == 12: return pd.Timestamp(y + 1, 4, 30)
    if m == 3:  return pd.Timestamp(y, 4, 30)
    if m == 6:  return pd.Timestamp(y, 8, 31)
    if m == 9:  return pd.Timestamp(y, 10, 31)
    return accper + pd.Timedelta(days=120)


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
    fp = fp[fp["available_at"] < SPLIT]      # 训练窗
    return attach_label(fp)


def attach_label(fp):
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
        m = m.drop(columns=["date"])
        out.append(m)
    return pd.concat(out, ignore_index=True)


def ic_series(df, f, label="label", min_n=50):
    ics = []
    for _, g in df.groupby("accper_dt"):
        s = g[[f, label]].dropna()
        if len(s) < min_n:
            continue
        v = spear(s[f].values, s[label].values)
        if np.isfinite(v):
            ics.append(v)
    return pd.Series(ics)


def summarize(ics):
    if len(ics) == 0:
        return {"mean": np.nan, "icir": np.nan, "n": 0, "pos": np.nan}
    return {"mean": ics.mean(),
            "icir": ics.mean()/ics.std() if ics.std() > 0 else np.nan,
            "n": len(ics), "pos": (ics > 0).mean()}


def composite_ic(df, factors, signs, weights, label="label", min_n=50):
    ics = []
    for _, g in df.groupby("accper_dt"):
        num = pd.Series(0.0, index=g.index)
        den = pd.Series(0.0, index=g.index)
        for f in factors:
            r = g[f].rank(pct=True)
            ok = r.notna()
            num[ok] += r[ok] * (abs(weights[f]) * signs[f])
            den[ok] += abs(weights[f])
        score = num / den.replace(0, np.nan)
        ok = score.notna() & g[label].notna()
        if ok.sum() < min_n:
            continue
        v = spear(score[ok].values, g.loc[ok, label].values)
        if np.isfinite(v):
            ics.append(v)
    return float(np.mean(ics)) if ics else np.nan, len(ics)


def main():
    log("加载 + 对齐 …")
    df = load()
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,}")

    all_res = {}
    for gname, factors in GROUPS.items():
        log(f"\n=== {gname} ===")
        stats = {}
        for f in factors:
            s = summarize(ic_series(df, f))
            stats[f] = s
            log(f"  {f:22s} IC {s['mean']:+.4f} ICIR {s['icir']:+.3f} n={s['n']}")
        # 实测符号
        signs = {f: (1 if stats[f]["mean"] >= 0 else -1) for f in factors}
        weights = {f: abs(stats[f]["mean"]) for f in factors}

        order = sorted(factors, key=lambda f: abs(stats[f]["mean"]), reverse=True)
        incr, kept, prev = [], [], 0.0
        for i, f in enumerate(order, 1):
            cur, n = composite_ic(df, kept + [f], signs, weights)
            delta = cur - prev
            keep = (i == 1) or (delta >= 0.003)
            incr.append({"step": i, "factor": f, "ic": cur, "delta": delta,
                         "keep": keep, "n": n})
            if keep:
                kept.append(f); prev = cur
            log(f"  {i}. {f:22s} IC {cur:+.4f} Δ{delta:+.4f} "
                f"n={n} {'保留' if keep else '剔除'}")
        all_res[gname] = {"stats": stats, "signs": signs,
                          "incr": incr, "kept": kept}

    log("\n=== 汇总 ===")
    for gname, r in all_res.items():
        log(f"  {gname} → 保留 {r['kept']}")

    # 写回 feature_governance.md 的冻结清单节
    p = ROOT / "feature_governance.md"
    if p.exists():
        t = p.read_text(encoding="utf-8")
        L = []
        for gname, r in all_res.items():
            L.append(f"### {gname}\n")
            L.append("| 特征 | 方向 | 单因子 IC | ICIR | 期间数 | 增量 | 决定 |")
            L.append("|---|---|---|---|---|---|---|")
            for x in r["incr"]:
                s = r["stats"][x["factor"]]
                L.append(f"| `{x['factor']}` | {r['signs'][x['factor']]:+d} | "
                         f"{s['mean']:+.4f} | {s['icir']:+.3f} | {s['n']} | "
                         f"{x['delta']:+.4f} | {'保留' if x['keep'] else '剔除'} |")
            L.append(f"\n**本组保留**：{', '.join(f'`{k}`' for k in r['kept'])}\n")
        block = "\n".join(L)
        # 替换第五节内容
        i = t.find("## 五、冻结的最终特征清单")
        j = t.find("## 六、已知局限")
        if i > 0 and j > i:
            new = ("## 五、冻结的最终特征清单（按样本组分别做增量测试）\n\n"
                   "**为什么分组**：`gm/dgm_q/gm_vol` 覆盖率 94~98%，"
                   "`transfer_intensity/dep_rate` 覆盖率 19.8%（只有制造业披露附注）。\n"
                   "两组覆盖率互斥——混在一套合成里，实际只有 19.8% 的股票四个因子齐全。\n"
                   "**所以必须分组做增量测试，各自得到自己的保留清单。**\n\n"
                   + block + "\n" + t[j:])
            t = t[:i] + new
            p.write_text(t, encoding="utf-8")
            log("  → feature_governance.md 第五节已按分组重写")

    (CLEAN / "meta" / "f7_grouped.json").write_text(
        __import__("json").dumps(
            {g: {"kept": r["kept"],
                 "incr": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                           for k, v in x.items()} for x in r["incr"]],
                 "stats": {f: {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                               for k, v in s.items()} for f, s in r["stats"].items()},
                 "signs": r["signs"]}
             for g, r in all_res.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
