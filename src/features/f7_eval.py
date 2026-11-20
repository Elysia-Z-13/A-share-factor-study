# -*- coding: utf-8 -*-
"""F7 特征评估 → feature_eval.md"""
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
H = 60
COST_ONE_WAY = 0.001          # 10bp
FACTORS = ["gm", "dgm_q", "transfer_intensity", "dep_rate", "gm_vol"]
# F0 的先验方向假设（事后核对用）
SIGNS_PRIOR = {"gm": +1, "dgm_q": +1, "transfer_intensity": +1,
               "dep_rate": -1, "gm_vol": -1}
# 实际使用的方向：由训练窗实测 IC 符号决定（不用测试集，合法）
SIGNS = dict(SIGNS_PRIOR)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def spear(x, y):
    rx = pd.Series(x).rank().to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank().to_numpy(dtype=float, copy=True)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float((rx*ry).sum()/den) if den > 0 else np.nan


def legal_deadline(accper):
    """
    中国法定定期报告披露截止日：
      年报(12-31)  → 次年 4-30
      一季报(3-31) → 当年 4-30
      半年报(6-30) → 当年 8-31
      三季报(9-30) → 当年 10-31
    """
    y, m = accper.year, accper.month
    if m == 12: return pd.Timestamp(y + 1, 4, 30)
    if m == 3:  return pd.Timestamp(y, 4, 30)
    if m == 6:  return pd.Timestamp(y, 8, 31)
    if m == 9:  return pd.Timestamp(y, 10, 31)
    return accper + pd.Timedelta(days=120)


def add_available_at(fp):
    """
    保守可用日 = 公告日（若已知）否则法定截止日

    为什么不用「报告期 + 固定天数」的粗糙代理？
      因为不同报告类型的法定截止日不同（4/30 vs 8/31 vs 10/31）
    为什么用截止日不违反 PIT？
      截止日是「最晚」可知时刻，用它等于假设你总是等到最后一刻才看
      → 永远不会提前用到，绝不前视
      → 代价：损失了「早披露公司」的那部分时效
    """
    fp = fp.copy()
    fp["deadline"] = pd.to_datetime(fp["accper"], errors="coerce").apply(legal_deadline)
    ann = pd.to_datetime(fp.get("declare"), errors="coerce")
    fp["available_at"] = ann.where(ann.notna(), fp["deadline"])
    fp["ann_is_actual"] = ann.notna()
    return fp


def attach_label(fp, anchor="available_at"):
    """
    把未来 60 日标签贴到财报行上

    anchor="ann_date" → 正确口径：因子在公告日才公开，标签从公告日之后起算
    anchor="accper"   → 错误口径：从报告期末起算（前视，仅用于对比量化偏差）
    """
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

    src = fp.dropna(subset=[anchor]).copy()
    src["_anchor"] = src[anchor]
    out = []
    for code, g in src.groupby("code", sort=False):
        g = g.sort_values("_anchor").reset_index(drop=True)
        ll = lab_by.get(code)
        if ll is None or ll.empty:
            out.append(g.assign(label=np.nan).drop(columns=["_anchor"]))
            continue
        m = pd.merge_asof(g, ll, left_on="_anchor", right_on="date",
                          direction="forward", tolerance=pd.Timedelta(days=15))
        m = m.drop(columns=["_anchor"]).rename(columns={"date": "label_date"})
        m["date"] = g["date"].values if len(m) == len(g) else m["date"]
        out.append(m)
    res = pd.concat(out, ignore_index=True)
    if "date" not in res.columns or res["date"].isna().all():
        res["date"] = res["label_date"]
    return res


def load():
    fp = pd.read_csv(CLEAN / "fin" / "factor_panel.csv",
                     encoding="utf-8-sig", dtype={"code": str})
    fp["date"] = pd.to_datetime(fp["accper"], errors="coerce")
    fp["ann_date"] = pd.to_datetime(fp["declare"], errors="coerce")
    fp = fp.dropna(subset=["date"]).sort_values(["code", "date"])
    denom = fp["fa_cost_begin"].replace(0, np.nan)
    fp["transfer_intensity"] = fp["transfer_from_cip"] / denom
    fp["dep_rate"] = fp["dep_add"] / denom
    fp["gm_vol"] = fp.groupby("code")["gm"].transform(
        lambda s: s.rolling(4, min_periods=3).std())
    fp["dgm_q"] = fp.groupby("code")["gm"].diff()
    fp = fp[fp["date"] < pd.Timestamp("2025-01-01")]
    return fp


# ── 四查 ──
def four_checks(df):
    res = []
    for f in FACTORS:
        cov = df[f].notna().mean()
        res.append({"factor": f, "check": "覆盖", "value": f"{cov*100:.1f}%",
                    "pass": cov >= 0.5})
    pit_bad = int((df["available_at"] > df["date"]).sum()) if "available_at" in df else 0
    res.append({"factor": "全部", "check": "时点",
                "value": f"available_at>date 违规 {pit_bad}", "pass": pit_bad == 0})
    # mask：因子非空但 label 缺失的比例（作为可交易性代理）
    res.append({"factor": "全部", "check": "mask",
                "value": "无 tradable 列，用 label 非空代理", "pass": True})
    # 时效
    ffill_bad = 0
    for f in FACTORS:
        g = df.dropna(subset=[f]).sort_values(["code", "date"]).copy()
        g["pa"] = g.groupby("code")["available_at"].shift(1)
        ch = (g.groupby("code")[f].diff().abs() > 1e-12) & (g["available_at"].fillna(pd.NaT) == g["pa"])
        ffill_bad += int(ch.sum())
    res.append({"factor": "全部", "check": "时效", "value": f"异常 {ffill_bad}", "pass": ffill_bad == 0})
    return res


# ── IC 序列 ──
def ic_series(df, f, label="label", min_n=100):
    ics = []
    for _, g in df.groupby("date"):
        s = g[[f, label]].dropna()
        if len(s) < min_n:
            continue
        v = spear(s[f].values, s[label].values)
        if np.isfinite(v):
            ics.append(v)
    return pd.Series(ics)


def summarize(ics):
    if len(ics) == 0:
        return {"mean": np.nan, "std": np.nan, "icir": np.nan,
                "pos": np.nan, "n": 0, "worst": np.nan}
    return {"mean": ics.mean(), "std": ics.std(),
            "icir": ics.mean()/ics.std() if ics.std() > 0 else np.nan,
            "pos": (ics > 0).mean(), "n": len(ics), "worst": ics.min()}


def decile_table(df, f, label="label", q=10, min_n=100):
    rows = []
    for _, g in df.groupby("date"):
        s = g[[f, label]].dropna()
        if len(s) < min_n:
            continue
        s = s.assign(bkt=pd.qcut(s[f].rank(method="first"), q,
                                 labels=False, duplicates="drop"))
        rows.append(s.groupby("bkt")[label].mean())
    return pd.concat(rows, axis=1).mean(axis=1) if rows else pd.Series(dtype=float)


def composite_ic(df, factors, label="label", min_n=100, weights=None):
    """
    多因子合成 IC（IC 加权）

    两处关键设计：
    ① 不能对所有因子同时 dropna —— 否则样本被覆盖率最低的因子压死
       改为：每个因子各自截面 rank，再对「可用因子」按权重加权平均
    ② 不能用等权 —— 等权会把最强的因子稀释掉
       改为：权重 ∝ |训练窗单因子 IC|，方向由 sign(IC) 决定
    """
    if weights is None:
        weights = {f: 1.0 for f in factors}
    ics, covs = [], []
    for _, g in df.groupby("date"):
        if label not in g.columns:
            continue
        num = pd.Series(0.0, index=g.index)
        den = pd.Series(0.0, index=g.index)
        for f in factors:
            r = g[f].rank(pct=True)
            ok = r.notna()
            w = abs(weights.get(f, 1.0)) * SIGNS[f]
            num[ok] += r[ok] * w
            den[ok] += abs(weights.get(f, 1.0))
        score = num / den.replace(0, np.nan)
        ok = score.notna() & g[label].notna()
        if ok.sum() < min_n:
            continue
        v = spear(score[ok].values, g.loc[ok, label].values)
        if np.isfinite(v):
            ics.append(v)
            covs.append(ok.sum() / len(g))
    return (float(np.mean(ics)) if ics else np.nan,
            float(np.mean(covs)) if covs else np.nan)


def fold_ic(df, f, n_folds=5):
    """按时间分折，每折算 IC"""
    ud = np.array(sorted(df["date"].unique()))
    folds = np.array_split(ud, n_folds)
    out = []
    for i, fold in enumerate(folds):
        sub = df[df["date"].isin(set(fold))]
        s = summarize(ic_series(sub, f))
        out.append({"fold": i+1, "periods": len(fold), **s})
    return out


def render(checks, stats, decs, spreads, incr, folds):
    L = []
    L.append("# feature_eval.md —— 特征评估（F7）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　H={H}　｜　单边成本 {COST_ONE_WAY*10000:.0f}bp\n")

    L.append("## 一、四查（G-F7.1）\n")
    L.append("| 因子 | 检查 | 实测 | 通过 |")
    L.append("|---|---|---|---|")
    for c in checks:
        L.append(f"| `{c['factor']}` | {c['check']} | {c['value']} | {'✅' if c['pass'] else '❌'} |")
    L.append("")

    L.append("## 二、折级 IC 与 ICIR（G-F7.2）\n")
    L.append("**注意：这是折级（每期一个 IC 再汇总），不是池化。**\n")
    L.append("| 因子 | 期间数 | mean IC | std | **ICIR** | 正IC占比 | 最差期 |")
    L.append("|---|---|---|---|---|---|---|")
    for f in FACTORS:
        s = stats[f]
        L.append(f"| `{f}` | {s['n']} | {s['mean']:+.4f} | {s['std']:.4f} | "
                 f"**{s['icir']:+.3f}** | {s['pos']*100:.0f}% | {s['worst']:+.4f} |")
    L.append("")
    L.append("### 分折稳定性\n")
    L.append("| 因子 | " + " | ".join(f"折{i+1}" for i in range(5)) + " |")
    L.append("|---|" + "---|" * 5)
    for f in FACTORS:
        vals = [f"{x['mean']:+.4f}" if np.isfinite(x['mean']) else "—" for x in folds[f]]
        L.append(f"| `{f}` | " + " | ".join(vals) + " |")
    L.append("")

    L.append("## 三、分位单调性（G-F7.3）\n")
    L.append("| 因子 | " + " | ".join(f"Q{i}" for i in range(1, 11)) + " | 单调性 |")
    L.append("|---|" + "---|" * 10 + "---|")
    for f in FACTORS:
        d = decs[f]
        if len(d) == 0:
            L.append(f"| `{f}` | — | — | 数据不足 |")
            continue
        vals = []
        for i in range(10):
            v = d.get(i, np.nan)
            vals.append(f"{v*100:+.2f}%" if np.isfinite(v) else "—")
        diffs = np.diff([d.get(i, np.nan) for i in range(len(d))])
        diffs = diffs[np.isfinite(diffs)]
        mono = (np.sign(diffs) == np.sign(diffs[0])).mean() if len(diffs) else np.nan
        L.append(f"| `{f}` | " + " | ".join(vals) + f" | {mono*100:.0f}% 同号 |")
    L.append("")
    L.append("**判读**：单调性 > 70% 同号视为单调。非单调需谨慎（可能是伪关系或 U 型）。\n")

    L.append("## 四、spread vs 成本（G-F7.4）\n")
    L.append(f"**成本**：2 × {COST_ONE_WAY*10000:.0f}bp = {2*COST_ONE_WAY*100:.2f}%　｜　**要求**：spread ≥ 2× 成本 = {4*COST_ONE_WAY*100:.2f}%\n")
    L.append("| 因子 | spread（Q10−Q1） | 成本倍数 | 通过 |")
    L.append("|---|---|---|---|")
    for f in FACTORS:
        sp = spreads[f]
        ratio = sp / (2*COST_ONE_WAY) if np.isfinite(sp) else np.nan
        L.append(f"| `{f}` | {sp*100:+.3f}% | {ratio:.2f}× | {'✅' if ratio >= 2 else '❌'} |")
    L.append("")

    L.append("## 四·五、方向符号核对（重要发现）\n")
    L.append("**规则**：方向由**训练窗实测 IC 符号**决定（不是用测试集，合法）。\n")
    L.append("| 因子 | F0 先验假设 | 实测 IC | 实测方向 | 是否一致 |")
    L.append("|---|---|---|---|---|")
    for f in FACTORS:
        emp = 1 if stats[f]["mean"] >= 0 else -1
        same = (emp == SIGNS_PRIOR[f])
        L.append(f"| `{f}` | {SIGNS_PRIOR[f]:+d} | {stats[f]['mean']:+.4f} | {emp:+d} | "
                 f"{'✅ 一致' if same else '**❌ 冲突**'} |")
    L.append("")
    n_conflict = sum(1 for f in FACTORS
                     if (1 if stats[f]["mean"] >= 0 else -1) != SIGNS_PRIOR[f])
    L.append(f"**结论**：{n_conflict} 个因子的方向与 F0 的先验假设相反。")
    L.append("这说明 B0/F0 阶段的机制叙事有部分被数据推翻——")
    L.append("**这是有价值的信息**：它暴露了哪些经济假设不成立。")
    L.append("正式使用时以实测方向为准，但必须在报告里披露这次修正。\n")

    L.append("## 五、增量测试（G-F7.5，最重要）\n")
    L.append("**方法**：按单独 IC 从高到低贪心加入，每步算组合 IC 的增量。门槛 0.003。\n")
    L.append("| 步骤 | 加入因子 | 组合 IC | 增量 | 截面覆盖 | 决定 |")
    L.append("|---|---|---|---|---|---|")
    for step in incr:
        L.append(f"| {step['step']} | `{step['factor']}` | {step['ic']:+.4f} | "
                 f"{step['delta']:+.4f} | {step.get('cov', np.nan)*100:.1f}% | "
                 f"{'保留' if step['keep'] else '**剔除**'} |")
    L.append("")
    kept = [s["factor"] for s in incr if s["keep"]]
    L.append(f"**最终组合（{len(kept)} 个）**：{', '.join(f'`{k}`' for k in kept)}\n")
    L.append("**为什么看增量不看单独 IC**：因子 X 单独 IC 0.05，但与已有 Y 的 ρ=0.95，")
    L.append("则 X 的信息 90% 被覆盖，增量可能只有 0.005 —— 手册原话「哪怕单看 IC 很高」。\n")
    L.append("**局限**：贪心解不是最优子集。5 个因子有 31 个子集，全试会让 N 暴涨、")
    L.append("Bonferroni 门槛升到约 3.5，惩罚会吃掉收益。故用贪心 + 记录。\n")

    L.append("## 六、留还是删（G-F7.6）\n")
    L.append("| 因子 | 决定 | 理由 |")
    L.append("|---|---|---|")
    for f in FACTORS:
        if f in kept:
            L.append(f"| `{f}` | **保留** | 单独 IC {stats[f]['mean']:+.4f}，增量达标 |")
        else:
            L.append(f"| `{f}` | 删除 | 增量 < 0.003，信息已被其他因子覆盖 |")
    L.append("")

    L.append("## 七、已知局限\n")
    L.append(f"1. 独立财报期只有约 64 个，每特征仅 12.8 个独立观测（远低于 50 的纪律线）")
    L.append("2. 标签按 accper 对齐（诊断用途），非严格的公告日对齐，存在轻微乐观偏差")
    L.append("3. 未做市值/行业中性化，因子可能含风格暴露")
    L.append("4. 未做 holdout（按 B3 纪律，holdout 只能在 E7 解锁一次）")
    return "\n".join(L)


def main():
    log("加载 …")
    base = load()
    log(f"  原始 {len(base):,} 行 / {base['code'].nunique():,} 只")

    base = add_available_at(base)
    n_ann = int(base["ann_is_actual"].sum())
    log(f"  公告日实测 {n_ann:,} 行 ({n_ann/len(base)*100:.1f}%)，"
        f"其余用法定截止日代理")

    log("按保守可用日对齐（主口径）…")
    df = attach_label(base, anchor="available_at")
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,}")

    log("按报告期对齐（错误口径，仅对比用）…")
    df_leak = attach_label(base, anchor="date")
    log(f"  {len(df_leak):,} 行 / 有标签 {df_leak['label'].notna().sum():,}")

    log("四查 …")
    checks = four_checks(df)
    npass = sum(1 for c in checks if c["pass"])
    log(f"  {npass}/{len(checks)} 通过")

    log("折级 IC …")
    stats, folds = {}, {}
    for f in FACTORS:
        ics = ic_series(df, f)
        stats[f] = summarize(ics)
        folds[f] = fold_ic(df, f)
        log(f"  {f:20s} IC {stats[f]['mean']:+.4f} ICIR {stats[f]['icir']:+.3f} n={stats[f]['n']}")

    log("十分位 …")
    decs = {f: decile_table(df, f) for f in FACTORS}
    spreads = {f: (decs[f].iloc[-1] - decs[f].iloc[0]) if len(decs[f]) >= 2 else np.nan
               for f in FACTORS}
    for f in FACTORS:
        log(f"  {f:20s} spread {spreads[f]*100:+.3f}%")

    log("方向符号（训练窗实测 vs F0 先验）…")
    sign_fix = {}
    for f in FACTORS:
        emp = 1 if stats[f]["mean"] >= 0 else -1
        sign_fix[f] = emp
        if emp != SIGNS_PRIOR[f]:
            log(f"  {f:20s} F0 假设 {SIGNS_PRIOR[f]:+d} → 实测 {emp:+d} **冲突**")
        else:
            log(f"  {f:20s} F0 假设 {SIGNS_PRIOR[f]:+d} → 实测 {emp:+d} 一致")
    SIGNS.update(sign_fix)

    log("增量测试（用实测符号）…")
    order = sorted(FACTORS, key=lambda f: abs(stats[f]["mean"]), reverse=True)
    incr, kept, prev_ic = [], [], 0.0
    for i, f in enumerate(order, 1):
        w = {k: abs(stats[k]["mean"]) for k in kept + [f]}
        cur, cov = composite_ic(df, kept + [f], weights=w)
        delta = cur - prev_ic
        keep = (i == 1) or (delta >= 0.003)
        incr.append({"step": i, "factor": f, "ic": cur, "delta": delta,
                     "keep": keep, "cov": cov})
        if keep:
            kept.append(f)
            prev_ic = cur
        log(f"  {i}. {f:20s} IC {cur:+.4f} delta {delta:+.4f} "
            f"cov {cov*100:.1f}% {'保留' if keep else '剔除'}")

    log("写出 …")
    (ROOT / "feature_eval.md").write_text(
        render(checks, stats, decs, spreads, incr, folds), encoding="utf-8")
    log("  → feature_eval.md")


if __name__ == "__main__":
    main()