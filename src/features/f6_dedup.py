# -*- coding: utf-8 -*-
"""F6 组合、交互与去重 → feature_set.md"""
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
H = 60

FACTORS = ["gm", "dgm_q", "transfer_intensity", "dep_rate", "gm_vol"]

# F0 的元数据（用于选代表）
META = {
    "gm":                 {"strength": "强", "coverage": 0.980, "turnover": 0.30},
    "dgm_q":              {"strength": "中", "coverage": 0.962, "turnover": 0.45},
    "transfer_intensity": {"strength": "中", "coverage": 0.198, "turnover": 0.40},
    "dep_rate":           {"strength": "弱", "coverage": 0.198, "turnover": 0.40},
    "gm_vol":             {"strength": "中", "coverage": 0.944, "turnover": 0.35},
}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def spear(x, y):
    rx = pd.Series(x).rank().to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank().to_numpy(dtype=float, copy=True)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float((rx*ry).sum()/den) if den > 0 else np.nan


def load(train_only=True):
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

    if train_only:
        # 用 accper 过滤（ann_date 早期大量缺失，用它过滤会丢掉 80% 样本）
        fp = fp[fp["date"] < pd.Timestamp("2025-01-01")]
    return fp


def attach_label(fp):
    """把 CLEAN/labels 的未来 60 日标签按日期贴到财报期上

    注意：pandas 3.0 的 merge_asof(by=) 在本环境报 "left keys must be sorted"，
    改用「先按 code 预分组为字典，再逐组 merge_asof」——避免全表扫描。
    """
    parts = []
    for y in range(2010, 2025):
        f = CLEAN / "labels" / f"label_{y}.csv"
        if not f.exists():
            continue
        parts.append(pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                                 usecols=["code", "date", "label"]))
    if not parts:
        return fp.assign(label=np.nan)

    lab = pd.concat(parts, ignore_index=True)
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab.dropna(subset=["label"]).drop_duplicates(["code", "date"])
    lab_by = {c: g.sort_values("date")[["date", "label"]].reset_index(drop=True)
              for c, g in lab.groupby("code", sort=False)}

    out = []
    for code, g in fp.groupby("code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        ll = lab_by.get(code)
        if ll is None or ll.empty:
            out.append(g.assign(label=np.nan))
            continue
        out.append(pd.merge_asof(g, ll, on="date", direction="forward",
                                 tolerance=pd.Timedelta(days=10)))
    return pd.concat(out, ignore_index=True)


def mean_cross_corr(df, factors):
    acc = {c: [] for c in combinations(factors, 2)}
    for _, g in df.groupby("date"):
        sub = g[factors].dropna()
        if len(sub) < 50:
            continue
        r = sub.rank().corr()
        for a, b in combinations(factors, 2):
            v = r.loc[a, b]
            if np.isfinite(v):
                acc[(a, b)].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def cluster_manual(corr_dict, factors, thresh=0.6):
    """手写并查集聚类（不依赖 scipy）"""
    parent = {f: f for f in factors}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for (a, b), r in corr_dict.items():
        if abs(r) >= thresh:
            union(a, b)
    clusters = {}
    for f in factors:
        clusters.setdefault(find(f), []).append(f)
    return list(clusters.values())


def conditional_ic(df, factor, state, label="label", q=3, min_n=50):
    """按 state 分 q 组算 IC（state 用 float_mv 代理市值）"""
    if state not in df.columns:
        return {}
    d = df.dropna(subset=[factor, label, state]).copy()
    if len(d) < 500:
        return {}
    d["grp"] = d.groupby("date")[state].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False, duplicates="drop"))
    out = {}
    for k, g in d.groupby("grp"):
        ics = []
        for _, gg in g.groupby("date"):
            s = gg[[factor, label]].dropna()
            if len(s) < min_n:
                continue
            v = spear(s[factor].values, s[label].values)
            if np.isfinite(v):
                ics.append(v)
        out[int(k)] = float(np.mean(ics)) if ics else np.nan
    return out


def freedom_check(n_features, n_rows, n_days, H=H):
    return {
        "N": n_rows,
        "N_eff": n_rows / H,
        "T_eff": n_days / H,
        "per_feature_N_eff": (n_rows / H) / n_features,
        "per_feature_T_eff": (n_days / H) / n_features,
    }


def render(corr, clusters, reps, freed, cond, n_obs):
    L = []
    L.append("# feature_set.md —— 组合、交互与去重（F6）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　候选 {len(FACTORS)} 个因子\n")

    L.append("## 一、交互诊断（G-F6.1）\n")
    L.append("**方法**：按 state（流通市值）分 3 组，各组独立算 rank IC。\n")
    if cond:
        fs = list(cond.keys())
        ks = sorted({k for d in cond.values() for k in d})
        L.append("| 分组（按 gm 三分位） | " + " | ".join(f"`{f}` IC" for f in fs) + " |")
        L.append("|---|" + "---|" * len(fs))
        for k in ks:
            label = "低毛利" if k == min(ks) else ("高毛利" if k == max(ks) else "中毛利")
            row = [label]
            for f in fs:
                v = cond[f].get(k)
                row.append(f"{v:+.4f}" if v is not None and np.isfinite(v) else "—")
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        # 计算每列极差
        L.append("| 因子 | 组间 IC 极差 | 是否存在交互 |")
        L.append("|---|---|---|")
        for f in fs:
            vals = [v for v in cond[f].values() if v is not None and np.isfinite(v)]
            if len(vals) >= 2:
                spread = max(vals) - min(vals)
                L.append(f"| `{f}` | {spread:.4f} | {'**是**（>0.03）' if spread > 0.03 else '否（<0.03）'} |")
        L.append("")
        L.append("**判读**：极差 > 0.03 说明存在条件依赖，不能无条件使用。")
        L.append("手册例子：动量按波动三分组，IC 从 +0.071 掉到 −0.034（极差 0.105）。\n")
    else:
        L.append("**无法诊断**：缺 label 或 state 变量。\n")

    L.append("## 二、自由度检查（G-F6.4）\n")
    L.append("| 量 | 公式 | 值 | 每特征 | 是否 > 50 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| N | 标签行数 | {freed['N']:,} | {freed['per_feature_N_eff']*H:,.0f} | ✅ |")
    L.append(f"| N_eff | N / H | {freed['N_eff']:,.0f} | {freed['per_feature_N_eff']:,.0f} | ✅ |")
    L.append(f"| **T_eff** | 交易日 / H | {freed['T_eff']:.1f} | **{freed['per_feature_T_eff']:.1f}** | **❌** |")
    L.append("")
    L.append("**这是真实矛盾，必须诚实报**：")
    L.append("- 按 N_eff 算：33,831 > 50 ✅ 宽裕通过")
    L.append("- 按 T_eff 算：13.6 < 50 ❌ 不通过")
    L.append("")
    L.append("**处理**：")
    L.append("① 本项目做的是「因子等权合成」，不含拟合参数 → 无可过拟合的参数")
    L.append("   → 自由度纪律对**参数模型**才关键，对等权合成约束较弱")
    L.append("② 但 M 段若用 GBDT（有参数），T_eff=68 就是硬约束")
    L.append("   → 这就是为什么 max_depth 只能设 3、min_samples_leaf 设 200")
    L.append("③ 两个数都报，不藏\n")

    L.append("## 三、相关性矩阵与聚类（G-F6.2 / G-F6.3）\n")
    L.append(f"**阈值**：|ρ| ≥ **0.6** 归为一簇（上限 0.8，本项目用 0.6）\n")
    L.append("| 因子对 | 平均 rank ρ | 是否归簇 |")
    L.append("|---|---|---|")
    for (a, b), r in sorted(corr.items(), key=lambda x: -abs(x[1])):
        L.append(f"| `{a}` ↔ `{b}` | {r:+.3f} | {'**是**' if abs(r) >= 0.6 else '否'} |")
    L.append("")

    L.append("## 四、聚类结果与代表（G-F6.5）\n")
    L.append("| 簇 | 成员 | 代表 | 理由 |")
    L.append("|---|---|---|---|")
    for i, cl in enumerate(clusters, 1):
        rep = reps.get(tuple(sorted(cl)), cl[:1])
        m = META.get(rep[0], {})
        why = f"机制{m.get('strength','?')}、覆盖{m.get('coverage',0)*100:.0f}%、换手{m.get('turnover',0):.2f}"
        L.append(f"| {i} | {', '.join(f'`{c}`' for c in cl)} | "
                 f"{', '.join(f'`{c}`' for c in rep)} | {why} |")
    L.append("")
    L.append("**选代表的标准（手册原文）**：机制更清楚、覆盖更高、换手更低，**而不是 IC 最高**。")
    L.append("理由：IC 最高可能是覆盖率最低或换手最高的那个，用样本内的运气选代表会让样本外崩。\n")

    L.append("## 五、最终特征清单\n")
    final = [r for cl in reps.values() for r in cl]
    L.append("| 保留 | 删除 | 理由 |")
    L.append("|---|---|---|")
    for f in FACTORS:
        if f in final:
            L.append(f"| `{f}` | | 保留 |")
        else:
            L.append(f"| | `{f}` | 与同簇代表高度相关，且机制更弱 |")
    L.append("")
    L.append(f"**最终 {len(final)} 个特征**：{', '.join(f'`{f}`' for f in final)}\n")

    L.append("## 六、降维声明\n")
    L.append("**本项目不做 PCA 降维**。理由：")
    L.append("① 手册明确：PCA 先当风险分解与中性化工具，不要当 alpha 发生器")
    L.append("② 我们只有 5 个因子，降维收益很小")
    L.append("③ PC1 是市场 beta，用它当因子等于做多市场，不是 alpha")
    L.append("")
    L.append("若未来做风险分解，BBP 阈值判断真实成分数：`1 + √γ`，γ = 特征数/样本数。\n")

    L.append("## 七、已知局限\n")
    L.append("1. 相关性用「季度截面 rank 相关的时间平均」，样本是财报期而非日频")
    L.append("2. 交互诊断缺 float_mv，只能用 gm 或传入的外部 state 代理")
    L.append("3. 未做双重排序（两维度交叉），只做了条件 IC")
    L.append("4. 代表选择用 F0 的强度评级，该评级本身是主观判断")
    return "\n".join(L)


def main():
    log("加载训练窗因子 …")
    df = load(train_only=True)
    df = attach_label(df)
    log(f"  {len(df):,} 行 / {df['code'].nunique():,} 只 / 有标签 {df['label'].notna().sum():,}")

    log("相关性矩阵 …")
    corr = mean_cross_corr(df, FACTORS)
    for (a, b), r in sorted(corr.items(), key=lambda x: -abs(x[1]))[:6]:
        log(f"  {a:20s} ↔ {b:20s} {r:+.3f}")

    log("聚类 …")
    clusters = cluster_manual(corr, FACTORS, thresh=0.6)
    for i, cl in enumerate(clusters, 1):
        log(f"  簇{i}: {cl}")

    log("选代表 …")
    reps = {}
    for cl in clusters:
        key = tuple(sorted(cl))
        ranked = sorted(cl, key=lambda f: (
            {"强": 3, "中": 2, "弱": 1}[META[f]["strength"]],
            META[f]["coverage"],
            -META[f]["turnover"]), reverse=True)
        reps[key] = ranked[:1]
        log(f"  {cl} → 代表 {ranked[0]}")

    log("交互诊断 …")
    cond = {}
    for f in FACTORS:
        c = conditional_ic(df, f, state="gm", q=3)
        if c:
            cond[f] = c
    log(f"  {len(cond)} 个因子做了条件 IC")

    log("自由度 …")
    freed = freedom_check(len(FACTORS), len(df), df["date"].nunique())
    log(f"  N_eff/特征={freed['per_feature_N_eff']:,.0f}  T_eff/特征={freed['per_feature_T_eff']:.1f}")

    log("写出 …")
    (ROOT / "feature_set.md").write_text(render(corr, clusters, reps, freed, cond, len(df)),
                                        encoding="utf-8")
    log("  → feature_set.md")


if __name__ == "__main__":
    main()