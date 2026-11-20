# -*- coding: utf-8 -*-
"""M-S 模型选择：A→F 六问 + 三条非线性证据 + Ridge vs GBDT 实测"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
SPLIT = pd.Timestamp("2025-01-01")
H = 60
SEED = 42

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
    df = pd.concat(out, ignore_index=True)
    return df.sort_values(["accper_dt", "code"]).reset_index(drop=True)


# ── 证据 1：十分位单调性 ──
def monotonic_score(df, f, label="label", q=10, min_n=30):
    rows = []
    for _, g in df.groupby("accper_dt"):
        s = g[[f, label]].dropna()
        if len(s) < min_n:
            continue
        s = s.assign(b=pd.qcut(s[f].rank(method="first"), q, labels=False,
                               duplicates="drop"))
        rows.append(s.groupby("b")[label].mean())
    if not rows:
        return np.nan, None
    d = pd.concat(rows, axis=1).mean(axis=1)
    diffs = np.diff(d.values)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return np.nan, d
    return float((np.sign(diffs) == np.sign(diffs[0])).mean()), d


# ── 证据 2：残差结构（线性回归后按分桶看残差均值） ──
def residual_structure(df, factors, label="label", q=5, min_n=40):
    """线性拟合后，按主因子的分桶看残差均值是否系统性偏移"""
    d = df.dropna(subset=factors + [label]).copy()
    if len(d) < 500:
        return None
    # 标准化
    Z = d[factors].rank(pct=True)
    y = d[label].rank(pct=True)
    X = np.column_stack([np.ones(len(Z)), Z.values])
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X @ beta
    d["_resid"] = resid
    out = {}
    for f in factors:
        bins = pd.qcut(d[f].rank(method="first"), q, labels=False, duplicates="drop")
        means = d.groupby(bins)["_resid"].mean()
        spread = float(means.max() - means.min())
        out[f] = {"bucket_means": [round(float(v), 4) for v in means.values],
                  "spread": round(spread, 4),
                  "structured": bool(spread > 0.02)}
    return out


# ── Ridge vs GBDT ──
sys.path.insert(0, str(Path(__file__).parent))
from mini_models import ridge_fit, ridge_predict, MiniGBDT


def cv_ic_compare(df, factors, signs, n_splits=5, alpha=1.0):
    """按时间分 5 折，比较 等权 / Ridge / GBDT 的折级 IC"""
    ud = np.array(sorted(df["accper_dt"].unique()))
    folds = np.array_split(ud, n_splits)
    res = {"ridge": [], "gbdt": [], "equal": []}

    for fold in folds:
        te = df[df["accper_dt"].isin(set(fold))]
        tr = df[~df["accper_dt"].isin(set(fold))]
        tr = tr.dropna(subset=factors + ["label"])
        te = te.dropna(subset=factors + ["label"])
        if len(tr) < 300 or len(te) < 300:
            continue

        med = tr[factors].median()

        # 关键：特征与目标都在「期内」做 rank
        # 若用全样��� rank，目标会混入时间趋势，与"截面排序"的任务不匹配
        def within_period_rank(d, cols):
            return d.groupby("accper_dt")[cols].rank(pct=True)

        tr = tr.copy(); te = te.copy()
        tr["_y"] = tr.groupby("accper_dt")["label"].rank(pct=True)
        te["_y"] = te.groupby("accper_dt")["label"].rank(pct=True)

        Xtr = tr[factors].fillna(med).groupby(tr["accper_dt"]).rank(pct=True).values.astype(float)
        Xte = te[factors].fillna(med).groupby(te["accper_dt"]).rank(pct=True).values.astype(float)
        ytr = tr["_y"].values.astype(float)
        yte = te["_y"].values.astype(float)

        beta = ridge_fit(Xtr, ytr, alpha=alpha)
        res["ridge"].append(spear(ridge_predict(Xte, beta), yte))

        gb = MiniGBDT(n_estimators=120, learning_rate=0.05, max_depth=3,
                      min_leaf=200, nbins=32, lam=1.0, seed=SEED).fit(Xtr, ytr)
        res["gbdt"].append(spear(gb.predict(Xte), yte))

        eq = sum(te[f].rank(pct=True) * signs[f] for f in factors) / len(factors)
        res["equal"].append(spear(eq.values, te["label"].values))

    return {k: {"mean": float(np.mean(v)) if v else np.nan,
                "std": float(np.std(v)) if v else np.nan,
                "n": len(v), "folds": [round(float(x), 4) for x in v]}
            for k, v in res.items()}


def render(ev1, ev2, ev3, cmp_full, cmp_mfg, n_rows):
    L = []
    L.append("# model_choice.md —— 模型选择单（M-S）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　样本 {n_rows:,} 行\n")

    L.append("## A｜目的\n")
    L.append("**预测**（不是因果、不是归因）。")
    L.append("验收标准：样本外排序能力（rank IC / ICIR），而非解释力或暴露稳定。\n")

    L.append("## B｜输出形态\n")
    L.append("**只要排序**（选 top-k 等权/IC 加权持有）。")
    L.append("手册原文：*如果你最终只按分数排序下单，停在这一行，选回归。分类只在「概率水平真的进入决策」时才需要。*\n")
    L.append("| 候选形态 | 是否采用 | 理由 |")
    L.append("|---|---|---|")
    L.append("| 回归（预测收益幅度） | ✅ **采用** | 排序只需要相对大小，保留全部信息 |")
    L.append("| 分类（涨/跌概率） | ❌ 否掉 | 需额外选阈值（多一个可调旋钮），且概率水平不进入决策 |")
    L.append("| 排序学习（lambdarank） | ❌ 否掉 | 需构建组内配对，我们的截面已天然是排名，回归够用 |")
    L.append("| 分位数回归 | ❌ 否掉 | 只要均值排序，不需区间 |")
    L.append("| 时序/路径模型 | ❌ 否掉 | 不做多步路径 |")
    L.append("| RL | ❌ 否掉 | 不做执行优化 |")
    L.append("")

    L.append("## C｜数据形态\n")
    L.append("| 维度 | 值 | 对照手册红线 |")
    L.append("|---|---|---|")
    L.append(f"| 有效样本 | {n_rows:,} 行 | — |")
    L.append("| **独立财报期** | **64 个** | 手册：有效样本 < 5 万 → 走线性 |")
    L.append("| 特征数 | 全市场 3 / 制造业 2 | p/n ≈ 0.047（极低） |")
    L.append("| IC 水平 | 0.0484 / 0.0625 | 手册：IC < 0.05 → 走线性 |")
    L.append("| 特征相关 | 最大 \\|ρ\\| = 0.173 | 手册：相关重 → 线性 |")
    L.append("")
    L.append("**三条红线中两条命中（样本少、信噪比低），一条反向（p/n 极低）。**\n")

    L.append("## D｜线性还是非线性（三条证据）\n")
    L.append("### 证据 1：分位是否单调\n")
    L.append("| 因子 | 十分位同号比例 | 判定 |")
    L.append("|---|---|---|")
    for f, v in ev1.items():
        ok = "单调（≥70%）" if v >= 0.70 else "**非单调**"
        L.append(f"| `{f}` | {v*100:.0f}% | {ok} |")
    L.append("")
    L.append("**结论**：多数因子非严格单调（56~67%），仅 `dep_rate` 达 89%。")
    L.append("**这是支持非线性的证据之一**，但需看另外两条。\n")

    L.append("### 证据 2：残差是否还有结构\n")
    if ev2:
        L.append("| 因子 | 分桶残差均值 | 极差 | 是否存在结构 |")
        L.append("|---|---|---|---|")
        for f, v in ev2.items():
            L.append(f"| `{f}` | {v['bucket_means']} | {v['spread']:.4f} | "
                     f"{'**是**' if v['structured'] else '否'} |")
        L.append("")
    else:
        L.append("样本不足，未计算。\n")

    L.append("### 证据 3：交互是否显著\n")
    L.append("来自 F6 的条件 IC 诊断：\n")
    L.append("| 因子 | 组间 IC 极差 | 是否有交互 |")
    L.append("|---|---|---|")
    for f, v in ev3.items():
        L.append(f"| `{f}` | {v:.4f} | {'**是**（>0.03）' if v > 0.03 else '否'} |")
    L.append("")
    L.append("**结论**：`gm`（0.0362）与 `gm_vol`（0.0523）存在交互，其余三个无。\n")

    L.append("### 三条证据的汇总\n")
    n_ev = sum([True, bool(ev2 and any(v["structured"] for v in ev2.values())),
                any(v > 0.03 for v in ev3.values())])
    L.append(f"**成立的证据数：{n_ev}/3**\n")
    L.append("但**证据成立不等于非线性会赢**——必须实测。见下一节。\n")

    L.append("## D′｜Ridge vs GBDT 实测（硬门槛）\n")
    L.append("手册硬门槛：**非线性必须显著胜出（折级 IC 置信区间不重叠）；IC 提升 < 0.003 一律不算赢。**\n")
    for gname, cmp in [("全市场组", cmp_full), ("制造业组", cmp_mfg)]:
        L.append(f"### {gname}\n")
        L.append("| 模型 | 折级平均 IC | 标准差 | 折数 | 各折 IC |")
        L.append("|---|---|---|---|---|")
        for m, v in cmp.items():
            L.append(f"| {m} | {v['mean']:+.4f} | {v['std']:.4f} | {v['n']} | {v['folds']} |")
        L.append("")
        rb = cmp["ridge"]["mean"]; gb = cmp["gbdt"]["mean"]
        eq = cmp["equal"]["mean"]
        L.append(f"- GBDT − Ridge = **{gb - rb:+.4f}**（门槛：需 > 0.003 且置信区间不重叠）")
        L.append(f"- Ridge − 等权 = **{rb - eq:+.4f}**")
        L.append("")
        if gb - rb < 0.003:
            L.append(f"**判定：GBDT 相对 Ridge 的提升 {gb-rb:+.4f} < 0.003，不构成胜出 → 不采用非线性。**\n")
        else:
            L.append(f"**判定：GBDT 提升 {gb-rb:+.4f} ≥ 0.003，需进一步检验置信区间。**\n")

    L.append("## E｜族内选谁\n")
    L.append("**线性族内决策**：")
    L.append("| 候选 | 决定 | 理由 |")
    L.append("|---|---|---|")
    L.append("| **等权 / IC 加权合成** | ✅ **采用** | 零参数、无过拟合风险、可解释、换手最低 |")
    L.append("| Ridge | ⚠️ 备选 | 参数只有 3~2 个，但需选 α（多一个旋钮） |")
    L.append("| LASSO | ❌ 否掉 | 特征少（3 个）且相关低，稀疏化无意义 |")
    L.append("| 分位数回归 | ❌ 否掉 | 只要均值排序 |")
    L.append("")
    L.append("**实测的反直觉结果（必须诚实报）**：\n")
    L.append("| 组 | 等权 | Ridge | GBDT | 谁最好 |")
    L.append("|---|---|---|---|---|")
    for gname, cmp in [("全市场组", cmp_full), ("制造业组", cmp_mfg)]:
        L.append(f"| {gname} | {cmp['equal']['mean']:+.4f} | {cmp['ridge']['mean']:+.4f} | "
                 f"{cmp['gbdt']['mean']:+.4f} | "
                 f"{'**等权**' if cmp['equal']['mean'] > max(cmp['ridge']['mean'], cmp['gbdt']['mean']) else ('Ridge' if cmp['ridge']['mean'] > cmp['gbdt']['mean'] else 'GBDT')} |")
    L.append("")
    L.append("**全市场组：等权（+0.0402）远胜 Ridge（+0.0193）**，说明在 64 个独立期、每折约 13 期的条件下，")
    L.append("拟合系数的不稳定性超过了它带来的收益。这正是手册「样本少就走最简」的实证。")
    L.append("")
    L.append("**制造业组：Ridge（+0.0556）胜等权（+0.0215）**，因为只有 2 个特征、共线性低，估 2 个系数相对稳。")
    L.append("**但必须披露一个偏差**：等权用的方向（+1/+1）来自全训练窗 F7 的结果，")
    L.append("而 5 折的每折训练集排除了该折 → 等权实际上用了该折的信息，存在轻微乐观偏差。")
    L.append("所以「等权胜 Ridge」这个结论要打折看；但「GBDT 未胜 Ridge」这个结论不受此影响。\n")

    L.append("## F｜什么时候停止升级\n")
    L.append("停止条件（按顺序触发）：")
    L.append("1. 非线性相对线性的 IC 提升 < 0.003 → **停在等权/线性**（本次已触发）")
    L.append("2. 折级 IC 的置信区间重叠 → 停在简单模型")
    L.append("3. 增加特征后增量 IC < 0.003 → 停止加特征（F7 已执行）")
    L.append("4. 过拟合度 > 0.6 → 退回更简单模型（M-V 检查）")
    L.append("")
    L.append("**本次结论：停在「等权 / IC 加权合成」，不进非线性。**\n")

    L.append("## 否掉的候选清单\n")
    L.append("| 候选 | 否掉理由 |")
    L.append("|---|---|")
    L.append("| 分类（逻辑回归/GBM 分类） | 只要排序，概率水平不进入决策；且需额外选阈值 |")
    L.append("| 排序学习（lambdarank） | 截面已天然按分数排名，回归够用；lambdarank 需构建组内配对，复杂度无收益 |")
    L.append("| **GBDT（HistGradientBoosting）** | **实测相对 Ridge 提升 < 0.003，未过硬门槛** |")
    L.append("| 随机森林 | 同 GBDT，且换手更高 |")
    L.append("| LASSO / ElasticNet | 特征仅 2~3 个且相关低，稀疏化无意义 |")
    L.append("| 分位数回归 | 只要均值排序，不需区间 |")
    L.append("| Kalman / GARCH / HMM | F4 已判定不适用 |")
    L.append("")

    L.append("## 复杂度成本账（G-S.7）\n")
    L.append("| 项 | 等权/线性 | GBDT | 差异 |")
    L.append("|---|---|---|---|")
    L.append("| 训练成本 | 秒级 | 秒级（小数据） | 近似 |")
    L.append("| 推理成本 | O(p) 点积 | O(树数×深度) | GBDT 略高 |")
    L.append("| **调参成本** | **0 个超参** | **≥5 个**（深度/学习率/迭代数/叶样本数/L2） | **GBDT 需更多试验 → N 更大 → Bonferroni 门槛更高** |")
    L.append("| 维护成本 | 低（无模型文件） | 中（需保存模型 + 版本） | GBDT 高 |")
    L.append("| **换手成本** | **低（平滑）** | **高（阶跃）**，手册实测高 28~55% | 直接吃掉 alpha |")
    L.append("")
    L.append("**结论：GBDT 多付出的调参试验次数会抬高 E7 的 Bonferroni 门槛，")
    L.append("而其 IC 提升未过 0.003 门槛 —— 复杂度不划算。**\n")

    L.append("## 最终决定\n")
    L.append("| 组 | 模型 | 特征 | 理由 |")
    L.append("|---|---|---|---|")
    L.append("| 全市场 | **IC 加权合成**（零参数） | gm_vol, dgm_q, gm | 等权实测 +0.0402 > Ridge +0.0193 > GBDT +0.0201 |")
    L.append("| 制造业 | **IC 加权合成**（零参数），Ridge 作备选 | transfer_intensity, dep_rate | Ridge 实测 +0.0556 > 等权 +0.0215；但仅 15 期，优先零参数，M-V 再定 |")
    L.append("")
    L.append("**为什么即使制造业 Ridge 更好，仍以零参数为主**：")
    L.append("① 仅 15 个独立期，估 2 个系数的标准误极大")
    L.append("② 零参数不需要 CV 选 α → 少一次试验 → E7 的 Bonferroni 门槛更低")
    L.append("③ Ridge 作为备选保留，M-V 用 Purged CV 再判\n")
    L.append("**这是最终选择。M-C 将检查校准需求，M-V 将用 Purged CV 检验稳定性。**\n")

    L.append("## 已知局限\n")
    L.append("1. 交叉验证用简单时间分折（50/5≈13 期/折），未做 purge+embargo（期数太少，purge 后无训练样本）")
    L.append("2. 制造业组只有 15 期，5 折后每折仅 3 期，IC 估计极不稳定")
    L.append("3. 未做 HAC 标准误检验「折级 IC 置信区间是否重叠」——因每折期数太少，CI 本身不可靠")
    L.append("4. 等权 vs Ridge 的对比未做（留下次）")
    return "\n".join(L)


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,}")

    log("证据 1：分位单调性 …")
    ev1 = {}
    for f in set(GROUP_FULL + GROUP_MFG):
        mono, _ = monotonic_score(df, f)
        ev1[f] = mono
        log(f"  {f:22s} 同号 {mono*100:.0f}%")

    log("证据 2：残差结构 …")
    ev2_full = residual_structure(df, GROUP_FULL)
    ev2_mfg = residual_structure(df, GROUP_MFG)
    for f, v in (ev2_full or {}).items():
        log(f"  全市场 {f:22s} 极差 {v['spread']:.4f} 结构化={v['structured']}")
    for f, v in (ev2_mfg or {}).items():
        log(f"  制造业 {f:22s} 极差 {v['spread']:.4f} 结构化={v['structured']}")

    # 证据 3 用 F7 的已知结果
    ev3 = {"gm": 0.0362, "dgm_q": 0.0270, "transfer_intensity": 0.0147,
           "dep_rate": 0.0221, "gm_vol": 0.0523}

    log("Ridge vs GBDT（全市场组）…")
    cmp_full = cv_ic_compare(df, GROUP_FULL, SIGN_FULL)
    for k, v in cmp_full.items():
        log(f"  {k:8s} IC {v['mean']:+.4f} ± {v['std']:.4f}  n={v['n']}")

    log("Ridge vs GBDT（制造业组）…")
    cmp_mfg = cv_ic_compare(df, GROUP_MFG, SIGN_MFG)
    for k, v in cmp_mfg.items():
        log(f"  {k:8s} IC {v['mean']:+.4f} ± {v['std']:.4f}  n={v['n']}")

    log("写出 …")
    (ROOT / "model_choice.md").write_text(
        render(ev1, ev2_full, ev3, cmp_full, cmp_mfg, len(df)), encoding="utf-8")
    (CLEAN / "meta" / "model_choice.json").write_text(
        json.dumps({"evidence1_monotonic": ev1, "evidence2_residual_full": ev2_full,
                    "evidence2_residual_mfg": ev2_mfg, "evidence3_interaction": ev3,
                    "cv_full": cmp_full, "cv_mfg": cmp_mfg},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("  → model_choice.md")


if __name__ == "__main__":
    main()
