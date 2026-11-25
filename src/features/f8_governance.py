# -*- coding: utf-8 -*-
"""F8 特征治理 → feature_governance.md"""
import os
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
SPLIT = pd.Timestamp("2025-01-01")

FACTORS = ["gm", "dgm_q", "transfer_intensity", "dep_rate", "gm_vol"]

VERSIONS = [
    {"name": "gm", "version": "v1.0", "effective": "2026-11-20", "status": "active",
     "defn": "(营业收入−营业成本)/营业收入", "inputs": "FS_Comins.B001101000 / B001201000",
     "window": "最近一期", "ref": "横截面", "rep": "rank",
     "note": "首次冻结"},
    {"name": "dgm_q", "version": "v1.0", "effective": "2026-11-20", "status": "active",
     "defn": "gm_t − gm_{t−1}", "inputs": "gm 相邻两期", "window": "最近两期",
     "ref": "横截面", "rep": "rank", "note": "首次冻结"},
    {"name": "transfer_intensity", "version": "v1.0", "effective": "2026-11-20",
     "status": "active", "defn": "ΣFn02007(类型1)/ΣFn02003(类型1)",
     "inputs": "FN_Fn020", "window": "最近一期", "ref": "横截面", "rep": "rank",
     "note": "分母已从 cip_prev 改为 fa_cost_begin（改含义 v0.9→v1.0）"},
    {"name": "dep_rate", "version": "v1.1", "effective": "2026-11-20", "status": "active",
     "defn": "ΣFn02004(类型2)/ΣFn02003(类型1)", "inputs": "FN_Fn020",
     "window": "最近一期", "ref": "横截面", "rep": "rank",
     "note": "符号由 −1 修正为 +1（F7 实测推翻 F0 假设，v1.0→v1.1）"},
    {"name": "gm_vol", "version": "v1.0", "effective": "2026-11-20", "status": "retired",
     "defn": "std(gm 最近4期)", "inputs": "gm 最近 4 期", "window": "4 期滚动",
     "ref": "横截面", "rep": "rank",
     "note": "F7 增量 ΔIC=−0.0253，淘汰"},
]

CHANGELOG = [
    {"date": "2026-11-19", "feature": "transfer_intensity",
     "change": "分母 cip_prev → fa_cost_begin", "type": "改含义",
     "version": "v0.9 → v1.0",
     "reason": "茅台转固率用在建工程作分母得 254%（>100% 无意义）；改用固定资产原值后 11.06%"},
    {"date": "2026-11-20", "feature": "dep_rate", "change": "方向 −1 → +1",
     "type": "符号修正", "version": "v1.0 → v1.1",
     "reason": "F7 实测 IC=+0.0346，与 F0 假设（折旧率高→未来收益低）相反；可能机制是折旧率反映产能利用率"},
    {"date": "2026-11-20", "feature": "gm_vol", "change": "淘汰", "type": "退役",
     "version": "v1.0 → retired",
     "reason": "F7 增量测试：加入后组合 IC 0.0626→0.0372，Δ=−0.0253"},
    {"date": "2026-11-20", "feature": "全部", "change": "可用日口径",
     "type": "改含义（数据层）", "version": "—",
     "reason": "CSMAR DeclareDate 仅 5-8% 覆盖；未知时用法定披露截止日代理（年报4/30、半年报8/31、三季报10/31），永不前视"},
]

RETIREMENT_RULES = [
    {"id": "R1", "name": "连续两轮 IC 不显著",
     "trigger": "连续 2 次评估 |t| < Bonferroni 门槛", "action": "标记候选退役，进入下一轮观察"},
    {"id": "R2", "name": "重要度持续近零",
     "trigger": "SHAP 跨折均值 < 0.01 且标准差 < 0.005", "action": "删除并记入 changelog"},
    {"id": "R3", "name": "重要度突然跃升",
     "trigger": "SHAP 环比跃升 > 5 倍",
     "action": "先查数据（该因子/其他因子覆盖率突变、数据源变更），查清前不动用"},
]

FROZEN = {
    "full_market": {"features": ["gm", "dgm_q"],
                    "signs": {"gm": +1, "dgm_q": +1},
                    "coverage": {"gm": 0.980, "dgm_q": 0.962}},
    "manufacturing": {"features": ["transfer_intensity", "dep_rate"],
                      "signs": {"transfer_intensity": +1, "dep_rate": +1},
                      "coverage": {"transfer_intensity": 0.198, "dep_rate": 0.198}},
    "retired": ["gm_vol"],
    "frozen_at": "2026-11-20",
    "rule": "冻结后进 M 段，特征不得再改；任何改动需新建版本并重跑 F7",
}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


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
    return fp


def psi(base, cur, bins=10, eps=1e-4):
    base = pd.Series(base).dropna()
    cur = pd.Series(cur).dropna()
    if len(base) < 100 or len(cur) < 100:
        return np.nan
    qs = np.quantile(base, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.nan
    b = np.clip(np.histogram(base, bins=qs)[0] / len(base), eps, None)
    c = np.clip(np.histogram(cur, bins=qs)[0] / len(cur), eps, None)
    return float(np.sum((c - b) * np.log(c / b)))


def verdict(p):
    if not np.isfinite(p): return "样本不足"
    if p < 0.10: return "稳定"
    if p < 0.25: return "关注"
    return "**重训**"


def render(psi_res):
    L = []
    L.append("# feature_governance.md —— 特征治理（F8）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　冻结日 {FROZEN['frozen_at']}\n")

    L.append("## 一、特征版本表（G-F7.9）\n")
    L.append("| 特征 | 版本 | 生效日 | 状态 | 定义 | 输入 | 窗口 | 参考系 | 表示 | 备注 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in VERSIONS:
        st = "**active**" if v["status"] == "active" else "~~retired~~"
        L.append(f"| `{v['name']}` | {v['version']} | {v['effective']} | {st} | "
                 f"{v['defn']} | {v['inputs']} | {v['window']} | {v['ref']} | "
                 f"{v['rep']} | {v['note']} |")
    L.append("")
    L.append("**规则**：改含义 → 主版本+1；降噪/符号修正 → 次版本+1；退役 → 状态改 retired 但定义保留。\n")

    L.append("## 二、PSI 漂移监控（G-F7.7）\n")
    L.append("**口径**：用 `available_at` 分期（基准 = 训练窗 <2025-01-01，当前 = ≥2025-01-01）。")
    L.append("这与 D7 用 `declare` 分期的结果不同——D7 只有 8.5% 样本，此处 100%。\n")
    L.append("| 特征 | 基准期样本 | 当前期样本 | PSI | 判定 |")
    L.append("|---|---|---|---|---|")
    for r in psi_res:
        L.append(f"| `{r['factor']}` | {r['n_base']:,} | {r['n_cur']:,} | "
                 f"{r['psi']:.4f} | {r['verdict']} |")
    L.append("")
    L.append("**阈值**：< 0.10 稳定 ｜ 0.10~0.25 关注 ｜ > 0.25 重训\n")

    L.append("## 三、停用与退役规则（G-F7.8）\n")
    L.append("| ID | 规则 | 触发条件 | 动作 |")
    L.append("|---|---|---|---|")
    for r in RETIREMENT_RULES:
        L.append(f"| {r['id']} | {r['name']} | {r['trigger']} | {r['action']} |")
    L.append("")
    L.append("**R3 为什么最反直觉**：重要度突然跃升 ≠ 因子突然有效。")
    L.append("更可能是另一个因子数据断供，模型只能依赖剩下的那个，导致其重要度被动上升。")
    L.append("所以先查三步：①该因子覆盖率突变？②其他因子覆盖率突变？③是否与数据源变更吻合？\n")

    L.append("## 四、变更日志\n")
    L.append("| 日期 | 特征 | 改动 | 类型 | 版本 | 理由 |")
    L.append("|---|---|---|---|---|---|")
    for c in CHANGELOG:
        L.append(f"| {c['date']} | `{c['feature']}` | {c['change']} | "
                 f"{c['type']} | {c['version']} | {c['reason']} |")
    L.append("")

    L.append("## 五、冻结的最终特征清单\n")
    L.append("**为什么分两套**：`transfer_intensity` 与 `dep_rate` 覆盖率仅 19.8%（只有制造业披露附注），")
    L.append("不能外推到全市场；若强行与全市场因子合成，实际只有 19.8% 的股票四个因子齐全。\n")
    for name, cfg in [("全市场样本", FROZEN["full_market"]),
                      ("制造业子样本", FROZEN["manufacturing"])]:
        L.append(f"### {name}\n")
        L.append("| 特征 | 方向 | 覆盖率 |")
        L.append("|---|---|---|")
        for f in cfg["features"]:
            L.append(f"| `{f}` | {cfg['signs'][f]:+d} | {cfg['coverage'][f]*100:.1f}% |")
        L.append("")
    L.append(f"**退役**：{', '.join(f'`{x}`' for x in FROZEN['retired'])}\n")
    L.append(f"**冻结规则**：{FROZEN['rule']}\n")

    L.append("## 六、已知局限\n")
    L.append("1. PSI 对重尾/大量重复值的变量不稳（D7 已发现 dgm_q 的 PSI 异常高）")
    L.append("2. 退役规则 R2 依赖 SHAP，而本项目尚未进入 M 段，SHAP 未计算")
    L.append("3. 两套清单的权重需在 M 段确定，此处只冻结「用哪些特征」")
    L.append("4. 符号修正是用训练窗实测 IC 决定的，存在「样本内选方向」的轻微乐观偏差")
    return "\n".join(L)


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行 / {df['code'].nunique():,} 只")

    log("PSI 计算（available_at 分期）…")
    base = df[df["available_at"] < SPLIT]
    cur = df[df["available_at"] >= SPLIT]
    log(f"  基准 {len(base):,} 行 / 当前 {len(cur):,} 行")

    res = []
    for f in FACTORS:
        b = base[f].dropna(); c = cur[f].dropna()
        p = psi(b, c)
        res.append({"factor": f, "n_base": len(b), "n_cur": len(c),
                    "psi": p, "verdict": verdict(p)})
        log(f"  {f:22s} PSI {p:8.4f}  {verdict(p)}")

    log("写出 …")
    (ROOT / "feature_governance.md").write_text(render(res), encoding="utf-8")
    (CLEAN / "meta" / "feature_governance.json").write_text(
        json.dumps({"versions": VERSIONS, "changelog": CHANGELOG,
                    "rules": RETIREMENT_RULES, "frozen": FROZEN,
                    "psi": res}, ensure_ascii=False, indent=2), encoding="utf-8")
    log("  → feature_governance.md / feature_governance.json")


if __name__ == "__main__":
    main()