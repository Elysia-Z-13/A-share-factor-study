# -*- coding: utf-8 -*-
"""F0 总则：三步滤网 → feature_plan.md"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"

H = 60

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


# ── 第一步：时限对齐 ──
FEATURES = [
    {"name": "gm", "lookback": "最近一期财报", "days": 60,
     "match": "等长", "freshness_ok": True,
     "desc": "毛利率 = (营业收入 − 营业成本) / 营业收入"},
    {"name": "dgm_q", "lookback": "最近两期", "days": 60,
     "match": "等长", "freshness_ok": True,
     "desc": "单季毛利率变化 = gm_t − gm_{t-1}"},
    {"name": "transfer_intensity", "lookback": "最近一期（年报）", "days": 250,
     "match": "4×H 错配", "freshness_ok": False,
     "desc": "转固强度 = 在建工程转入 / 固定资产原值期初"},
    {"name": "dep_rate", "lookback": "最近一期（年报）", "days": 250,
     "match": "4×H 错配", "freshness_ok": False,
     "desc": "折旧率 = 累计折旧本期增加 / 固定资产原值期初"},
    {"name": "gm_vol", "lookback": "最近 4 期", "days": 240,
     "match": "倍长 4×H", "freshness_ok": True,
     "desc": "毛利率 4 季滚动标准差"},
]

# ── 第二步：驱动假设 ──
# 手册 F0.2 的四类驱动：持续性 / 均值回复 / 风险补偿 / 可预测时钟
HYPOTHESES = {
    "gm": ("风险补偿（质量溢价）：高毛利=竞争壁垒，市场对质量要求补偿",
           +1, "强", "风险偏好反转时 IC 应下降"),
    "dgm_q": ("持续性：毛利率改善会延续（基本面动量），市场反应不足",
              +1, "中", "震荡市 IC 应下降"),
    "transfer_intensity": ("持续性：产能在建工程转固→产能落地→营收增长是持续过程",
                           +1, "中", "产能过剩/需求不足时失效"),
    "dep_rate": ("持续性（弱）：资产老化→持续资本开支压力",
                 -1, "弱", "公司完成资产更新后失效"),
    "gm_vol": ("风险补偿（低波动异象）：投资者偏好彩票型，低估稳定股",
               -1, "中", "风险偏好反转时 IC 应下降"),
}

# ── 第三步：角色分离 ──
ROLES = {
    "signal": ["gm", "dgm_q", "transfer_intensity", "dep_rate", "gm_vol"],
    "state": ["industry_code", "float_mv", "st_flag", "listed_days"],
    "label": ["label"],
}

# ── 改动分类 ──
CHANGE_LOG = [
    ("transfer_intensity 分母从 cip_prev 改为 fa_cost_begin", "改含义",
     "分母从在建工程变成固定资产；经济含义从转固完成率变为投资强度"),
    ("缩尾 1% → 0.5%", "降噪", "只影响极端值处理"),
    ("MAD 3σ → 3.5σ", "降噪", "同上"),
    ("单季差分 → 累计同比", "改含义", "时间尺度改变"),
    ("填中位数 → 填行业中位数", "改含义", "隐含行业内可比假设"),
    ("z-score → rank", "降噪", "单调变换，秩不变"),
]


def check_roles():
    s, st, l = (set(ROLES["signal"]), set(ROLES["state"]), set(ROLES["label"]))
    assert not (s & st), f"signal 与 state 重叠：{s & st}"
    assert not (s & l), f"signal 与 label 重叠：{s & l}"
    assert not (st & l), f"state 与 label 重叠：{st & l}"
    return True


def check_horizon():
    """回看窗与 H 的比例应在 [0.5, 3] 之间

    为什么上限是 3 而不是 5：
      4 倍以上的错配会让 IC 被稀释到一半以下（√(1/4)=0.5）
      → 等于白扔一半的信号
    """
    bad = []
    for f in FEATURES:
        ratio = f["days"] / H
        if ratio < 0.5 or ratio > 3:
            bad.append((f["name"], round(ratio, 2)))
    return bad


def render():
    L = []
    L.append("# feature_plan.md —— 特征工程总则（F0）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　标签期限 H = {H} 交易日\n")

    L.append("## 一、时限对齐（Horizon Alignment）\n")
    L.append("| 因子 | 回看窗 | 天数 | 与 H 关系 | 说明 |")
    L.append("|---|---|---|---|---|")
    for f in FEATURES:
        L.append(f"| `{f['name']}` | {f['lookback']} | {f['days']} | "
                 f"{f['match']} | {f['desc']} |")
    L.append("\n**规则**：回看窗应 ≈ H 或 = k×H。")
    L.append("财务因子天然满足——财报季度更新 ≈ 60 个交易日。")
    bad = check_horizon()
    L.append(f"\n**尺度错配检查**：{'无' if not bad else bad}\n")

    L.append("## 二、驱动假设（Driver Hypothesis）\n")
    L.append("| 因子 | 驱动类别 | 机制假设 | 方向 | 强度 | 失效条件 |")
    L.append("|---|---|---|---|---|---|")
    for k, (h, sign, strength, fail) in HYPOTHESES.items():
        cat = "风险补偿" if "风险补偿" in h else ("持续性" if "持续性" in h else "均值回复" if "均值回复" in h else "可预测时钟")
        L.append(f"| `{k}` | **{cat}** | {h} | {'+' if sign > 0 else '−'} | {strength} | {fail} |")
    L.append("\n**驱动分布**：风险补偿 2 个（gm、gm_vol）、持续性 3 个（dgm_q、transfer_intensity、dep_rate）、"
             "**均值回复 0 个、可预测时钟 0 个**。")
    L.append("说明因子集合同质化，组合分散效果有限——这是 F6 去重的重点，也是未来补因子的方向。")
    L.append("\n**可证伪性检查**：每个假设都能回答「如果错了会观察到什么」。")
    L.append("例如「毛利率不代表定价能力」→ 应观察到 IC ≈ 0。\n")
    L.append("**为什么必须有假设**：试 N 个无意义因子，H0 下 |t|>3 的期望个数 = N×0.0027。")
    L.append("N=1000 时会有 2.7 个「显著」，但它们样本外必然归零。\n")

    L.append("## 三、角色分离（Role Separation）\n")
    L.append("| 角色 | 变量 | 用法 |")
    L.append("|---|---|---|")
    L.append(f"| **signal** | {', '.join(f'`{x}`' for x in ROLES['signal'])} | 进因子排序 |")
    L.append(f"| **state** | {', '.join(f'`{x}`' for x in ROLES['state'])} | 分组/过滤，不进排序 |")
    L.append(f"| **label** | {', '.join(f'`{x}`' for x in ROLES['label'])} | 预测目标 |")
    L.append("\n**为什么分开**：把行业当 signal 会让因子变成「选哪个行业」，不是「选哪只股票」。")
    L.append("state 只能用于 groupby（行业内排序）或过滤。\n")
    L.append(f"**重叠检查**：{'通过 ✅' if check_roles() else '失败'}\n")

    L.append("## 四、改含义 vs 降噪\n")
    L.append("| 改动 | 分类 | 理由 |")
    L.append("|---|---|---|")
    for c, t, why in CHANGE_LOG:
        L.append(f"| {c} | **{t}** | {why} |")
    L.append("\n**纪律**：改含义 = 新实验，必须重跑全套并重新记账；")
    L.append("降噪 = 同假设，只需重跑评估。\n")

    L.append("## 五、因子清单与优先级\n")
    L.append("| 优先级 | 因子 | 理由 |")
    L.append("|---|---|---|")
    L.append("| P1 | `gm` | 覆盖 98%，机制最强 |")
    L.append("| P2 | `dgm_q` | 覆盖 96.2%，但 PSI=0.49 需关注 |")
    L.append("| P3 | `transfer_intensity` | 覆盖仅 19.8%，只对制造业有效 |")
    L.append("| P4 | `dep_rate` | 覆盖 19.8%，假设最弱 |")
    L.append("| P5 | `gm_vol` | 覆盖 94.4%，降噪型因子 |")
    return "\n".join(L)


def main():
    log("时限对齐检查 …")
    bad = check_horizon()
    log(f"  尺度错配：{bad if bad else '无'}")

    log("角色重叠检查 …")
    check_roles()
    log("  通过")

    log("写出 …")
    (ROOT / "feature_plan.md").write_text(render(), encoding="utf-8")
    log("  → feature_plan.md")


if __name__ == "__main__":
    main()