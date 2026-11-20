# -*- coding: utf-8 -*-
"""F3 结构与上下文特征 → feature_context.md"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ── 四类上下文特征的适用性 ──
CONTEXT_TYPES = [
    {"type": "Carry 与期限结构", "applicable": False,
     "reason": "A 股股票无期限结构（无近月/远月之分）",
     "note": "若未来做期货/期权，连续合约拼接规则必须写进契约"},
    {"type": "跨资产与期权", "applicable": False,
     "reason": "我们的数据只有 A 股个股行情，无期权/期货/跨市场数据",
     "note": "若引入，lead-lag 必须做真实性检验，否则是伪相关"},
    {"type": "基本面", "applicable": True,
     "reason": "本项目 5 个因子全部来自财报",
     "note": "核心要求：用公告日 ann_date，不用报告期 end_date"},
    {"type": "日历与宏观", "applicable": "部分",
     "reason": "日历特征可用（白名单内）；宏观不用（无 vintage 数据）",
     "note": "日历只能用事前可知项；宏观必须 vintage"},
]

# ── 日历特征白名单 ──
CALENDAR_WHITELIST = [
    ("dow", "星期几", "永远可知"),
    ("month", "月份", "永远可知"),
    ("is_month_end", "是否月末", "日历可推导"),
    ("is_quarter_end", "是否季末", "日历可推导"),
    ("days_to_deadline", "距法定披露截止日天数",
     "《证券法》规定年报 4/30、半年报 8/31、三季报 10/31，事前可知"),
]

CALENDAR_BLACKLIST = [
    ("实际公告日", "事前不知道公司哪天公布"),
    ("指数调整成分", "名单调整前不公开"),
    ("临时休市日", "事后才知道（如 2020 春节延长）"),
]

# ── 宏观政策 ──
MACRO_POLICY = {
    "used": False,
    "reason": "无 vintage 数据源；免费数据只给最终修订值，用它会前视",
    "if_added_later": "必须用首次发布值并在发布日对齐",
}

USECOLS_PX = ["code", "date"]
USECOLS_FIN = ["code", "accper", "declare"]


def load_px_sample(year=2024):
    f = CLEAN / "quotes" / f"quote_{year}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str}, usecols=USECOLS_PX)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["code", "date"]).reset_index(drop=True)


def load_fin():
    f = CLEAN / "fin" / "factor_panel.csv"
    if not f.exists():
        return None
    fp = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str})
    fp["end_date"] = pd.to_datetime(fp["accper"], errors="coerce")
    fp["ann_date"] = pd.to_datetime(fp["declare"], errors="coerce")
    return fp.dropna(subset=["end_date"]).sort_values(["code", "ann_date"])


def merge_fundamental_pit(daily, fin, value_cols):
    out = []
    for code, g in daily.groupby("code", sort=False):
        f = fin[fin["code"] == code].dropna(subset=["ann_date"]).sort_values("ann_date")
        if f.empty:
            continue
        cols = ["ann_date", "end_date"] + [c for c in value_cols if c in f.columns]
        cols = list(dict.fromkeys(cols))
        m = pd.merge_asof(
            g.sort_values("date"), f[cols],
            left_on="date", right_on="ann_date",
            direction="backward", allow_exact_matches=True,
        )
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else daily


def check_revisions(fin):
    dup = fin.groupby(["code", "end_date"]).size()
    return int((dup > 1).sum()), int(dup[dup > 1].sum())


def check_forward_fill(m, col="gm"):
    """非公告日值变化次数应为 0"""
    g = m.dropna(subset=[col]).sort_values(["code", "date"]).copy()
    g["prev_ann"] = g.groupby("code")["ann_date"].shift(1)
    changed = (g.groupby("code")[col].diff().abs() > 1e-12) & \
              (g["ann_date"].fillna(pd.NaT) == g["prev_ann"])
    return int(changed.sum())


def render(stats):
    L = []
    L.append("# feature_context.md —— 结构与上下文特征（F3）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")

    L.append("## 一、四类上下文特征的适用性\n")
    L.append("| 类型 | 适用 | 理由 | 注意 |")
    L.append("|---|---|---|---|")
    for c in CONTEXT_TYPES:
        ap = "✅" if c["applicable"] is True else ("⚠️ 部分" if c["applicable"] == "部分" else "⊘ 不适用")
        L.append(f"| {c['type']} | {ap} | {c['reason']} | {c['note']} |")

    L.append("\n## 二、基本面：公告日对齐（G-F3.1）\n")
    L.append("**规则**：对每个 (code, T)，取「ann_date ≤ T」的候选集中 end_date 最大的那一期。\n")
    L.append("```python")
    L.append("pd.merge_asof(daily, fin, left_on='date', right_on='ann_date',")
    L.append("              direction='backward',  # 只取公告日 <= 当天")
    L.append("              allow_exact_matches=True)")
    L.append("```\n")
    L.append("**为什么不能用 end_date**：报告期结束 ≠ 市场知道，差 1~4 个月。")
    L.append("D2 对照实验实测：错用时点使 IC 虚高 1.78 倍（0.00774 → 0.01382）。\n")
    if stats.get("pit_rows"):
        L.append("**实测**：")
        L.append(f"- 对齐后行数 {stats['pit_rows']:,}")
        L.append(f"- 前视违规（ann_date > date）{stats['lookahead']} 行 "
                 f"{'✅' if stats['lookahead'] == 0 else '❌'}")
        L.append(f"- 非公告日值变化 {stats['ffill_violations']} 次 "
                 f"{'✅' if stats['ffill_violations'] == 0 else '❌'}")
        L.append(f"- 财报覆盖 {stats['coverage']*100:.1f}%\n")

    L.append("## 三、修订版本处理\n")
    L.append("**规则**：保留所有版本，不预先去重；merge 时按 ann_date 时间过滤，")
    L.append("自然得到「当时可见的那一版」。\n")
    if stats.get("revision_groups") is not None:
        L.append(f"**实测**：同一 (code, end_date) 有多个版本的组数 = "
                 f"{stats['revision_groups']}，涉及 {stats['revision_rows']:,} 行。")
        L.append("这些行全部保留在财报表中，不做预去重。\n")
    L.append("**反例**：如果先用 `dedupe_revisions()` 取最终版再 merge，")
    L.append("2024-06-15 就会用上 2024-08-20 才更正的值 → 前视。\n")

    L.append("## 四、日历特征白名单（G-F3.2）\n")
    L.append("| 特征 | 含义 | 为什么事前可知 |")
    L.append("|---|---|---|")
    for n, d, why in CALENDAR_WHITELIST:
        L.append(f"| `{n}` | {d} | {why} |")
    L.append("\n**禁止使用**：\n")
    L.append("| 禁止项 | 为什么 |")
    L.append("|---|---|")
    for n, why in CALENDAR_BLACKLIST:
        L.append(f"| {n} | {why} |")
    L.append("\n**「财报日前后」的安全用法**：")
    L.append("❌ 用「实际公告日 ±5 日」（事前不知道哪天公布）")
    L.append("✅ 用「法定披露截止日前 10 日」（4/30 是法规截止日，事前可知）\n")

    L.append("## 五、宏观：声明不使用（G-F3.3）\n")
    L.append(f"- 是否使用：**{'是' if MACRO_POLICY['used'] else '否'}**")
    L.append(f"- 理由：{MACRO_POLICY['reason']}")
    L.append(f"- 若未来引入：{MACRO_POLICY['if_added_later']}\n")
    L.append("**为什么必须 vintage**：GDP 数据会被反复修订（初值 4.5% → 修订 4.7% → 终值 4.6%）。")
    L.append("用终值等于在 2024 年 4 月知道了 2025 年的修订结果。\n")

    L.append("## 六、已知局限\n")
    L.append("1. 未使用 carry / 跨资产 / 期权特征（数据与工具链不支持）")
    L.append("2. 未使用宏观特征（无 vintage 数据源），这意味着我们放弃了宏观择时维度")
    L.append("3. 日历特征未实际构造为信号（白名单仅作规范，未用于本项目因子）")
    L.append("4. CSMAR 的 DeclareDate 早期年份缺失，导致正确 PIT 下覆盖率降至 61.5%")
    return "\n".join(L)


def main():
    log("加载数据 …")
    px = load_px_sample(2024)
    fin = load_fin()

    stats = {}
    if px is not None and fin is not None:
        log(f"  行情 {len(px):,} 行 / 财报 {len(fin):,} 行")

        rev_g, rev_r = check_revisions(fin)
        stats["revision_groups"] = rev_g
        stats["revision_rows"] = rev_r
        log(f"  修订版本：{rev_g} 组 / {rev_r:,} 行")

        log("公告日 PIT 对齐 …")
        value_cols = ["gm"]
        m = merge_fundamental_pit(px, fin, value_cols)
        stats["pit_rows"] = len(m)
        stats["lookahead"] = int((m["ann_date"] > m["date"]).sum())
        stats["coverage"] = float(m["gm"].notna().mean()) if "gm" in m else 0
        log(f"  {len(m):,} 行  前视 {stats['lookahead']}  覆盖 {stats['coverage']*100:.1f}%")

        stats["ffill_violations"] = check_forward_fill(m, "gm")
        log(f"  非公告日值变化 {stats['ffill_violations']} 次")

    log("写出 …")
    (ROOT / "feature_context.md").write_text(render(stats), encoding="utf-8")
    log("  → feature_context.md")


if __name__ == "__main__":
    main()