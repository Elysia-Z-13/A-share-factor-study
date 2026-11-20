# -*- coding: utf-8 -*-
"""
D3 universe 与 mask：执行版（按《D3universe与mask流程手册》附录 A 实现）

输出：
  universe_rule.md            规则文档
  CLEAN/meta/universe_flags.csv  (code,date,universe_ok,tradable)
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
DATA = ROOT / "data"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

MASK_META = {
    "version": "v1.0",
    "effective_date": "2026-11-20",
    "rules": {
        "min_listed_days": 250,
        "exclude_st": True,
        "amount_rank_min": 0.20,
        "require_quote": True,
        "exclude_limit_up": True,
        "exclude_limit_down": False,
    },
}

USECOLS = ["code","date","close","close_adj","volume","amount","float_mv",
           "trade_status","hit_limit_up","hit_limit_down","limit_up","limit_down"]


def load_all():
    parts = []
    for f in sorted((CLEAN / "quotes").glob("quote_*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str}, usecols=USECOLS)
        parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["code","date"]).reset_index(drop=True)

    listing = pd.read_csv(DATA / "listing.csv", encoding="utf-8-sig", dtype={"code": str})
    px = px.merge(listing, on="code", how="left")
    return px


def add_rules(df):
    # ① 上市天数：累计交易日数（自动跳过停牌/节假日）
    df["listed_days"] = df.groupby("code").cumcount() + 1

    # ② ST（trade_status 2/3/6）
    df["st_flag"] = df["trade_status"].isin([2, 3, 6]).astype(int)

    # ③ 成交额分位：20 日均值后按日截面 rank
    df["amount_ma20"] = (df.groupby("code")["amount"]
                           .transform(lambda s: s.rolling(20, min_periods=10).mean()))
    df["amount_rank"] = df.groupby("date")["amount_ma20"].rank(pct=True)

    # ④ 有成交
    df["has_quote"] = df["close"].notna() & (df["volume"].fillna(0) > 0)

    # ⑤ 涨跌停
    df["hit_limit_up"] = df["hit_limit_up"].fillna(False).astype(bool)
    df["hit_limit_down"] = df["hit_limit_down"].fillna(False).astype(bool)
    return df


def universe_ok(df):
    r = MASK_META["rules"]
    m = (df["listed_days"] >= r["min_listed_days"]) & df["has_quote"]
    if r["exclude_st"]:
        m &= (df["st_flag"] == 0)
    if r["amount_rank_min"] > 0:
        m &= (df["amount_rank"] > r["amount_rank_min"])
    return m


def tradable(df):
    r = MASK_META["rules"]
    m = universe_ok(df)
    if r["exclude_limit_up"]:
        m &= (~df["hit_limit_up"])
    if r["exclude_limit_down"]:
        m &= (~df["hit_limit_down"])
    return m


def check_consistency(df):
    u = universe_ok(df)
    t = tradable(df)
    assert (t & ~u).sum() == 0, "tradable 必须是 universe_ok 的子集"
    log(f"  universe_ok {u.sum():,} ({u.mean()*100:.2f}%)")
    log(f"  tradable    {t.sum():,} ({t.mean()*100:.2f}%)")
    log(f"  因涨停剔除  {int((u & ~t).sum()):,}")
    return u, t


def rule_impact(df):
    out = {}
    for name, cond in [
        ("上市≥250日", df["listed_days"] >= 250),
        ("非ST",       df["st_flag"] == 0),
        ("成交额>20%", df["amount_rank"] > 0.20),
        ("有成交",     df["has_quote"]),
    ]:
        out[name] = float(cond.mean())
    out["联合(universe_ok)"] = float(universe_ok(df).mean())
    out["最终(tradable)"] = float(tradable(df).mean())
    return out


def render(df, impact, u, t):
    yearly = df.assign(y=df["date"].dt.year).groupby("y").apply(
        lambda g: pd.Series({
            "股票数": g["code"].nunique(),
            "universe_ok": int(universe_ok(g).sum()),
            "tradable": int(tradable(g).sum()),
            "ST占比%": round(g["st_flag"].mean()*100, 2),
        })).reset_index()

    L = []
    L.append("# universe_rule.md —— universe 与 mask（D3）\n")
    L.append(f"> 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　样本 {len(df):,} 行 / {df['code'].nunique():,} 只\n")

    L.append("## 一、规则版本\n")
    L.append(f"- 版本：**{MASK_META['version']}**")
    L.append(f"- 生效日期：{MASK_META['effective_date']}")
    L.append("- 规则明细：\n")
    L.append("```json")
    L.append(json.dumps(MASK_META["rules"], indent=2, ensure_ascii=False))
    L.append("```\n")

    L.append("## 二、universe 四要素\n")
    L.append("| 要素 | 阈值 | 实现 | 理由 |")
    L.append("|---|---|---|---|")
    L.append("| 上市天数 | ≥ 250 交易日 | `groupby(code).cumcount()+1` | 保证能算 250 日动量等年度级因子 |")
    L.append("| 非 ST | `st_flag == 0` | `trade_status ∈ {2,3,6}` 取反 | ST 涨跌幅 5%，制度扭曲信号 |")
    L.append("| 成交额分位 | > 20% | 20 日均值后按日截面 `rank(pct=True)` | 保证容量与冲击成本可控 |")
    L.append("| 有成交 | `volume > 0 且 close 非空` | CSMAR 停牌日整行缺失 | 停牌无法成交 |")

    L.append("\n## 三、可交易性 mask\n")
    L.append("| 条件 | 处理 | 理由 |")
    L.append("|---|---|---|")
    L.append(f"| 停牌 | 剔除 | 无法成交（含在 universe_ok 的 has_quote） |")
    L.append(f"| 涨停 | **剔除** | 买不进（`hit_limit_up=True`） |")
    L.append(f"| 跌停 | 保留 | 多头买入不受阻；调仓卖出需在回测中单独记录 |")

    L.append("\n## 四、一致性保证\n")
    L.append("特征、标签、回测三处共用同一判定函数：\n")
    L.append("```python")
    L.append("tradable = universe_ok & ~hit_limit_up")
    L.append("# tradable ⊆ universe_ok 已用 assert 校验")
    L.append("```\n")
    L.append("实测：`(tradable & ~universe_ok).sum() == 0` ✅\n")

    L.append("\n## 五、覆盖率影响\n")
    L.append("| 规则 | 保留比例 | 剔除比例 |")
    L.append("|---|---|---|")
    for k, v in impact.items():
        L.append(f"| {k} | {v*100:.2f}% | {(1-v)*100:.2f}% |")

    L.append("\n### 分年度样本量\n")
    L.append("| 年份 | 股票数 | universe_ok | tradable | ST占比 |")
    L.append("|---|---|---|---|---|")
    for _, r in yearly.iterrows():
        L.append(f"| {int(r['y'])} | {int(r['股票数']):,} | {int(r['universe_ok']):,} | {int(r['tradable']):,} | {r['ST占比%']}% |")

    L.append("\n## 六、已知局限\n")
    L.append("1. **上市天数用 cumcount**：数据自 2010-01-04 起，2010 年前上市的股票其 listed_days 从 1 开始，导致 2010-2011 年 universe 偏小。这是数据起点截断，需在报告声明。")
    L.append("2. **成交额分位按当日截面**：极端行情日（如 2015 股灾）分位不稳定。")
    L.append("3. **北交所未单独处理**：涨跌幅 30%、流动性显著低于主板，未做差异化规则。")
    L.append("4. **退市股处理**：delist_date 由行情末行推导（当时可观察），未使用今日退市名单，避免幸存者偏差。")

    L.append("\n## 七、对后续步骤的约束\n")
    L.append("```")
    L.append("D4 标签：只在 tradable=True 的样本上算标签")
    L.append("F1 因子：只在 tradable=True 的截面上排序")
    L.append("E4 回测：只在 tradable=True 的股票中选股")
    L.append("```")
    return "\n".join(L)


def main():
    log("加载数据 …")
    df = load_all()
    log(f"  {len(df):,} 行 / {df['code'].nunique():,} 只")

    log("应用规则 …")
    df = add_rules(df)

    log("一致性检查 …")
    u, t = check_consistency(df)

    log("规则影响 …")
    impact = rule_impact(df)
    for k, v in impact.items():
        log(f"  {k:20s} 保留 {v*100:6.2f}%")

    log("写出 …")
    (ROOT / "universe_rule.md").write_text(render(df, impact, u, t), encoding="utf-8")
    out = CLEAN / "meta" / "universe_flags.csv"
    df[["code","date","listed_days","st_flag","amount_rank",
        "hit_limit_up"]].assign(universe_ok=u, tradable=t).to_csv(
        out, index=False, encoding="utf-8-sig", float_format="%.6g")
    log(f"  → universe_rule.md / {out.name}")


if __name__ == "__main__":
    main()
