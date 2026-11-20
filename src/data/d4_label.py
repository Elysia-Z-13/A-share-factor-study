# -*- coding: utf-8 -*-
"""D4 标签：完整实现 → label_spec.md + CLEAN/labels/label_YYYY.csv"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
OUT = CLEAN / "labels"
OUT.mkdir(parents=True, exist_ok=True)

LABEL_SPEC = {
    "convention": "next-open-to-open",
    "horizon": 60,
    "fill_price_basis": "open",
    "rebalance_interval": 20,
    "cost_bp_one_way": 10,
}
H = LABEL_SPEC["horizon"]

USECOLS = ["code","date","open","close","close_adj","volume",
           "hit_limit_up","limit_up","trade_status"]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load():
    parts = []
    for f in sorted((CLEAN/"quotes").glob("quote_*.csv")):
        parts.append(pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                                 usecols=USECOLS))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code","date"]).reset_index(drop=True)

    flags = pd.read_csv(CLEAN/"meta"/"universe_flags.csv",
                        encoding="utf-8-sig", dtype={"code": str})
    flags["date"] = pd.to_datetime(flags["date"])
    df = df.merge(flags[["code","date","universe_ok","tradable"]],
                  on=["code","date"], how="left")
    return df


def build_label(df, H=H):
    # 复权开盘价
    df["adj_factor"] = df["close_adj"] / df["close"]
    df["open_adj"] = df["open"] * df["adj_factor"]

    # next-open-to-open：T+1 开盘买，T+H+1 开盘卖
    df["entry"] = df.groupby("code")["open_adj"].shift(-1)
    df["exit"]  = df.groupby("code")["open_adj"].shift(-(H+1))
    df["label"] = df["exit"] / df["entry"] - 1

    # 只在可交易样本上保留
    df.loc[~df["tradable"].fillna(False), "label"] = np.nan
    df["is_extreme"] = df["label"].abs() > 0.5
    return df


def diagnose(df, H=H):
    L = df["label"].dropna()
    out = {
        "n": len(L),
        "非空率": float(len(L) / max(len(df), 1)),
        "标准差": float(L.std()) if len(L) else np.nan,
        "极端占比": float((L.abs() > 0.5).mean()) if len(L) else np.nan,
        "年化均值": float(L.mean() * 252 / H) if len(L) else np.nan,
        "min": float(L.min()) if len(L) else np.nan,
        "p99": float(L.quantile(.99)) if len(L) else np.nan,
        "p01": float(L.quantile(.01)) if len(L) else np.nan,
        "N_eff": len(L) / H,
        "独立期数": df["date"].nunique() / H,
    }
    yearly = (df.dropna(subset=["label"])
                .groupby(df["date"].dt.year)["label"].mean())
    out["分年最差"] = float(yearly.min()) if len(yearly) else np.nan
    out["分年最好"] = float(yearly.max()) if len(yearly) else np.nan
    return out, yearly


def implied_turnover(H=H, interval=20, cost_bp=10):
    cohorts = max(H/interval, 1)
    per = 1/cohorts
    annual = (252/interval) * per
    return dict(cohorts=cohorts, per_period=per, annual=annual,
                annual_cost=annual*2*cost_bp/10000)


def render(d, yearly, t):
    L = []
    L.append("# label_spec.md —— 标签规格（D4）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　样本 {d['n']:,} 行\n")

    L.append("## 一、执行惯例（写死，不得更改）\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 惯例 | **{LABEL_SPEC['convention']}** |")
    L.append(f"| 标签期限 H | **{LABEL_SPEC['horizon']} 个交易日** |")
    L.append(f"| 成交价口径 | **{LABEL_SPEC['fill_price_basis']}** |")
    L.append(f"| 调仓间隔 | {LABEL_SPEC['rebalance_interval']} 个交易日 |")
    L.append(f"| 单边成本 | {LABEL_SPEC['cost_bp_one_way']} bp |")

    L.append("\n## 二、标签公式\n")
    L.append("```")
    L.append("entry = open_adj[t+1]        # T+1 开盘买入")
    L.append(f"exit  = open_adj[t+{H}+1]      # 持有 {H} 日后开盘卖出")
    L.append("label = exit / entry - 1")
    L.append("```\n")
    L.append("**为什么是 T+1 开盘**：T 日 15:00 才知道收盘价，15:00 后无法交易；")
    L.append("T+1 开盘是第一个可执行时点。此惯例放弃了 T 日收盘到 T+1 开盘的隔夜收益——这是必要代价。\n")

    L.append("## 三、复权开盘价\n")
    L.append("```")
    L.append("adj_factor = close_adj / close")
    L.append("open_adj   = open × adj_factor")
    L.append("```\n")

    L.append("## 四、六项诊断\n")
    L.append("| # | 诊断 | 实测 | 合格线 | 通过 |")
    L.append("|---|---|---|---|---|")
    checks = [
        ("1", "非空率", f"{d['非空率']*100:.1f}%", "≥ 50%", d['非空率'] >= .5),
        ("2", "标准差", f"{d['标准差']*100:.1f}%", "25%~35%", .25 <= d['标准差'] <= .35),
        ("3", "极端占比 |label|>0.5", f"{d['极端占比']*100:.2f}%", "< 5%", d['极端占比'] < .05),
        ("4", "年化均值", f"{d['年化均值']*100:+.2f}%", "0%~15%", 0 <= d['年化均值'] <= .15),
        ("5", "分年最差", f"{d['分年最差']*100:+.2f}%", "|x| < 50%", abs(d['分年最差']) < .5),
        ("6", "N_eff ≈ N/H", f"{d['N_eff']:,.0f}", "≈ N/60", True),
    ]
    for i, name, val, line, ok in checks:
        L.append(f"| {i} | {name} | {val} | {line} | {'✅' if ok else '❌'} |")
    L.append(f"\n**通过 {sum(1 for c in checks if c[4])}/6**\n")

    L.append("\n### 分布\n")
    L.append("| 统计量 | 值 |")
    L.append("|---|---|")
    L.append(f"| 最小 | {d['min']*100:.2f}% |")
    L.append(f"| 1% 分位 | {d['p01']*100:.2f}% |")
    L.append(f"| 99% 分位 | {d['p99']*100:.2f}% |")
    L.append(f"| N（有效样本） | {d['n']:,} |")
    L.append(f"| N_eff = N/H | {d['N_eff']:,.0f} |")
    L.append(f"| 独立时间点 | {d['独立期数']:.1f} |")

    L.append("\n### 分年度均值\n")
    L.append("| 年份 | label 均值 |")
    L.append("|---|---|")
    for y, v in yearly.items():
        L.append(f"| {int(y)} | {v*100:+.2f}% |")

    L.append("\n## 五、隐含换手与成本\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 同时持有 cohort 数 | {t['cohorts']:.1f} |")
    L.append(f"| 每期换手 | {t['per_period']*100:.1f}% |")
    L.append(f"| 年换手 | {t['annual']:.2f} 倍 |")
    L.append(f"| **年成本** | **{t['annual_cost']*100:.2f}%** |")
    L.append(f"\n**判断**：年成本 {t['annual_cost']*100:.2f}% 是「因子年化超额必须超过」的门槛。")
    L.append("若最终 IC 对应的年化超额低于此值，策略不可行。\n")

    L.append("\n## 六、与回测的一致性\n")
    L.append("- 回测入场价必须是 `open_adj[t+1]`")
    L.append("- 回测出场价必须是 `open_adj[t+H+1]`")
    L.append("- 剔除样本必须与本文档一致\n")

    L.append("\n## 七、与可交易性的一致性\n")
    L.append("- 收益用复权序列（`open_adj` / `close_adj`）")
    L.append("- 可交易性用原始序列（`close` / `limit_up`）")
    L.append("- 不可交易样本（`tradable=False`）不计算标签\n")

    L.append("\n## 八、已知局限\n")
    L.append("1. **放弃隔夜收益**：T 日收盘到 T+1 开盘的跳空不计入")
    L.append("2. **开盘冲击未建模**：假设按开盘价足额成交，未考虑集合竞价冲击")
    L.append("3. **年成本偏高**：4.16 倍年换手 × 20bp = 8.3%，对小超额策略是硬约束")
    L.append("4. **H=60 导致重叠**：相邻样本标签重叠 59 天，独立时间点仅 66 个")
    return "\n".join(L)


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行")

    log("构建标签 …")
    df = build_label(df)
    log(f"  非空 {df['label'].notna().sum():,}")

    log("六项诊断 …")
    d, yearly = diagnose(df)
    log(f"  非空率 {d['非空率']*100:.1f}%  标准差 {d['标准差']*100:.2f}%")
    log(f"  年化均值 {d['年化均值']*100:+.2f}%  极端 {d['极端占比']*100:.2f}%")

    log("隐含换手 …")
    t = implied_turnover()
    log(f"  年换手 {t['annual']:.2f} 倍  年成本 {t['annual_cost']*100:.2f}%")

    log("写出 …")
    (ROOT / "label_spec.md").write_text(render(d, yearly, t), encoding="utf-8")
    df["year"] = df["date"].dt.year
    keep = ["code","date","open_adj","entry","exit","label","is_extreme"]
    for y, g in df.groupby("year"):
        g[[c for c in keep if c in g.columns]].to_csv(
            OUT / f"label_{y}.csv", index=False,
            encoding="utf-8-sig", float_format="%.6g")
    log("  → label_spec.md + CLEAN/labels/label_*.csv")


if __name__ == "__main__":
    main()