# -*- coding: utf-8 -*-
"""F2 价格衍生特征 → feature_price.md（含对照因子实测）"""
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

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ── 构件定义 ──
COMPONENTS = {
    "动量": "过去涨得多→未来继续涨",
    "反转": "涨太多会回落（锚决定含义）",
    "波动": "风险大小（必须方差型估计量）",
    "量价流动性": "交易活跃度（右偏→取对数）",
    "技术指标组合": "上面四类的组合，不带来新机制",
}

# ── 特征 → 构件 映射 ──
FEATURE_MAP = [
    ("gm", "基本面", "F3 范围，不在 F2"),
    ("dgm_q", "基本面", "F3 范围"),
    ("transfer_intensity", "基本面", "F3 范围"),
    ("dep_rate", "基本面", "F3 范围"),
    ("gm_vol", "波动（财务口径）", "波动构件的变体，用财务数据"),
    ("mom_60", "动量", "对照因子"),
    ("rev_5_ma60", "反转", "锚=MA60，偏离窗=5 日"),
]

# ── 技术指标拆解 ──
INDICATOR_DECOMP = [
    ("MACD", "EMA(12) − EMA(26)", "动量（短窗−长窗）"),
    ("RSI", "100 − 100/(1+涨幅/跌幅)", "动量（归一化）"),
    ("布林带 %B", "(price−MA)/(2k·σ)+0.5", "反转（锚=均线）+ 波动"),
    ("KDJ", "基于 RSV 的平滑", "反转（锚=区间位置）"),
]

# ── 锚 ──
ANCHORS = {
    "rev_5_ma60": {
        "anchor": "MA60（60 日均线）",
        "anchor_window": 60,
        "deviation_window": 5,
        "meaning": "短期偏离中期趋势 → 反转构件",
    },
}

# ── 波动估计量 ──
VOL_EST = [
    ("收盘价标准差", "std(daily_ret, T)", "1×", "基准"),
    ("Parkinson", "sqrt(mean(ln(H/L)²)/(4ln2))", "5×", "用日内高低"),
    ("Garman-Klass", "0.5ln(H/L)²−(2ln2−1)ln(C/O)²", "7×", "加开收盘"),
    ("Yang-Zhang", "组合估计量", "8~14×", "处理跳空与趋势"),
    ("ATR", "mean(|H−L|,|H−C_prev|,|L−C_prev|)", "N/A",
     "❌ 绝对变动均值，不是方差估计量，不能当波动率"),
]

INDICATORS = ["MACD", "RSI", "KDJ", "BOLL", "布林", "金叉", "死叉"]


def check_no_technical_mechanism():
    bad = []
    for name, comp, note in FEATURE_MAP:
        if any(i in note for i in INDICATORS) and "构件" not in note:
            bad.append((name, note))
    return bad


def load_price_sample():
    """取一年数据做对照因子实测"""
    f = CLEAN / "quotes" / "quote_2024.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                    usecols=["code","date","open","high","low","close","close_adj","amount"])
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values(["code","date"]).reset_index(drop=True)


def build_controls(df, H=H):
    """构造两个对照因子：mom_60（动量）、rev_5_ma60（反转）"""
    g = df.groupby("code")
    df["ret"] = g["close_adj"].pct_change()

    # 动量：60 日累计收益
    df["mom_60"] = g["close_adj"].transform(lambda s: s / s.shift(H) - 1)

    # 反转：相对 MA60 的偏离，取最近 5 日均值
    ma60 = g["close_adj"].transform(lambda s: s.rolling(H, min_periods=40).mean())
    dev = df["close_adj"] / ma60 - 1
    df["rev_5_ma60"] = dev.groupby(df["code"]).transform(
        lambda s: s.rolling(5, min_periods=3).mean())

    # 波动（方差型）
    df["vol_close"] = g["ret"].transform(lambda s: s.rolling(H).std())
    df["vol_parkinson"] = (np.log(df["high"]/df["low"])**2).groupby(
        df["code"]).transform(lambda s: s.rolling(H).mean())
    df["vol_parkinson"] = np.sqrt(df["vol_parkinson"] / (4*np.log(2)))

    # 标签（用 close_adj 未来 H 日）
    df["label"] = g["close_adj"].transform(lambda s: s.shift(-H) / s - 1)
    return df


def spear(x, y):
    # pandas 3.0 下 .values 可能是只读视图，用 to_numpy(copy=True)
    rx = pd.Series(x).rank().to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank().to_numpy(dtype=float, copy=True)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float((rx*ry).sum()/den) if den > 0 else np.nan


def ic_by_date(df, factor):
    ics = []
    for d, g in df.groupby("date"):
        s = g[[factor, "label"]].dropna()
        if len(s) < 100:
            continue
        v = spear(s[factor].values, s["label"].values)
        if np.isfinite(v):
            ics.append(v)
    return pd.Series(ics)


def render(df, ic_res, bad_tech):
    L = []
    L.append("# feature_price.md —— 价格衍生特征（F2）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")

    L.append("## 一、五类构件\n")
    L.append("| 构件 | 本质 |")
    L.append("|---|---|")
    for k, v in COMPONENTS.items():
        L.append(f"| **{k}** | {v} |")

    L.append("\n## 二、本项目因子的归属（为什么几乎不做价格衍生）\n")
    L.append("| 特征 | 构件 | 说明 |")
    L.append("|---|---|---|")
    for name, comp, note in FEATURE_MAP:
        L.append(f"| `{name}` | {comp} | {note} |")
    L.append("\n**结论**：5 个主因子中 4 个属于「基本面」构件（不在 F2 范围），")
    L.append("只有 `gm_vol` 属于「波动」构件（但用财务数据而非价格）。")
    L.append("本项目**不产生价格衍生特征**，F2 的作用是方法论确认 + 防止未来引入时重复计数。\n")

    L.append("## 三、技术指标拆解（F2.2）\n")
    L.append("| 指标 | 公式 | 本质构件 |")
    L.append("|---|---|---|")
    for name, f, comp in INDICATOR_DECOMP:
        L.append(f"| {name} | `{f}` | {comp} |")
    L.append("\n**规则**：先能用构件表达，就不用指标；指标只作为变体，不额外带来机制。")
    L.append(f"\n**反向检查**：把技术指标当机制的条目 = {len(bad_tech)} 个 "
             f"{'✅' if not bad_tech else '❌ ' + str(bad_tech)}\n")

    L.append("## 四、锚登记（F2.3）\n")
    L.append("| 特征 | 锚 | 锚窗口 | 偏离窗口 | 含义 |")
    L.append("|---|---|---|---|---|")
    for name, a in ANCHORS.items():
        L.append(f"| `{name}` | {a['anchor']} | {a['anchor_window']} | "
                 f"{a['deviation_window']} | {a['meaning']} |")
    L.append("\n**为什么锚窗口与偏离窗口要分开**：")
    L.append("若同窗口，`price/MA60−1` 测的是「60 日反转」；分开后测「短期偏离中期趋势」。\n")

    L.append("## 五、波动估计量（F2.4）\n")
    L.append("| 估计量 | 公式 | 相对效率 | 说明 |")
    L.append("|---|---|---|---|")
    for name, f, eff, note in VOL_EST:
        L.append(f"| {name} | `{f}` | {eff} | {note} |")
    L.append("\n**本项目选择**：方差型估计量（`std`）。")
    L.append("明确排除 ATR —— 它是绝对变动的均值，单位是「元」，")
    L.append("直接比较会让高价股被误判为高波动。\n")

    if df is not None and ic_res:
        L.append("## 六、对照因子实测（2024 年，H=60）\n")
        L.append("| 对照因子 | 构件 | 平均 rank IC | IC 期数 |")
        L.append("|---|---|---|---|")
        for name, (comp, mean_ic, n) in ic_res.items():
            L.append(f"| `{name}` | {comp} | {mean_ic:+.4f} | {n} |")
        L.append("\n**用途**：这两个对照因子就是 D2 正交化用的基准（rev20 / mom250 的同源构件），")
        L.append("用于检验财务因子的信号是否独立于「动量/反转」。\n")
        L.append("**波动估计量对比**：")
        vol_corr = df[["vol_close","vol_parkinson"]].dropna().corr().iloc[0,1]
        L.append(f"`vol_close` 与 `vol_parkinson` 的相关系数 = **{vol_corr:.4f}** "
                 f"（同源但不同效率，Parkinson 理论效率高 5 倍）\n")

    L.append("## 七、已知局限\n")
    L.append("1. 本项目未使用价格衍生特征作为信号，F2 主要为方法论确认")
    L.append("2. 对照因子仅用 2024 年数据，未做完整 IC 检验（那属于 F7）")
    L.append("3. 未实现 Yang-Zhang（需处理跳空，复杂度高）")
    L.append("4. 量价/流动性构件未实测（`amihud` 在 D2 已作为价量因子检验过）")
    return "\n".join(L)


def main():
    log("反向检查：技术指标是否被当机制 …")
    bad = check_no_technical_mechanism()
    log(f"  违规 {len(bad)} 个")

    log("加载 2024 年价格数据 …")
    df = load_price_sample()
    ic_res = {}
    if df is not None:
        log(f"  {len(df):,} 行")
        df = build_controls(df)
        for name, comp in [("mom_60", "动量"), ("rev_5_ma60", "反转")]:
            s = ic_by_date(df, name)
            ic_res[name] = (comp, s.mean() if len(s) else np.nan, len(s))
            log(f"  {name:14s} IC {ic_res[name][1]:+.4f}  期数 {len(s)}")
    else:
        log("  无数据，跳过实测")

    log("写出 …")
    (ROOT / "feature_price.md").write_text(render(df, ic_res, bad), encoding="utf-8")
    log("  → feature_price.md")


if __name__ == "__main__":
    main()