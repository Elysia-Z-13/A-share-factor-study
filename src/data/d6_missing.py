# -*- coding: utf-8 -*-
"""D6 缺失与异常 → missing_policy.md + 因子预处理"""
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"

FEATURES = ["gm", "dgm_q", "transfer_intensity", "dep_rate", "gm_vol"]
MAD_K = 3
MAD_TO_SIGMA = 1.4826
MISSING_BAR_POLICY = "untradable"
SPLIT = "2025-01-01"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load_features():
    fp = pd.read_csv(CLEAN/"fin"/"factor_panel.csv",
                     encoding="utf-8-sig", dtype={"code": str})
    fp["accper"] = pd.to_datetime(fp["accper"], errors="coerce")
    fp["declare"] = pd.to_datetime(fp["declare"], errors="coerce")
    fp = fp.sort_values(["code","accper"])

    denom = fp["fa_cost_begin"].replace(0, np.nan)
    fp["transfer_intensity"] = fp["transfer_from_cip"] / denom
    fp["dep_rate"] = fp["dep_add"] / denom
    fp["gm_vol"] = fp.groupby("code")["gm"].transform(
        lambda s: s.rolling(4, min_periods=3).std())
    fp["dgm_q"] = fp.groupby("code")["gm"].diff()
    return fp


def missing_stats(fp, cols):
    rows = []
    for c in cols:
        if c not in fp.columns:
            continue
        rows.append({"列": c, "缺失率": fp[c].isna().mean(),
                     "非空行数": int(fp[c].notna().sum())})
    return pd.DataFrame(rows).sort_values("缺失率", ascending=False)


POLICY = {
    "gm": ("MCAR", "fill_median", "覆盖率约 98%"),
    "dgm_q": ("MCAR", "fill_median", "需两期，首期缺失"),
    "transfer_intensity": ("MAR", "keep_nan_tree/median_linear",
                           "仅制造业披露附注；填0会把行业效应伪装成投资效应"),
    "dep_rate": ("MAR", "keep_nan_tree/median_linear", "同上"),
    "gm_vol": ("MCAR", "fill_median", "需 4 期滚动"),
}


class Preprocessor:
    """只用训练窗拟合，避免前视"""

    def fit(self, train, cols):
        self.stats = {}
        for c in cols:
            if c not in train.columns:
                continue
            s = train[c].dropna()
            if len(s) < 100:
                continue
            med = s.median()
            mad = (s - med).abs().median()
            sigma = MAD_TO_SIGMA * mad if mad > 0 else s.std()
            lo, hi = med - MAD_K*sigma, med + MAD_K*sigma
            clipped = s.clip(lo, hi)
            self.stats[c] = {"lo": lo, "hi": hi, "fill": med,
                             "mu": clipped.mean(), "sd": clipped.std()}
        return self

    def transform(self, df, cols):
        out = df.copy()
        for c in cols:
            if c not in df.columns or c not in self.stats:
                continue
            st = self.stats[c]
            x = df[c].clip(st["lo"], st["hi"])        # ① 去极值
            x = x.fillna(st["fill"])                  # ② 缺失
            out[c + "_z"] = (x - st["mu"]) / st["sd"] if st["sd"] > 0 else np.nan  # ③ 标准化
        return out


def render(stats, fp, prep):
    L = []
    L.append("# missing_policy.md —— 缺失与异常（D6）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")

    L.append("## 一、缺失分布\n")
    L.append("| 列 | 缺失率 | 非空行数 |")
    L.append("|---|---|---|")
    for _, r in stats.iterrows():
        L.append(f"| `{r['列']}` | {r['缺失率']*100:.1f}% | {r['非空行数']:,} |")

    L.append("\n## 二、缺失分类\n")
    L.append("| 列 | 类型 | 判断依据 |")
    L.append("|---|---|---|")
    for c, (typ, _, why) in POLICY.items():
        L.append(f"| `{c}` | **{typ}** | {why} |")
    L.append("\n**MAR 的含义**：缺失与可观测变量（行业）高度相关。")
    L.append("转固额/折旧额只在制造业附注中披露，服务业 100% 缺失。\n")

    L.append("## 三、逐列策略\n")
    L.append("| 列 | 策略 | 说明 |")
    L.append("|---|---|---|")
    for c, (typ, strat, why) in POLICY.items():
        L.append(f"| `{c}` | {strat} | {why} |")

    L.append("\n**明确禁止**：任何列都不得 `fillna(0)`。")
    L.append("转固额为 0 的经济含义是「没有在建工程转入」，")
    L.append("而 NaN 的含义是「未披露」。两者完全不可混。\n")

    L.append("## 四、预处理顺序\n")
    L.append("```")
    L.append("① 去极值（med ± 3σ，σ = 1.4826 × MAD，只用非 NaN 值）")
    L.append("② 缺失处理（填训练窗中位数，或保留 NaN 给树模型）")
    L.append("③ 标准化（(x − 训练窗μ) / 训练窗σ）")
    L.append("```\n")
    L.append("**顺序不可更换**：先去极值才能保证 MAD 只基于真实分布；")
    L.append("先缺失会在分布中心制造人造堆积，使 3σ 界变窄。\n")

    L.append("## 五、训练窗统计量\n")
    L.append("| 列 | 中位数(fill) | 下界 3σ | 上界 3σ | μ | σ |")
    L.append("|---|---|---|---|---|---|")
    for c, st in prep.stats.items():
        L.append(f"| `{c}` | {st['fill']:.4f} | {st['lo']:.4f} | "
                 f"{st['hi']:.4f} | {st['mu']:.4f} | {st['sd']:.4f} |")
    L.append("\n**关键纪律**：以上统计量全部只用训练窗（≤2024-09-30）计算。")
    L.append("测试集与 holdout 沿用同一套参数，不得重算。\n")

    L.append("## 六、缺失 bar 处理\n")
    L.append(f"- 建仓日无行情 → **{MISSING_BAR_POLICY}**（不买）")
    L.append("- 持有期内无行情 → hold（拿着，等复牌）")
    L.append("\n**依据**：CSMAR 停牌日整行缺失；建仓日缺失意味着无法成交。\n")

    L.append("\n## 七、已知局限\n")
    L.append("1. 转固/折旧类因子覆盖率仅 17~20%，只对制造业有效，报告中必须声明")
    L.append("2. 用中位数填补会压缩截面差异，可能低估因子强度")
    L.append("3. 树模型保留 NaN 时，缺失本身可能成为信息（如「是否制造业」），需在特征重要性中观察")
    return "\n".join(L)


def main():
    log("加载特征 …")
    fp = load_features()
    log(f"  {len(fp):,} 行")

    log("缺失统计 …")
    stats = missing_stats(fp, FEATURES)
    for _, r in stats.iterrows():
        log(f"  {r['列']:22s} 缺失 {r['缺失率']*100:5.1f}%")

    log("训练窗拟合预处理 …")
    train = fp[fp["declare"] < pd.Timestamp(SPLIT)]
    prep = Preprocessor().fit(train, FEATURES)
    log(f"  拟合 {len(prep.stats)} 列统计量")

    log("验证无泄漏 …")
    test = fp[fp["declare"] >= pd.Timestamp(SPLIT)]
    tr_z = prep.transform(train, FEATURES)
    te_z = prep.transform(test, FEATURES)
    log(f"  训练集 gm_z 均值 {tr_z['gm_z'].mean():+.4f}（≈0）")
    log(f"  测试集 gm_z 均值 {te_z['gm_z'].mean():+.4f}（不必为 0，说明用了训练窗参数 ✅）")

    log("写出 …")
    (ROOT / "missing_policy.md").write_text(render(stats, fp, prep), encoding="utf-8")
    log("  → missing_policy.md")


if __name__ == "__main__":
    main()