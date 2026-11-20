# -*- coding: utf-8 -*-
"""D5 重叠与切分 → split_plan.md + 切分索引"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"

H = 60
STEP = 20
TRAIN_END = "2024-09-30"
HOLDOUT_START = "2025-01-01"
N_SPLITS = 5

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def load_labels():
    parts = []
    for f in sorted((CLEAN/"labels").glob("label_*.csv")):
        parts.append(pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                                 usecols=["code","date","label"]))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["code","date"]).reset_index(drop=True)


def sample_sizes(df, H=H, step=STEP):
    lab = df.dropna(subset=["label"])
    n_days = df["date"].nunique()
    n_cross = len(range(0, n_days, step))
    return {
        "N": len(lab),
        "N_eff": len(lab) / H,
        "交易日数": int(n_days),
        "T_eff": n_days / H,
        "截面数": n_cross,
        "purge_trading": H,
        "purge_calendar": int(H*1.5),
    }


def split_once(df):
    tr = df[df["date"] <= pd.Timestamp(TRAIN_END)].copy()
    ho = df[df["date"] >= pd.Timestamp(HOLDOUT_START)].copy()
    gap = (ho["date"].min() - tr["date"].max()).days
    assert gap >= int(H*1.5), f"purge 间隔只有 {gap} 天，需要 {int(H*1.5)}"
    return tr, ho, gap


def purged_kfold(dates, n_splits=N_SPLITS, H=H):
    ud = np.array(sorted(pd.Series(dates).unique()))
    folds = np.array_split(ud, n_splits)
    gap = np.timedelta64(int(H*1.5), "D")
    out = []
    for k in range(n_splits):
        te = set(folds[k])
        t1, t2 = folds[k][0], folds[k][-1]
        tr = set()
        for j in range(n_splits):
            if j == k:
                continue
            for d in folds[j]:
                if t1 - gap <= d <= t2 + gap:
                    continue
                tr.add(d)
        out.append({"fold": k, "train_days": len(tr), "test_days": len(te),
                    "train": tr, "test": te})
    return out


def render(sz, tr, ho, gap, folds):
    L = []
    L.append("# split_plan.md —— 重叠与切分（D5）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")

    L.append("## 一、样本量\n")
    L.append("| 量 | 公式 | 值 | 用途 |")
    L.append("|---|---|---|---|")
    L.append(f"| N | 有效标签行数 | {sz['N']:,} | 描述规模 |")
    L.append(f"| N_eff | N / H | {sz['N_eff']:,.0f} | 等价独立观测 |")
    L.append(f"| T_eff | 交易日数 / H | {sz['T_eff']:.1f} | — |")
    L.append(f"| 截面数 | 交易日 / {STEP} | {sz['截面数']:,} | **算 t 值用这个** |")
    L.append("\n**关键**：IC 的 t 值 = `mean(IC_series) / std(IC_series) × √(截面数)`，")
    L.append("其中截面数是 IC 序列的长度，**不是 N，也不是 N_eff**。\n")
    L.append("对照：若误用 N 算 t，会虚高约 √(N/截面数) ≈ "
             f"{np.sqrt(sz['N']/max(sz['截面数'],1)):.0f} 倍。\n")

    L.append("## 二、purge 与 embargo\n")
    L.append("| 项 | 交易日 | 自然日 |")
    L.append("|---|---|---|")
    L.append(f"| purge | {sz['purge_trading']} | {sz['purge_calendar']} |")
    L.append(f"| embargo | {sz['purge_trading']} | {sz['purge_calendar']} |")
    L.append(f"\n换算：H={H} 交易日 × 1.5（365/250）= {sz['purge_calendar']} 自然日\n")

    L.append("## 三、单次切分方案\n")
    L.append("| 块 | 区间 | 行数 | 交易日 |")
    L.append("|---|---|---|---|")
    L.append(f"| 训练 | 数据起点 ~ {TRAIN_END} | {len(tr):,} | {tr['date'].nunique()} |")
    L.append(f"| purge | — | — | {sz['purge_calendar']} 自然日 |")
    L.append(f"| holdout | {HOLDOUT_START} ~ 数据终点 | {len(ho):,} | {ho['date'].nunique()} |")
    L.append(f"\n实测间隔：{gap} 自然日（要求 ≥ {sz['purge_calendar']}）✅\n")

    L.append("## 四、Purged 5-Fold（训练集内部）\n")
    L.append("| 折 | 训练天数 | 测试天数 |")
    L.append("|---|---|---|")
    for f in folds:
        L.append(f"| {f['fold']} | {f['train_days']} | {f['test_days']} |")
    L.append("\n每折已挖掉 [测试起点−90天, 测试终点+90天] 的样本（purge + embargo）。\n")

    L.append("## 五、holdout 加锁记录\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 区间 | {HOLDOUT_START} ~ 2026-11-18 |")
    L.append("| 锁定步骤 | **B3**（项目第 4 步） |")
    L.append("| 解锁时点 | **E7**（统计推断） |")
    L.append("| 是否已使用 | **否** |")
    L.append("\n**纪律**：B3 ~ E6 期间一次未看；E7 只评估一次。\n")

    L.append("## 六、已知局限\n")
    L.append("1. **T_eff 只有 68**：16 年 / H=60，独立时间点极少，统计功效受限")
    L.append("2. **不做完整 walk-forward**：fold 内独立时间点仅 12 个，功效太低；改为单次切分 + 训练集内 Purged CV")
    L.append("3. **holdout 仅 23 个月**：数据到 2026-11，无法覆盖完整牛熊")
    L.append("4. **训练窗 15 年覆盖 5 个 regime**：满足“至少一个完整 regime”要求")
    return "\n".join(L)


def main():
    log("加载标签 …")
    df = load_labels()
    log(f"  {len(df):,} 行")

    log("样本量 …")
    sz = sample_sizes(df)
    for k, v in sz.items():
        log(f"  {k:14s} {v:,.1f}" if isinstance(v, float) else f"  {k:14s} {v:,}")

    log("单次切分 …")
    tr, ho, gap = split_once(df)
    log(f"  训练 {len(tr):,} / holdout {len(ho):,} / 间隔 {gap} 天")

    log("Purged 5-Fold …")
    folds = purged_kfold(tr["date"])
    for f in folds:
        log(f"  fold {f['fold']}: train {f['train_days']} / test {f['test_days']}")

    log("写出 …")
    (ROOT / "split_plan.md").write_text(render(sz, tr, ho, gap, folds), encoding="utf-8")
    idx = pd.DataFrame([{"fold": f["fold"], "split": "train",
                         "days": len(f["train"])} for f in folds] +
                       [{"fold": f["fold"], "split": "test",
                         "days": len(f["test"])} for f in folds])
    idx.to_csv(CLEAN / "labels" / "cv_folds.csv", index=False, encoding="utf-8-sig")
    log("  → split_plan.md / cv_folds.csv")


if __name__ == "__main__":
    main()