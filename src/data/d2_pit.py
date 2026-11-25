# -*- coding: utf-8 -*-
"""
D2 时点正确：两个对照实验

实验一：财报时点泄漏的代价
  A 组（正确）：announce_date <= T 取最新一期
  B 组（错误）：end_date <= T 取最新一期   ← 相当于报告期一结束就可见
  比较同一个因子（毛利率 gm）的 rank IC

实验二：幸存者偏差的代价
  A 组（含退市）：全历史所有股票
  B 组（不含退市）：只保留「数据最后一天仍在交易」的股票
  比较等权组合的年化收益
"""
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

H = 60
MONTH_STEP = 20          # 每 20 个交易日取一个截面


def load_prices():
    parts = []
    for f in sorted((CLEAN / "quotes").glob("quote_*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                        usecols=["code", "date", "close", "close_adj"])
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["fwd"] = df.groupby("code")["close_adj"].shift(-H)
    df["label"] = df["fwd"] / df["close_adj"] - 1
    return df


def load_fin():
    fp = pd.read_csv(CLEAN / "fin" / "factor_panel.csv",
                     encoding="utf-8-sig", dtype={"code": str})
    fp["accper"] = pd.to_datetime(fp["accper"], errors="coerce")
    fp["declare"] = pd.to_datetime(fp["declare"], errors="coerce")
    fp = fp.dropna(subset=["accper"])
    return fp[["code", "accper", "declare", "gm"]].copy()


def merge_pit(daily, fin, key):
    """key='declare' 为正确时点；key='accper' 为错误时点（前视）"""
    out = []
    f = fin.dropna(subset=[key]).sort_values(key)
    for code, g in daily.groupby("code", sort=False):
        ff = f[f["code"] == code]
        if ff.empty:
            continue
        gg = g.sort_values("date")
        cols = [key, "gm"] if key == "accper" else [key, "accper", "gm"]
        m = pd.merge_asof(gg, ff[cols],
                          left_on="date", right_on=key,
                          direction="backward")
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else daily.assign(gm=np.nan)


def spear(x, y):
    """Spearman = 秩后的 Pearson（无 scipy 时自己实现）"""
    rx = pd.Series(x).rank(method="average").values
    ry = pd.Series(y).rank(method="average").values
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float((rx*ry).sum()/den) if den > 0 else np.nan


def ic_of(df, factor="gm", label="label"):
    ics, dates = [], []
    for d, g in df.groupby("date"):
        s = g[[factor, label]].dropna()
        if len(s) < 100:
            continue
        ic = spear(s[factor].values, s[label].values)
        if np.isfinite(ic):
            ics.append(ic)
            dates.append(d)
    s = pd.Series(ics, index=pd.to_datetime(dates))
    return s


def main():
    log("加载行情 …")
    px = load_prices()
    log(f"  {len(px):,} 行")
    log("加载财报 …")
    fin = load_fin()
    log(f"  {len(fin):,} 行")

    # 取月末截面（每 20 个交易日）
    all_dates = np.array(sorted(px["date"].unique()))
    sample_dates = set(all_dates[::MONTH_STEP])

    # 训练期（不含 holdout）
    px_tr = px[(px["date"] < pd.Timestamp("2025-01-01")) &
               (px["date"].isin(sample_dates))].copy()
    log(f"训练期截面日 {px_tr['date'].nunique()} 个，{len(px_tr):,} 行")

    # ── 实验一：时点泄漏 ──
    log("=" * 60)
    log("实验一：财报时点泄漏的代价")
    a = merge_pit(px_tr, fin, "declare")     # 正确
    b = merge_pit(px_tr, fin, "accper")      # 错误（报告期即可见）
    ic_a = ic_of(a)
    ic_b = ic_of(b)
    log(f"  A 组（正确，announce<=T）：{len(ic_a)} 期  IC均值 {ic_a.mean():+.5f}  t={ic_a.mean()/ic_a.std()*np.sqrt(len(ic_a)):+.2f}")
    log(f"  B 组（错误，accper<=T）  ：{len(ic_b)} 期  IC均值 {ic_b.mean():+.5f}  t={ic_b.mean()/ic_b.std()*np.sqrt(len(ic_b)):+.2f}")
    if abs(ic_a.mean()) > 0:
        log(f"  B/A 倍数 = {abs(ic_b.mean())/abs(ic_a.mean()):.2f}x")
    # 覆盖差异
    log(f"  财报覆盖：正确组 {a['gm'].notna().mean()*100:.1f}% / 错误组 {b['gm'].notna().mean()*100:.1f}%")

    # ── 实验二：幸存者偏差 ──
    log("=" * 60)
    log("实验二：幸存者偏差的代价")
    last_day = px["date"].max()
    alive = px.groupby("code")["date"].max() >= (last_day - pd.Timedelta(days=10))
    alive_codes = set(alive[alive].index)
    log(f"  全样本 {px['code'].nunique():,} 只；最后一天仍在交易 {len(alive_codes):,} 只")
    log(f"  退市/消失 {px['code'].nunique()-len(alive_codes):,} 只")

    def ew_return(sub):
        """等权组合：每 20 日调仓，持有 60 日（用重叠近似）"""
        r = sub.dropna(subset=["label"])
        r = r[r["date"].isin(sample_dates)]
        by_date = r.groupby("date")["label"].mean()
        # 每个截面持有 60 日，年化 = 平均 * (252/60)
        return by_date.mean(), by_date.mean() * (252 / H), len(by_date)

    full_m, full_a, n1 = ew_return(px)
    surv = px[px["code"].isin(alive_codes)]
    surv_m, surv_a, n2 = ew_return(surv)
    log(f"  A 组（含退市）  ：{n1} 期  每期 {full_m*100:+.3f}%  年化 {full_a*100:+.2f}%")
    log(f"  B 组（不含退市）：{n2} 期  每期 {surv_m*100:+.3f}%  年化 {surv_a*100:+.2f}%")
    log(f"  幸存者偏差高估 = {(surv_a - full_a)*100:+.2f}% 年化")

    # ── 结论 ──
    log("=" * 60)
    log("结论")
    log(f"  实验一：错用时点使 |IC| 从 {abs(ic_a.mean()):.5f} 变为 {abs(ic_b.mean()):.5f}"
        f"（{abs(ic_b.mean())/max(abs(ic_a.mean()),1e-9):.2f}x）")
    log(f"  实验二：剔除退市股使年化收益虚高 {(surv_a-full_a)*100:+.2f}%")


if __name__ == "__main__":
    main()
