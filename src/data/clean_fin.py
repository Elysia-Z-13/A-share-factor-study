# -*- coding: utf-8 -*-
"""
财务数据清洗（修正版）
—— 科目代码已用已知公司年报反推验证

验证依据：
  贵州茅台 600519 / 2024-12-31
    营业总收入 B001100000 = 1741.44 亿
    营业收入   B001101000 = 1708.99 亿
    营业成本   B001201000 = 137.89 亿  → 毛利率 91.93%
    固定资产   A001212000 = 218.71 亿
    在建工程   A001213000 = 21.50 亿
    资产总计   A001000000 = 2989.45 亿
"""
import os, glob, time
from pathlib import Path
import numpy as np
import pandas as pd

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "DATA"
OUT = ROOT / "CLEAN" / "fin"
OUT.mkdir(parents=True, exist_ok=True)
ENC = "utf-8-sig"
t0 = time.time()

def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

def save(df, name):
    p = OUT / f"{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    log(f"  → {p.name}  {len(df):,} 行")
    return p

def find(pattern):
    return sorted(glob.glob(str(RAW / pattern)))

# 已核对的科目
BS_COLS = ["Stkcd", "Accper", "Typrep", "DeclareDate",
           "A001000000",   # 资产总计
           "A001200000",   # 非流动资产合计（待复核）
           "A001212000",   # 固定资产净额  ✅
           "A001213000",   # 在建工程      ✅
           "A002000000",   # 负债合计
           "A003000000"]   # 所有者权益合计
INC_COLS = ["Stkcd", "Accper", "Typrep", "DeclareDate",
            "B001100000",   # 营业总收入
            "B001101000",   # 营业收入      ✅
            "B001200000",   # 营业总成本
            "B001201000",   # 营业成本      ✅
            "B001200000",
            "B002000000"]   # 净利润

def load_std(paths, cols, name, typrep_filter="A"):
    if not paths:
        raise FileNotFoundError(name)
    frames = []
    for f in paths:
        header = pd.read_csv(f, nrows=0, encoding=ENC).columns.tolist()
        use = [c for c in dict.fromkeys(cols) if c in header]
        d = pd.read_csv(f, encoding=ENC, usecols=use, dtype={"Stkcd": str})
        frames.append(d)
        log(f"  {name}: {os.path.basename(f)}  {len(d):,} 行  列 {len(use)}/{len(header)}")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Stkcd": "code", "Accper": "accper",
                            "Typrep": "typrep", "DeclareDate": "declare"})
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["accper"] = pd.to_datetime(df["accper"], errors="coerce")
    if "declare" in df.columns:
        df["declare"] = pd.to_datetime(df["declare"], errors="coerce")
    else:
        df["declare"] = pd.NaT
    for c in df.columns:
        if c not in ("code", "accper", "typrep", "declare"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["code"].str.match(r"^\d{6}$", na=False)]
    if typrep_filter and "typrep" in df.columns:
        n0 = len(df)
        df = df[df["typrep"].astype(str).str.upper().str.startswith(typrep_filter)]
        log(f"  {name}: 合并报表 {n0:,} → {len(df):,}")
    # 更正版：每个 (code, accper) 只留最后公告的那份
    df = df.sort_values(["code", "accper", "declare"])
    df = df.groupby(["code", "accper"], dropna=False).tail(1).reset_index(drop=True)
    df = df.dropna(subset=["accper"])
    return df


def main():
    # 资产负债表
    bs = load_std(find("资产负债表*/FS_Combas*.csv"), BS_COLS, "资产负债表")

    # 利润表
    inc = load_std(find("利润表*/FS_Comins*.csv"), INC_COLS, "利润表")

    # 现金流量表（间接法）——折旧摊销藏在这里，整表保留
    cf_paths = find("现金流量表(间接法)*/FS_Comscfi*.csv")
    cf = load_std(cf_paths, None if False else pd.read_csv(cf_paths[0], nrows=0, encoding=ENC).columns.tolist()
                  if cf_paths else [], "现金流量表(间接法)")

    # 固定资产品类明细（折旧率备查）
    fa = load_std(find("固定资产、累计折旧*/FN_Fn020*.csv"),
                  None if False else pd.read_csv(find("固定资产、累计折旧*/FN_Fn020*.csv")[0],
                                                 nrows=0, encoding=ENC).columns.tolist(),
                  "固定资产明细", typrep_filter=None)

    # 在建工程明细
    cip_paths = find("在建工程*/FN_FN017A*.csv")
    cip_cols = pd.read_csv(cip_paths[0], nrows=0, encoding=ENC).columns.tolist()
    cip = pd.concat([pd.read_csv(f, encoding=ENC, dtype={"stkcd": str}) for f in cip_paths],
                    ignore_index=True)
    cip.columns = [c.lower() for c in cip.columns]
    cip = cip.rename(columns={"stkcd": "code", "accper": "accper", "typrep": "typrep"})
    cip["code"] = cip["code"].astype(str).str.zfill(6)
    cip["accper"] = pd.to_datetime(cip["accper"], errors="coerce")
    for c in cip.columns:
        if c.startswith("fn_fn017a"):
            cip[c] = pd.to_numeric(cip[c], errors="coerce")
    cip = cip.dropna(subset=["accper"])
    log(f"  在建工程明细: {len(cip):,} 行")

    # 派生：毛利率、净利率、单季还原
    log("派生财务指标 …")
    inc["revenue"] = inc.get("B001101000")
    inc["sales_total"] = inc.get("B001100000")
    inc["cogs"] = inc.get("B001201000")
    inc["gross_profit"] = inc["revenue"] - inc["cogs"]
    inc["gm"] = np.where(inc["revenue"].abs() > 0, inc["gross_profit"] / inc["revenue"], np.nan)
    inc["nm"] = np.where(inc["revenue"].abs() > 0, inc.get("B002000000") / inc["revenue"], np.nan)

    # 单季还原（利润表是累计值）
    inc = inc.sort_values(["code", "accper"])
    for c in ["revenue", "cogs", "gross_profit"]:
        inc[c + "_q"] = inc.groupby("code")[c].diff()
        q1 = inc["accper"].dt.month == 3
        inc.loc[q1, c + "_q"] = inc.loc[q1, c]

    bs = bs.rename(columns={"A001212000": "fa_net", "A001213000": "cip",
                            "A001000000": "total_asset", "A002000000": "total_liab",
                            "A003000000": "equity", "A001200000": "noncur_asset"})

    save(bs, "bs")
    save(inc, "inc")
    save(cf, "cf_indirect")
    save(fa, "fa_detail")
    save(cip, "cip_detail")

    # ── 校验 ──
    log("=" * 50)
    log("字段校验（对照年报）")
    for code, name, exp in [("600519", "贵州茅台", 91.9), ("000858", "五粮液", 75.0),
                            ("000002", "万科A", 10.0), ("600036", "招商银行", None)]:
        g = inc[(inc.code == code) & (inc.accper == pd.Timestamp("2024-12-31"))]
        if g.empty:
            log(f"  {code} {name}: 无 2024 年报"); continue
        r = g.iloc[0]
        gm = r["gm"] * 100
        ok = "" if exp is None else ("✅" if abs(gm - exp) < 8 else "❌")
        log(f"  {code} {name:6s} 营收 {r['revenue']/1e8:8.1f}亿  成本 {(r['cogs']/1e8 if pd.notna(r['cogs']) else float('nan')):8.1f}亿  毛利率 {gm:6.2f}%  {ok}")

    g = bs[(bs.code == "600519") & (bs.accper == pd.Timestamp("2024-12-31"))]
    if not g.empty:
        r = g.iloc[0]
        log(f"  600519 固定资产 {r['fa_net']/1e8:8.2f}亿（年报 218.71）| 在建工程 {r['cip']/1e8:6.2f}亿（年报 21.50）| 总资产 {r['total_asset']/1e8:8.2f}亿")

if __name__ == "__main__":
    main()
