# -*- coding: utf-8 -*-
"""
CSMAR 数据清洗总脚本（Test 项目）

输入：DATA/ 下的 CSMAR 原始文件（UTF-8 with BOM）
输出：CLEAN/
    quotes/quote_YYYY.parquet       日频行情（清洗+衍生字段）
    quotes/calendar.csv             交易日历
    fin/bs.parquet                  资产负债表（精简字段）
    fin/inc.parquet                 利润表（精简字段）
    fin/cf.parquet                  现金流量表（直接法，精简字段）
    fin/fa_detail.parquet           固定资产明细（按类别）
    fin/cip_detail.parquet          在建工程明细
    meta/company.csv                公司基本信息（行业）
    meta/universe.parquet           (code,date) 网格 + universe/mask 标记
    audit_report.md                 质量报告

用法：python clean_all.py
"""
import os, sys, glob, re, json, time
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────── 配置 ───────────────────────────
# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
RAW  = ROOT / "DATA"
OUT  = ROOT / "CLEAN"
for sub in ["quotes", "fin", "meta"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

ENCODING = "utf-8-sig"          # 实测：CSMAR 导出为 UTF-8 with BOM
TOL_LIMIT   = 0.005             # 涨跌停容差（半分钱）
EXTREME_RET = 0.5               # 极端收益阈值
GEM_REFORM  = pd.Timestamp("2020-08-24")

QUOTE_FILES = sorted(glob.glob(str(RAW / "日个股回报率文件*" / "TRD_Dalyr*.csv")))

# 注意：CSMAR 目录名带后缀（如「资产负债表225954392(仅供…使用)」），必须用通配符
FIN_FILES = {
    "bs":  sorted(glob.glob(str(RAW / "资产负债表*"      / "FS_Combas*.csv"))),
    "inc": sorted(glob.glob(str(RAW / "利润表*"          / "FS_Comins*.csv"))),
    "cf":  sorted(glob.glob(str(RAW / "现金流量表(直接法)*" / "FS_Comscfd*.csv"))),
    "cfs": sorted(glob.glob(str(RAW / "现金流量表(间接法)*" / "FS_Comscfi*.csv"))),
    "fa":  sorted(glob.glob(str(RAW / "固定资产、累计折旧*" / "FN_Fn020*.csv"))),
    "cip": sorted(glob.glob(str(RAW / "在建工程*"        / "FN_FN017A*.csv"))),
}
for _k, _v in FIN_FILES.items():
    print(f"[glob] {_k}: {len(_v)} 个文件")
    for _f in _v:
        print("        ", _f)
COMPANY_FILE = sorted(glob.glob(str(RAW / "上市公司基本信息年度表*" / "*.csv")))

# 精简保留的财务字段（键 + 少数已验证/待验证的科目）
# ⚠️ 标 PROVISIONAL 的必须用公司年报核对后才能用于研究
BS_COLS = ["Stkcd", "Accper", "Typrep", "IfCorrect", "DeclareDate",
           "A001000000",   # 资产总计（PROVISIONAL）
           "A002000000",   # 负债合计（PROVISIONAL）
           "A003000000"]   # 所有者权益合计（PROVISIONAL）
INC_COLS = ["Stkcd", "Accper", "Typrep", "IfCorrect", "DeclareDate",
            "B001100000",   # 营业收入（PROVISIONAL）
            "B001101000",   # 营业成本（PROVISIONAL）
            "B001200000",   # 营业利润（PROVISIONAL）
            "B002000000"]   # 净利润（PROVISIONAL）
CF_COLS  = ["Stkcd", "Accper", "Typrep", "IfCorrect", "DeclareDate",
            "C001000000"]   # 经营活动现金流净额（PROVISIONAL）

RENAME_COMMON = {
    "Stkcd": "code", "Symbol": "code", "stkcd": "code",
    "Accper": "accper", "accper": "accper",
    "Typrep": "typrep", "typrep": "typrep",
    "IfCorrect": "if_correct", "DeclareDate": "declare",
    "WriteOff": "if_correct",
}

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def save(df, path, index=False):
    """pyarrow 不可用，统一写 CSV（浮点压缩到 6 位有效数字省空间）"""
    p = str(path).replace(".parquet", ".csv")
    df.to_csv(p, index=index, encoding="utf-8-sig", float_format="%.6g")
    return p


# ─────────────────────── 1. 日频行情 ───────────────────────
QUOTE_MAP = {
    "Stkcd": "code", "Trddt": "date",
    "Opnprc": "open", "Hiprc": "high", "Loprc": "low", "Clsprc": "close",
    "Adjprcwd": "close_adj", "Dretwd": "ret_total", "Dretnd": "ret_price",
    "Dnshrtrd": "volume", "Dnvaltrd": "amount",
    "Dsmvosd": "float_mv", "Dsmvtll": "total_mv",
    "Markettype": "market", "Trdsta": "trade_status",
}
QUOTE_NUM = ["open", "high", "low", "close", "close_adj", "volume", "amount",
             "float_mv", "total_mv", "ret_total", "ret_price", "trade_status"]


def load_quotes():
    log(f"读取行情 {len(QUOTE_FILES)} 个分卷 …")
    parts = []
    for f in QUOTE_FILES:
        d = pd.read_csv(f, dtype={"Stkcd": str}, encoding=ENCODING)
        d = d.rename(columns=QUOTE_MAP)
        parts.append(d[list(QUOTE_MAP.values())])
        log(f"  {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}  {len(d):,} 行")
    df = pd.concat(parts, ignore_index=True)
    del parts

    df["code"] = df["code"].str.strip().str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in QUOTE_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["trade_status"] = pd.to_numeric(df["trade_status"], errors="coerce").astype("Int16")
    df["market"] = pd.to_numeric(df["market"], errors="coerce").astype("Int8")

    df = df.dropna(subset=["code", "date", "close"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    log(f"合并后 {len(df):,} 行 / {df['code'].nunique():,} 只 / {df['date'].nunique():,} 个交易日")
    return df


def derive_quotes(df):
    log("衍生字段：复权因子 / 上市首日 / 极端收益 / 涨跌停 …")

    # ① 复权因子
    df["adj_factor"] = df["close_adj"] / df["close"].replace(0, np.nan)

    # ② 上市首日 & 最后一日
    first_day = df.groupby("code")["date"].transform("min")
    last_day  = df.groupby("code")["date"].transform("max")
    df["is_first_row"] = df["date"] == first_day
    df["is_last_row"]  = df["date"] == last_day

    # ③ 极端收益分类
    df["is_extreme"] = df["ret_total"].abs() > EXTREME_RET
    df["extreme_kind"] = ""
    df.loc[df["is_extreme"] & df["is_first_row"], "extreme_kind"] = "ipo_or_first"
    df.loc[df["is_extreme"] & (~df["is_first_row"]), "extreme_kind"] = "to_check"

    # ④ 涨跌停幅度（按板块 + ST + 改革日）
    codes = df["code"].astype("string")
    rate = pd.Series(0.10, index=df.index, dtype="float32")
    rate[codes.str.startswith("688").fillna(False)] = 0.20
    gem = codes.str.startswith("300").fillna(False) | codes.str.startswith("301").fillna(False)
    rate[gem & (df["date"] >= GEM_REFORM)] = 0.20
    rate[gem & (df["date"] < GEM_REFORM)]  = 0.10
    bj = codes.str.startswith("8").fillna(False) | codes.str.startswith("4").fillna(False)
    rate[bj] = 0.30
    rate[df["trade_status"].isin([2, 3])] = 0.05   # 疑似 ST/*ST（待字典确认）
    df["limit_rate"] = rate

    # ⑤ 涨跌停价（必须用未复权 close，且前收盘用停牌前最后一天）
    prev = df.groupby("code")["close"].shift(1)
    df["prev_close"] = prev
    df["limit_up"]   = (prev * (1 + df["limit_rate"])).round(2)
    df["limit_down"] = (prev * (1 - df["limit_rate"])).round(2)
    df["hit_limit_up"]   = (df["close"] - df["limit_up"]).abs()   < TOL_LIMIT
    df["hit_limit_down"] = (df["close"] - df["limit_down"]).abs() < TOL_LIMIT
    df.loc[prev.isna(), ["hit_limit_up", "hit_limit_down"]] = False

    # ⑥ 复权收益对账
    r_calc = df.groupby("code")["close_adj"].pct_change()
    df["adj_ret_err"] = (r_calc - df["ret_total"]).abs()

    return df


def audit_quotes(df):
    log("行情质量检查 …")
    r = {}
    r["rows"]  = int(len(df))
    r["codes"] = int(df["code"].nunique())
    r["days"]  = int(df["date"].nunique())
    r["date_min"] = str(df["date"].min().date())
    r["date_max"] = str(df["date"].max().date())
    r["dup_keys"] = int(df.duplicated(["code", "date"]).sum())

    r["ohlc_bad"] = int((~(
        (df["high"] >= df[["open", "close", "low"]].max(axis=1)) &
        (df["low"]  <= df[["open", "close", "high"]].min(axis=1)) &
        (df["high"] >= df["low"])
    )).sum())

    vwap = df["amount"] / df["volume"].replace(0, np.nan)
    r["vwap_ok"] = float(((vwap >= df["low"]) & (vwap <= df["high"])).mean())

    r["adj_err_p50"] = float(df["adj_ret_err"].quantile(0.50))
    r["adj_err_p99"] = float(df["adj_ret_err"].quantile(0.99))
    r["adj_err_gt_1e4"] = int((df["adj_ret_err"] > 1e-4).sum())

    r["extreme_rows"] = int(df["is_extreme"].sum())
    r["extreme_ipo"]  = int((df["extreme_kind"] == "ipo_or_first").sum())
    r["extreme_check"] = int((df["extreme_kind"] == "to_check").sum())

    r["bad_price"] = int((df["close"] <= 0).sum())
    r["mv_null"] = float(df["float_mv"].isna().mean())
    r["limit_up_days"] = int(df["hit_limit_up"].sum())
    r["limit_down_days"] = int(df["hit_limit_down"].sum())
    r["limit_up_rate"] = float(df["hit_limit_up"].mean())

    cov = df.groupby(df["date"].dt.year)["code"].nunique()
    r["coverage"] = {int(k): int(v) for k, v in cov.items()}
    chg = cov / cov.shift(1) - 1
    r["coverage_worst"] = float(chg.min())

    r["trade_status"] = {str(k): int(v) for k, v in df["trade_status"].value_counts().items()}
    return r


# ─────────────────────── 2. 财务数据 ───────────────────────
def load_fin(paths, cols, name):
    """读一个财务文件，只保留 cols（不足的列跳过），统一类型"""
    if not paths:
        raise FileNotFoundError(f"{name}: 没有匹配到任何 CSV，请检查 RAW 目录名")
    frames = []
    for f in paths:
        header = pd.read_csv(f, nrows=0, encoding=ENCODING).columns.tolist()
        use = [c for c in cols if c in header]
        missing = [c for c in cols if c not in header]
        d = pd.read_csv(f, encoding=ENCODING, usecols=use, dtype={"Stkcd": str})
        for m in missing:
            d[m] = np.nan
        frames.append(d)
        log(f"  {name}  {os.path.basename(f)}  {len(d):,} 行  保留 {len(use)}/{len(header)} 列")
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns=RENAME_COMMON)
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["accper"] = pd.to_datetime(df["accper"], errors="coerce")
    df["declare"] = pd.to_datetime(df["declare"], errors="coerce")

    for c in cols:
        cc = RENAME_COMMON.get(c, c)
        if cc in df.columns and cc not in ("code", "accper", "declare", "typrep", "if_correct"):
            df[cc] = pd.to_numeric(df[cc], errors="coerce")

    df = df[df["code"].str.match(r"^\d{6}$", na=False)]
    # 只留合并报表
    if "typrep" in df.columns:
        before = len(df)
        df = df[df["typrep"].astype(str).str.upper().str.startswith("A")]
        log(f"  {name}  合并报表过滤：{before:,} → {len(df):,}")
    # 更正版：每个 (code, accper) 留最新公告
    df = df.sort_values(["code", "accper", "declare"])
    df = df.groupby(["code", "accper"], dropna=False).tail(1).reset_index(drop=True)
    df = df.dropna(subset=["accper"])
    return df


def load_detail(paths, name):
    """明细表（固定资产/在建工程）：按项目分行，字段名大小写不一"""
    frames = []
    for f in paths:
        d = pd.read_csv(f, encoding=ENCODING, dtype={"Stkcd": str, "stkcd": str})
        d.columns = [c.lower() for c in d.columns]
        frames.append(d)
        log(f"  {name}  {os.path.basename(f)}  {len(d):,} 行")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"stkcd": "code", "accper": "accper",
                            "typrep": "typrep", "datasources": "data_sources"})
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)
    df["accper"] = pd.to_datetime(df["accper"], errors="coerce")
    num_cols = [c for c in df.columns if re.fullmatch(r"fn_?fn?0?17a?\d+", c) or re.fullmatch(r"fn0\d+", c)]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["code"].str.match(r"^\d{6}$", na=False)]
    df = df.dropna(subset=["accper"]).reset_index(drop=True)
    return df


# ─────────────────────── 3. 公司信息 ───────────────────────
def load_company():
    if not COMPANY_FILE:
        return None
    f = COMPANY_FILE[0]
    d = pd.read_csv(f, dtype={"Symbol": str}, encoding=ENCODING)
    d["code"] = d["Symbol"].str.zfill(6)
    d["end_date"] = pd.to_datetime(d["EndDate"], errors="coerce")
    d = d[d["code"].str.match(r"^\d{6}$", na=False)]
    keep = ["code", "ShortName", "end_date", "IndustryName", "IndustryCode",
            "IndustryNameC", "IndustryCodeC"]
    d = d[[c for c in keep if c in d.columns]]
    d = d.sort_values(["code", "end_date"]).drop_duplicates("code", keep="last")
    log(f"  公司信息  {len(d):,} 只")
    return d


# ─────────────────────── 4. Universe ───────────────────────
def build_universe(df, company, start_year=2010):
    """
    生成 (code, date) 全网格并标注：
      not_listed / delisted / suspended / tradable
    上市日与退市日由行情首末行推导（数据从 2010 起，早于此的股票首行即 2010-01-04）
    """
    log("构建 universe 网格 …")
    first = df.groupby("code")["date"].min().rename("first_date")
    last  = df.groupby("code")["date"].max().rename("last_date")
    life = pd.concat([first, last], axis=1).reset_index()

    calendar = pd.DatetimeIndex(sorted(df["date"].unique()))
    cal_by_year = {y: calendar[calendar.year == y] for y in range(start_year, calendar.max().year + 1)}

    out = []
    for year, cal in cal_by_year.items():
        sub = df[(df["date"] >= cal[0]) & (df["date"] <= cal[-1])]
        codes_y = sub["code"].unique()
        grid = pd.MultiIndex.from_product([codes_y, cal], names=["code", "date"])
        g = sub.set_index(["code", "date"]).reindex(grid)

        for c in ["close", "volume", "limit_up", "hit_limit_up", "ret_total"]:
            if c not in g.columns:
                g[c] = np.nan
        g = g.reset_index()
        g = g.merge(life, on="code", how="left")
        g["not_listed"] = g["date"] < g["first_date"]
        g["delisted"]   = g["date"] > g["last_date"]
        g["suspended"]  = (~g["not_listed"]) & (~g["delisted"]) & (g["close"].isna())
        g["tradable"]   = (~g["not_listed"]) & (~g["delisted"]) & (~g["suspended"])
        out.append(g[["code", "date", "first_date", "last_date",
                      "not_listed", "delisted", "suspended", "tradable",
                      "hit_limit_up"]])
        log(f"  {year}: {len(g):,} 行  可交易 {int(g['tradable'].sum()):,}")

    uni = pd.concat(out, ignore_index=True)
    if company is not None:
        uni = uni.merge(company[["code", "IndustryCodeC", "IndustryNameC"]], on="code", how="left")
    return uni


# ─────────────────────── 5. 主流程 ───────────────────────
def main():
    log("=" * 60)
    log("开始清洗")
    log("=" * 60)

    # 行情
    q = load_quotes()
    q = derive_quotes(q)
    rep = audit_quotes(q)

    log("按年写出行情 …")
    for y, g in q.groupby(q["date"].dt.year):
        p = save(g.reset_index(drop=True), OUT / "quotes" / f"quote_{y}.parquet")
        log(f"  {y}: {len(g):,} 行 → {os.path.basename(p)}")
    cal = pd.DataFrame({"date": sorted(q["date"].unique())})
    save(cal, OUT / "quotes" / "calendar.parquet")

    # 财务
    log("读取财务数据 …")
    bs  = load_fin(FIN_FILES["bs"],  BS_COLS,  "资产负债表")
    inc = load_fin(FIN_FILES["inc"], INC_COLS, "利润表")
    cf  = load_fin(FIN_FILES["cf"],  CF_COLS,  "现金流量表")
    fa  = load_detail(FIN_FILES["fa"],  "固定资产明细")
    cip = load_detail(FIN_FILES["cip"], "在建工程明细")

    for name, d in [("bs", bs), ("inc", inc), ("cf", cf), ("fa", fa), ("cip", cip)]:
        if d is not None:
            p = save(d, OUT / "fin" / f"{name}.parquet")
            log(f"  {name}: {len(d):,} 行 → {os.path.basename(p)}")

    # 公司信息
    company = load_company()
    if company is not None:
        save(company, OUT / "meta" / "company.parquet")

    # Universe
    uni = build_universe(q, company)
    save(uni, OUT / "meta" / "universe.parquet")
    log(f"  universe: {len(uni):,} 行")

    # ── 报告 ──
    log("生成质量报告 …")
    lines = []
    lines.append("# CSMAR 数据清洗报告\n")
    lines.append(f"> 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")
    lines.append("## 一、行情数据\n")
    lines.append("| 项目 | 数值 |")
    lines.append("|---|---|")
    for k, v in [
        ("行数", f"{rep['rows']:,}"),
        ("股票数", f"{rep['codes']:,}"),
        ("交易日数", f"{rep['days']:,}"),
        ("时间范围", f"{rep['date_min']} ~ {rep['date_max']}"),
        ("重复主键", rep["dup_keys"]),
        ("OHLC 违规", rep["ohlc_bad"]),
        ("VWAP 在高低之间", f"{rep['vwap_ok']*100:.3f}%"),
        ("复权误差 P50", f"{rep['adj_err_p50']:.2e}"),
        ("复权误差 P99", f"{rep['adj_err_p99']:.2e}"),
        ("复权误差 >1e-4 行数", f"{rep['adj_err_gt_1e4']:,}"),
        ("极端收益（|r|>50%）", f"{rep['extreme_rows']:,}"),
        ("　其中首行/新股", f"{rep['extreme_ipo']:,}"),
        ("　待人工核查", f"{rep['extreme_check']:,}"),
        ("非正收盘价", rep["bad_price"]),
        ("流通市值缺失率", f"{rep['mv_null']*100:.2f}%"),
        ("涨停日数", f"{rep['limit_up_days']:,}"),
        ("跌停日数", f"{rep['limit_down_days']:,}"),
        ("涨停占全样本", f"{rep['limit_up_rate']*100:.3f}%"),
        ("覆盖率最差年跌幅", f"{rep['coverage_worst']*100:.2f}%"),
    ]:
        lines.append(f"| {k} | {v} |")

    lines.append("\n### 每年股票数\n")
    lines.append("| 年份 | 股票数 |")
    lines.append("|---|---|")
    for y, v in rep["coverage"].items():
        lines.append(f"| {y} | {v:,} |")

    lines.append("\n### 交易状态分布（Trdsta）\n")
    lines.append("| 值 | 行数 |")
    lines.append("|---|---|")
    for k, v in sorted(rep["trade_status"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v:,} |")

    lines.append("\n## 二、财务数据\n")
    lines.append("| 表 | 行数 |")
    lines.append("|---|---|")
    for name, d in [("资产负债表", bs), ("利润表", inc), ("现金流量表", cf),
                    ("固定资产明细", fa), ("在建工程明细", cip)]:
        lines.append(f"| {name} | {0 if d is None else len(d):,} |")

    lines.append("\n## 三、待确认事项\n")
    lines.append("1. **Trdsta 语义**：2/3 是否 ST/*ST 需查 CSMAR 字段字典（当前按 5% 涨跌幅处理）")
    lines.append("2. **财务科目代码**：A00*/B00* 的语义映射为 PROVISIONAL，需用年报核对")
    lines.append("3. **上市/退市日**：由行情首末行推导，早于 2010 年的上市日会被截断")
    lines.append("4. **固定资产/在建工程字段**：Fn0200x / FN_Fn017Ax 的含义需查明细表字段说明")

    report = OUT / "audit_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"报告 → {report}")

    log("=" * 60)
    log("完成")
    log("=" * 60)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
