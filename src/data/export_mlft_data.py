# -*- coding: utf-8 -*-
"""
把 CLEAN/ 的清洗结果导出为 MLFT 执行器识别的文件名，写入 Test/data/

执行器识别的文件名（见 exec.js）：
  prices_YYYY.csv         code,date,open,high,low,close(复权),close_raw(原始),volume,amount,float_mv
  limit_YYYY.csv          code,date,limit_up,limit_down
  st_YYYY.csv             code,date,st_flag
  trade_status_YYYY.csv   code,date,trade_status
  listing.csv             code,list_date,delist_date
  sector_YYYY.csv         code,industry_code
  cost_params.json
"""
import os
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ── 1) 行情 ──
log("导出 prices_YYYY.csv …")
for y in range(2010, 2027):
    f = CLEAN / "quotes" / f"quote_{y}.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                    usecols=["code","date","open","high","low","close","close_adj",
                             "volume","amount","float_mv","limit_up","limit_down",
                             "trade_status"])
    out = pd.DataFrame({
        "code": d["code"],
        "date": d["date"],
        "open": d["open"], "high": d["high"], "low": d["low"],
        "close": d["close_adj"],          # 复权
        "close_raw": d["close"],          # 未复权
        "volume": d["volume"], "amount": d["amount"],
        "float_mv": d["float_mv"],
    })
    out.to_csv(DATA / f"prices_{y}.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
    log(f"  prices_{y}.csv  {len(out):,} 行")

    # limit
    lim = d[["code","date","limit_up","limit_down"]].copy()
    lim.to_csv(DATA / f"limit_{y}.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # st（trade_status 2/3/6 = ST/*ST）
    st = pd.DataFrame({
        "code": d["code"], "date": d["date"],
        "st_flag": d["trade_status"].isin([2, 3, 6]).astype(int),
    })
    st.to_csv(DATA / f"st_{y}.csv", index=False, encoding="utf-8-sig")

    # trade_status
    ts = d[["code","date","trade_status"]].copy()
    ts.to_csv(DATA / f"trade_status_{y}.csv", index=False, encoding="utf-8-sig")

# ── 2) 上市/退市 ──
log("导出 listing.csv …")
uni = pd.read_csv(CLEAN / "meta" / "universe.csv", encoding="utf-8-sig", dtype={"code": str},
                  usecols=["code","date","first_date","last_date"])
life = uni.groupby("code").agg(first_date=("first_date","min"), last_date=("last_date","max")).reset_index()
cal_max = pd.to_datetime(uni["date"]).max()
life["list_date"] = pd.to_datetime(life["first_date"]).dt.strftime("%Y%m%d")
# 最后一天 = 数据最后一天 → 说明仍上市
alive = pd.to_datetime(life["last_date"]) >= cal_max - pd.Timedelta(days=10)
life["delist_date"] = np.where(alive, "", pd.to_datetime(life["last_date"]).dt.strftime("%Y%m%d"))
life[["code","list_date","delist_date"]].to_csv(DATA / "listing.csv", index=False, encoding="utf-8-sig")
log(f"  listing.csv  {len(life):,} 只（已退市 {int((~alive).sum()):,}）")

# ── 3) 行业快照 ──
log("导出 sector_YYYY.csv …")
comp = CLEAN / "meta" / "company.csv"
if comp.exists():
    c = pd.read_csv(comp, encoding="utf-8-sig", dtype={"code": str})
    ind_col = "IndustryCodeC" if "IndustryCodeC" in c.columns else c.columns[-1]
    for y in range(2010, 2027):
        sec = c[["code", ind_col]].rename(columns={ind_col: "industry_code"})
        sec["date"] = f"{y}-12-31"
        sec = sec[["code","date","industry_code"]]
        sec.to_csv(DATA / f"sector_{y}.csv", index=False, encoding="utf-8-sig")
    log(f"  sector_2010..2026.csv  每份 {len(c):,} 行")

# ── 4) 成本参数 ──
log("导出 cost_params.json …")
cost = {
    "commission_bp": 2.5,
    "stamp_duty_bp_sell": 5.0,
    "transfer_fee_bp": 0.1,
    "slippage_bp": 2.4,
    "one_way_bp": 10.0,
    "notes": "A 股通行假设；佣金万分之2.5、印花税卖出千分之1、过户费万分之0.1、滑点含在合计10bp内。待用真实成交回单校准。",
    "sensitivity_bp": [5, 10, 15, 20, 30],
}
(DATA / "cost_params.json").write_text(json.dumps(cost, ensure_ascii=False, indent=2), encoding="utf-8")

log("完成")
log(f"输出目录：{DATA}")
