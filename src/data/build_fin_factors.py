# -*- coding: utf-8 -*-
"""
从 FN_Fn020 明细表提取「转固额」与「折旧计提」，并构建季度因子面板。

字段字典（已核对）：
  Fn02001 [科目类型] 1=原值 2=累计折旧 3=净值 4=减值准备 5=净额
  Fn02002 [固定资产项目] 房屋及建筑物 / 机器设备 / ...
  Fn02003 [期初余额]
  Fn02004 [本期增加]
  Fn02005 [其中：增加子公司]
  Fn02006 [其中：购置增加]
  Fn02007 [其中：在建工程转入]   ← 转固额
  Fn02009 [本期减少]
  Fn02013 [期末余额]

派生因子：
  转固率   = Σ Fn02007(原值) / 在建工程期初余额
  折旧计提 = Σ Fn02004(累计折旧)     ← 本期新增累计折旧 = 本期计提（近似）
  折旧率   = 折旧计提 / 固定资产原值期初余额
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
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ── 1) 固定资产明细 ──
paths = sorted(glob.glob(str(RAW / "固定资产、累计折旧*" / "FN_Fn020*.csv")))
fa = pd.concat([pd.read_csv(f, encoding=ENC, dtype={"Stkcd": str}) for f in paths],
               ignore_index=True)
fa.columns = [c.strip() for c in fa.columns]
fa = fa.rename(columns={"Stkcd": "code", "Accper": "accper", "Typrep": "typrep"})
fa["code"] = fa["code"].astype(str).str.zfill(6)
fa["accper"] = pd.to_datetime(fa["accper"], errors="coerce")
for c in ["Fn02001", "Fn02003", "Fn02004", "Fn02005", "Fn02006", "Fn02007", "Fn02009", "Fn02013"]:
    if c in fa.columns:
        fa[c] = pd.to_numeric(fa[c], errors="coerce")
# 只留合并报表（Typrep: 1=合并, 2=母公司）
fa = fa[fa["typrep"].astype(str).isin(["1", "1.0"])]
fa = fa.dropna(subset=["accper"])
log(f"固定资产明细（合并）：{len(fa):,} 行")

# 按 code+accper+科目类型 汇总
g = fa.groupby(["code", "accper", "Fn02001"], dropna=False).agg(
    begin=("Fn02003", "sum"),
    add=("Fn02004", "sum"),
    from_cip=("Fn02007", "sum"),
    decrease=("Fn02009", "sum"),
    end=("Fn02013", "sum"),
).reset_index()

pivot = g.pivot_table(index=["code", "accper"], columns="Fn02001",
                      values=["begin", "add", "from_cip", "decrease", "end"],
                      aggfunc="sum")
pivot.columns = [f"{a}_{int(b)}" if pd.notna(b) else f"{a}_na" for a, b in pivot.columns]
pivot = pivot.reset_index()

# 1=原值 2=累计折旧
keep = ["code", "accper",
        "begin_1", "add_1", "from_cip_1", "decrease_1", "end_1",      # 原值
        "add_2", "end_2"]                                             # 累计折旧
fa_panel = pivot[[c for c in keep if c in pivot.columns]].copy()
fa_panel = fa_panel.rename(columns={
    "begin_1": "fa_cost_begin", "add_1": "fa_cost_add",
    "from_cip_1": "transfer_from_cip", "decrease_1": "fa_cost_dec",
    "end_1": "fa_cost_end", "add_2": "dep_add", "end_2": "dep_end",
})
log(f"固定资产季度面板：{len(fa_panel):,} 行（code×accper）")
fa_panel.to_csv(OUT / "fa_panel.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

# ── 2) 在建工程明细 ──
cp = sorted(glob.glob(str(RAW / "在建工程*" / "FN_FN017A*.csv")))
cip = pd.concat([pd.read_csv(f, encoding=ENC, dtype={"stkcd": str}) for f in cp], ignore_index=True)
cip.columns = [c.strip().lower() for c in cip.columns]
cip = cip.rename(columns={"stkcd": "code", "accper": "accper", "typrep": "typrep"})
cip["code"] = cip["code"].astype(str).str.zfill(6)
cip["accper"] = pd.to_datetime(cip["accper"], errors="coerce")
cip = cip[cip["typrep"].astype(str).isin(["1", "1.0"])].dropna(subset=["accper"])
num = [c for c in cip.columns if c.startswith("fn_fn017a")]
for c in num:
    cip[c] = pd.to_numeric(cip[c], errors="coerce")
cip_panel = cip.groupby(["code", "accper"])[num].sum().reset_index()
log(f"在建工程季度面板：{len(cip_panel):,} 行")

# ── 3) 合并成财务因子面板 ──
bs = pd.read_csv(OUT / "bs.csv", encoding="utf-8-sig", dtype={"code": str})
bs["accper"] = pd.to_datetime(bs["accper"], errors="coerce")
inc = pd.read_csv(OUT / "inc.csv", encoding="utf-8-sig", dtype={"code": str})
inc["accper"] = pd.to_datetime(inc["accper"], errors="coerce")

base = inc[["code", "accper", "declare", "revenue", "cogs", "gm",
            "revenue_q", "cogs_q", "gross_profit_q"]].copy()
base = base.merge(bs[["code", "accper", "fa_net", "cip", "total_asset", "equity"]],
                  on=["code", "accper"], how="left")
base = base.merge(fa_panel, on=["code", "accper"], how="left")

# 转固率：上期在建工程做分母
base = base.sort_values(["code", "accper"])
base["cip_prev"] = base.groupby("code")["cip"].shift(1)
base["transfer_rate"] = np.where(base["cip_prev"] > 0,
                                 base["transfer_from_cip"] / base["cip_prev"], np.nan)
# 折旧率：折旧计提 / 固定资产原值期初
base["dep_rate"] = np.where(base["fa_cost_begin"] > 0,
                            base["dep_add"] / base["fa_cost_begin"], np.nan)
# 毛利率变化（单季口径）
base["dgm_q"] = base.groupby("code")["gm"].diff()
base["sales_q"] = base["revenue_q"]
base["dsales_q"] = base.groupby("code")["sales_q"].diff()

base.to_csv(OUT / "factor_panel.csv", index=False, encoding="utf-8-sig", float_format="%.6g")
log(f"财务因子面板 → factor_panel.csv  {len(base):,} 行")

# ── 4) 校验 ──
log("=" * 55)
log("校验（600519 贵州茅台）")
m = base[(base.code == "600519") & (base.accper == pd.Timestamp("2024-12-31"))]
if not m.empty:
    r = m.iloc[0]
    for k in ["transfer_from_cip", "fa_cost_begin", "fa_cost_end", "dep_add",
              "transfer_rate", "dep_rate", "gm"]:
        v = r.get(k)
        unit = "" if "rate" in k or k == "gm" else "亿"
        val = v / 1e8 if unit == "亿" else v
        log(f"  {k:20s} = {val if pd.notna(val) else float('nan'):>14,.4f} {unit}")

log("\n覆盖统计：")
for c in ["transfer_rate", "dep_rate", "gm", "dgm_q"]:
    v = base[c]
    log(f"  {c:14s} 非空 {v.notna().sum():>7,} / {len(base):,}  ({v.notna().mean()*100:5.1f}%)")
