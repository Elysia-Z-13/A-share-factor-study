# -*- coding: utf-8 -*-
"""E5 成本评估：三层成本 → 参与率选档 → 滑点压测 → break-even"""
import os
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

# metrics.py 所在目录（仓库的 src/）
S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
import metrics as M

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
data = json.loads((CLEAN / "meta" / "e4_backtest.json").read_text(encoding="utf-8"))

EXPLICIT = {"commission_bp": 2.5, "stamp_sell_bp": 10.0, "transfer_bp": 0.1}
SPREAD_BP = {"large": 10, "mid": 20, "small": 50}
SLIP_FIXED_BP = {"large": 5, "mid": 15, "small": 25}
ETA_UPPER = {"large": 0.30, "mid": 0.40, "small": 0.60}
GRID = (0, 5, 10, 15, 20, 30, 40, 60, 80)
AUM = 1e8
PPY = 4

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def explicit_cost(is_sell):
    c = EXPLICIT["commission_bp"] + EXPLICIT["transfer_bp"]
    return c + (EXPLICIT["stamp_sell_bp"] if is_sell else 0.0)


def implicit_cost_bp(part, bucket, daily_vol_pct=2.0):
    if part < 0.005:
        return SPREAD_BP[bucket] * 0.5
    if part < 0.02:
        return SLIP_FIXED_BP[bucket] + 10000.0 * part * 0.5
    return ETA_UPPER[bucket] * (daily_vol_pct / 100.0) * np.sqrt(part) * 10000.0


def main():
    log("成本三层拆解 …")
    buy_exp = explicit_cost(False)
    sell_exp = explicit_cost(True)
    log(f"  显式 买入单边 {buy_exp:.1f}bp / 卖出单边 {sell_exp:.1f}bp / 往返 {buy_exp+sell_exp:.1f}bp")
    log(f"  （E4 曾用统一 10bp → 偏低 {(buy_exp+sell_exp)/2 - 10:+.1f}bp）")

    # 用 E4 的分年度数据反推每期 gross / bench / turnover
    res = {}
    for name in data:
        recs = data[name]["bt_records"]
        bt = pd.DataFrame(recs)
        bt["date"] = pd.to_datetime(bt["date"])
        gross = bt["gross"].values
        bench = bt["bench"].values
        turn = bt["turn"].values
        ann_bench = (1 + bench).prod() ** (PPY / len(bench)) - 1

        # 参与率分布：假设持仓等权、换手 t、AUM 固定
        # 用成交额分位近似：大/中/小三档各 1/3
        k = 723
        w_each = 1.0 / k
        parts = {}
        for bucket, addv in [("large", 5e8), ("mid", 2e8), ("small", 3e7)]:
            order = AUM * w_each * float(np.mean(turn))
            parts[bucket] = order / addv

        # 按参与率算每档成本
        bucket_cost = {}
        for bucket, p in parts.items():
            slip = implicit_cost_bp(p, bucket)
            bucket_cost[bucket] = {"participation": p,
                                   "implicit_bp": slip,
                                   "explicit_avg_bp": (buy_exp + sell_exp) / 2,
                                   "total_one_way_bp": slip + (buy_exp + sell_exp) / 2}

        # 组合加权平均单边成本（三档等权）
        avg_cost_bp = float(np.mean([v["total_one_way_bp"] for v in bucket_cost.values()]))
        log(f"\n=== {name} ===")
        log(f"  参与率: " + " ".join(f"{k2}:{v['participation']*100:.2f}%" for k2, v in bucket_cost.items()))
        log(f"  单边成本: " + " ".join(f"{k2}:{v['total_one_way_bp']:.1f}bp" for k2, v in bucket_cost.items()))
        log(f"  组合加权单边 {avg_cost_bp:.1f}bp（E4 用 10bp）")

        # 用真实成本重算净收益
        t_mean = float(np.mean(turn))
        net_real = (1 + gross) * (1 - t_mean * 2 * avg_cost_bp / 10000) - 1
        ann_net_real = (1 + net_real).prod() ** (PPY / len(net_real)) - 1
        ann_gross = (1 + gross).prod() ** (PPY / len(gross)) - 1
        log(f"  毛年化 {ann_gross*100:+.2f}%  净年化（真实成本）{ann_net_real*100:+.2f}%  "
            f"基准 {ann_bench*100:+.2f}%")
        log(f"  净超额（真实成本）{ann_net_real*100 - ann_bench*100:+.2f}%")

        # 成本网格与 break-even
        rows = []
        for c_bp in GRID:
            c = c_bp / 10000.0
            net = (1 + gross) * (1 - t_mean * 2 * c) - 1
            a = (1 + net).prod() ** (PPY / len(net)) - 1
            rows.append({"cost_bp": c_bp, "ann_net": float(a),
                         "ann_excess": float(a - ann_bench)})
        gdf = pd.DataFrame(rows)
        be = None
        for i in range(len(gdf) - 1):
            a, b2 = gdf.iloc[i]["ann_excess"], gdf.iloc[i + 1]["ann_excess"]
            if a > 0 >= b2:
                be = float(gdf.iloc[i]["cost_bp"] +
                           (gdf.iloc[i + 1]["cost_bp"] - gdf.iloc[i]["cost_bp"]) * a / (a - b2))
                break
        mult = be / avg_cost_bp if be else None
        log(f"  break-even {be:.1f}bp  真实 {avg_cost_bp:.1f}bp  倍数 {mult:.2f}×")

        # 滑点翻倍压测
        slip_share = avg_cost_bp - (buy_exp + sell_exp) / 2
        cost_double = (buy_exp + sell_exp) / 2 + slip_share * 2
        net_dbl = (1 + gross) * (1 - t_mean * 2 * cost_double / 10000) - 1
        ann_dbl = (1 + net_dbl).prod() ** (PPY / len(net_dbl)) - 1
        log(f"  滑点翻倍：单边 {cost_double:.1f}bp → 净超额 {ann_dbl*100 - ann_bench*100:+.2f}%")

        res[name] = {"explicit": {"buy_bp": buy_exp, "sell_bp": sell_exp,
                                  "roundtrip_bp": buy_exp + sell_exp},
                     "participation": {k2: v["participation"] for k2, v in bucket_cost.items()},
                     "bucket_cost": bucket_cost,
                     "avg_cost_one_way_bp": avg_cost_bp,
                     "ann_gross": float(ann_gross), "ann_net_real": float(ann_net_real),
                     "ann_bench": float(ann_bench),
                     "ann_excess_real": float(ann_net_real - ann_bench),
                     "grid": gdf.to_dict("records"), "breakeven_bp": be,
                     "breakeven_multiple": mult,
                     "slippage_double": {"cost_bp": cost_double,
                                         "ann_excess": float(ann_dbl - ann_bench)},
                     "turnover_mean": t_mean, "aum": AUM}

    (CLEAN / "meta" / "e5_cost.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("\n  → e5_cost.json")


if __name__ == "__main__":
    main()
