# -*- coding: utf-8 -*-
"""E6 换手、容量与 kill"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

S = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test\SCRIPT")
sys.path.insert(0, str(S))
import metrics as M

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"

e4 = json.loads((CLEAN / "meta" / "e4_backtest.json").read_text(encoding="utf-8"))
e5 = json.loads((CLEAN / "meta" / "e5_cost.json").read_text(encoding="utf-8"))
e2 = json.loads((CLEAN / "meta" / "e2_signal.json").read_text(encoding="utf-8"))

PPY = 4
TRADING_DAYS = 252
EXECUTION_DAYS = 5
PART_CAP = 0.05
AUM_TARGET = 1e8

KILL = [
    {"id": "K1", "name": "成本吃光超额",
     "rule": "滚动 12 个月净超额 ≤ 0，且非成本模型错",
     "action": "stop", "priority": "block"},
    {"id": "K2", "name": "容量不足",
     "rule": "目标资金 > 容量",
     "action": "reduce_size", "priority": "block"},
    {"id": "K3", "name": "符号翻转",
     "rule": "3 个 regime 切片中 ≥2 个滚动 IC 与全样本反号",
     "action": "stop", "priority": "block"},
    {"id": "K4", "name": "回撤失控",
     "rule": "回撤 > 回测 MDD 的 100%（即 −52.4%）",
     "action": "stop", "priority": "block"},
    {"id": "K5", "name": "信号衰减",
     "rule": "滚动 IC 连续 6 个月为负",
     "action": "stop", "priority": "block"},
]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    log("读 E4/E5/E2 结果 …")
    out = {}

    for name in e4:
        bt = pd.DataFrame(e4[name]["bt_records"])
        turn = bt["turn"].values
        m4 = e4[name]["metrics"]
        m5 = e5[name]

        t_mean = float(np.mean(turn))
        t_ann = t_mean * PPY
        hold_days = TRADING_DAYS / (t_ann * 2) if t_ann > 0 else np.inf
        # 持有期：一次完整往返对应的天数。年换手 t_ann 表示每年换掉 t_ann 倍仓位
        # 平均持有天数 = 252 / t_ann
        hold_days = TRADING_DAYS / t_ann if t_ann > 0 else np.inf

        cost_rt = m5["avg_cost_one_way_bp"] * 2 / 10000.0
        gross_ex = m5["ann_gross"] - m5["ann_bench"]

        max_turn_aggr = gross_ex / cost_rt if cost_rt > 0 else np.inf
        max_turn_cons = (abs(m4["sharpe_excess"]) * abs(m5["ann_excess_real"]) / cost_rt
                         if cost_rt > 0 else np.inf)

        min_edge = cost_rt / hold_days if hold_days > 0 else np.inf
        actual_edge = gross_ex / TRADING_DAYS

        log(f"\n=== {name} ===")
        log(f"  期换手 {t_mean*100:.1f}%  年换手 {t_ann:.2f}倍  平均持有 {hold_days:.0f} 交易日")
        log(f"  毛超额(年化) {gross_ex*100:+.2f}%  往返成本 {cost_rt*10000:.1f}bp")
        log(f"  最大可持续换手：激进版 {max_turn_aggr:.2f}倍/年  保守版 {max_turn_cons:.2f}倍/年")
        log(f"    → 当前 {t_ann:.2f}倍，余量 {max_turn_aggr/t_ann:.2f}× / {max_turn_cons/t_ann:.2f}×")
        log(f"  最小所需 edge {min_edge*10000:.2f}bp/天  实际 {actual_edge*10000:.2f}bp/天  "
            f"→ {actual_edge/min_edge if min_edge>0 else np.inf:.2f}×")

        # 容量：按三档 ADDV 分别算
        caps = {}
        for bucket, addv in [("large", 5e8), ("mid", 2e8), ("small", 3e7)]:
            # 每档占组合 1/3 资金
            aum_this = AUM_TARGET / 3
            order_per_day = aum_this * t_mean / EXECUTION_DAYS
            daily_cap = PART_CAP * addv
            max_aum_this = daily_cap * EXECUTION_DAYS / t_mean
            caps[bucket] = {"addv": addv, "daily_capacity": daily_cap,
                            "needed_per_day": order_per_day,
                            "utilization": order_per_day / daily_cap,
                            "max_aum_bucket": max_aum_this,
                            "max_aum_total": max_aum_this * 3}
        min_cap = min(v["max_aum_total"] for v in caps.values())
        log(f"  容量：各档最大 AUM " +
            " ".join(f"{k}:{v['max_aum_total']/1e8:.1f}亿" for k, v in caps.items()))
        log(f"    → 整体容量（受最小档约束）{min_cap/1e8:.1f}亿  "
            f"目标 {AUM_TARGET/1e8:.1f}亿  占比 {AUM_TARGET/min_cap*100:.1f}%  "
            f"{'✅ <50%' if AUM_TARGET/min_cap < 0.5 else '⚠️ >50%'}")

        # alpha-to-go：用 timing placebo 估半衰期
        # dgm_q 的 timing: k0 最强，k1 降到 41%
        half_life_periods = 1.0     # IC 在两个报告期之间衰减到约一半
        half_life_days = half_life_periods * (TRADING_DAYS / PPY)
        ratio = EXECUTION_DAYS / half_life_days
        retained = max(1 - ratio / 2, 0)
        log(f"  alpha-to-go：信号半衰期 ≈ {half_life_days:.0f} 交易日，建仓 {EXECUTION_DAYS} 天")
        log(f"    → 比值 {ratio*100:.1f}%  alpha 保留 ≈ {retained*100:.1f}%  "
            f"{'✅ 可忽略' if ratio < 0.1 else '⚠️'}")

        out[name] = {"turnover_per_period": t_mean, "turnover_annual": t_ann,
                     "holding_days": hold_days,
                     "gross_excess_ann": gross_ex, "cost_roundtrip_bp": cost_rt * 10000,
                     "max_turnover_aggressive": max_turn_aggr,
                     "max_turnover_conservative": max_turn_cons,
                     "margin_aggressive": max_turn_aggr / t_ann,
                     "margin_conservative": max_turn_cons / t_ann,
                     "min_edge_per_day_bp": min_edge * 10000,
                     "actual_edge_per_day_bp": actual_edge * 10000,
                     "edge_multiple": actual_edge / min_edge if min_edge > 0 else np.inf,
                     "capacity": caps, "capacity_total_min": min_cap,
                     "aum_target": AUM_TARGET,
                     "aum_vs_capacity": AUM_TARGET / min_cap,
                     "alpha_to_go": {"half_life_days": half_life_days,
                                     "exec_days": EXECUTION_DAYS,
                                     "ratio": ratio, "retained": retained},
                     "icir": next(iter(e2["results"].values()))["hac"]["icir"]}

    out["kill_criteria"] = KILL
    (CLEAN / "meta" / "e6_turnover.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log("\n  → e6_turnover.json")


if __name__ == "__main__":
    main()
