# -*- coding: utf-8 -*-
"""F1 特征规格 → feature_spec.md"""
import json, time
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")

# ═══════ 10 个必填字段 ═══════
SPECS = [
    {
        "name": "gm",
        "family": "基本面",
        "role": "signal",
        "driver": "风险补偿",
        "inputs": "FS_Comins.B001101000(营业收入), B001201000(营业成本)",
        "observable_at": "公告日 ann_date 盘后",
        "lookback": "最近一期财报；聚合=比率",
        "reference": "横截面",
        "representation": "rank",
        "expected_failure": "风险偏好反转（牛市炒垃圾）",
        "formula": "(营业收入 − 营业成本) / 营业收入",
        "version": "v1.0", "effective": "2026-11-20",
        "note": "首版；比率型，公告日可得",
    },
    {
        "name": "dgm_q",
        "family": "基本面",
        "role": "signal",
        "driver": "持续性",
        "inputs": "gm 的相邻两期",
        "observable_at": "公告日 ann_date 盘后",
        "lookback": "最近两期；聚合=差分",
        "reference": "横截面",
        "representation": "rank",
        "expected_failure": "震荡市、改善不可持续",
        "formula": "gm_t − gm_{t-1}",
        "version": "v1.0", "effective": "2026-11-20",
        "note": "首版；单季差分",
    },
    {
        "name": "transfer_intensity",
        "family": "基本面",
        "role": "signal",
        "driver": "持续性",
        "inputs": "FN_Fn020.Fn02007(在建工程转入,类型1), Fn02003(期初余额,类型1)",
        "observable_at": "公告日盘后；⚠️仅年报/半年报披露（年度频率）",
        "lookback": "最近一期（实为年报）；聚合=求和后相除",
        "reference": "横截面",
        "representation": "rank",
        "expected_failure": "产能过剩、需求不足",
        "formula": "ΣFn02007(类型1) / ΣFn02003(类型1)",
        "version": "v1.0", "effective": "2026-11-20",
        "note": "v1.0 已把分母从 cip_prev 改为 fa_cost_begin（改含义：从转固完成率变为投资强度）",
    },
    {
        "name": "dep_rate",
        "family": "基本面",
        "role": "signal",
        "driver": "持续性",
        "inputs": "FN_Fn020.Fn02004(累计折旧本期增加,类型2), Fn02003(期初,类型1)",
        "observable_at": "公告日盘后；⚠️仅年报/半年报披露（年度频率）",
        "lookback": "最近一期（实为年报）；聚合=求和后相除",
        "reference": "横截面",
        "representation": "rank",
        "expected_failure": "公司完成资产更新（趋势中断）",
        "formula": "ΣFn02004(类型2) / ΣFn02003(类型1)",
        "version": "v1.0", "effective": "2026-11-20",
        "note": "首版；驱动强度弱",
    },
    {
        "name": "gm_vol",
        "family": "基本面",
        "role": "signal",
        "driver": "风险补偿",
        "inputs": "gm 的最近 4 期",
        "observable_at": "公告日 ann_date 盘后",
        "lookback": "4 期滚动；聚合=标准差",
        "reference": "横截面",
        "representation": "rank",
        "expected_failure": "风险偏好反转（牛市追捧高波动）",
        "formula": "std(gm 最近4期, min_periods=3)",
        "version": "v1.0", "effective": "2026-11-20",
        "note": "首版；注意回看窗 240 日 = 4×H，标注为倍长",
    },
]

REQUIRED = ["name", "family", "role", "driver", "inputs",
            "observable_at", "lookback", "reference",
            "representation", "expected_failure"]

FAIL_RULES = {
    "持续性": ["震荡", "动量崩溃", "趋势", "产能过剩", "需求不足", "中断", "资产更新"],
    "均值回复": ["单边趋势", "流动性枯竭"],
    "风险补偿": ["风险偏好反转", "牛市", "散户"],
    "可预测时钟": ["规则改变", "制度变更"],
}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def validate(spec):
    missing = [f for f in REQUIRED if not spec.get(f)]
    if missing:
        return f"{spec['name']} 缺字段：{missing}"
    if spec["role"] not in ("signal", "state"):
        return f"{spec['name']}: role 非法"
    if spec["driver"] not in FAIL_RULES:
        return f"{spec['name']}: driver 非法"
    if spec["reference"] not in ("时序", "横截面", "相对基准"):
        return f"{spec['name']}: reference 非法"
    if spec["representation"] not in ("原始", "rank", "波动缩放", "去残差"):
        return f"{spec['name']}: representation 非法"
    return None


def cross_check(spec):
    allowed = FAIL_RULES[spec["driver"]]
    if not any(k in spec["expected_failure"] for k in allowed):
        return f"{spec['name']}: 失效条件「{spec['expected_failure']}」与驱动「{spec['driver']}」不匹配"
    if spec["role"] == "state" and spec["reference"] == "横截面":
        return f"{spec['name']}: state 不应使用横截面排序"
    if "收盘可得" in spec["observable_at"] and "Comins" in spec["inputs"]:
        return f"{spec['name']}: 声明 T 收盘可得，但输入依赖财报"
    return None


def render(results, cros):
    L = []
    L.append("# feature_spec.md —— 特征规格（F1）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}　｜　因子数 {len(SPECS)}\n")

    L.append("## 一、总览\n")
    L.append("| 因子 | 家族 | 角色 | 驱动 | 参考系 | 表示 | 版本 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in SPECS:
        L.append(f"| `{s['name']}` | {s['family']} | {s['role']} | "
                 f"**{s['driver']}** | {s['reference']} | {s['representation']} | {s['version']} |")

    L.append("\n## 二、逐因子规格（10 个必填字段）\n")
    for s in SPECS:
        L.append(f"### `{s['name']}`\n")
        L.append("| # | 字段 | 值 |")
        L.append("|---|---|---|")
        L.append(f"| 1 | 名称 | `{s['name']}` |")
        L.append(f"| 2 | 家族 | {s['family']} |")
        L.append(f"| 3 | 角色 | {s['role']} |")
        L.append(f"| 4 | 驱动 | **{s['driver']}** |")
        L.append(f"| 5 | 输入 | {s['inputs']} |")
        L.append(f"| 6 | **可观测性** | **{s['observable_at']}** |")
        L.append(f"| 7 | 回看与聚合 | {s['lookback']} |")
        L.append(f"| 8 | **参考系** | {s['reference']} |")
        L.append(f"| 9 | 表示 | {s['representation']} |")
        L.append(f"| 10 | 预期失效 | {s['expected_failure']} |")
        L.append(f"\n**公式**：`{s['formula']}`\n")
        L.append(f"**版本**：{s['version']}（{s['effective']}）— {s['note']}\n")

    L.append("## 三、校验结果\n")
    L.append("| 因子 | 字段完整性 | 交叉检查 |")
    L.append("|---|---|---|")
    for s, (v, c) in zip(SPECS, zip(results, cros)):
        L.append(f"| `{s['name']}` | {'✅' if not v else '❌ ' + v} | {'✅' if not c else '❌ ' + c} |")
    L.append("")

    L.append("## 四、三条高危字段说明\n")
    L.append("**可观测性**：5 个因子的输入全部来自财报（FS_Comins / FN_Fn020），")
    L.append("因此可用时点必须是 `公告日 ann_date 盘后`，不是报告期 end_date。")
    L.append("这条写错就是前视——D2 实验实测会使 IC 虚高 1.78 倍。\n")
    L.append("**参考系**：全部用横截面（因为策略是横截面选股）。")
    L.append("手册明确：换参考系 = 改含义 = 新假设 = 重跑全套。\n")
    L.append("**表示**：全部用 rank（降噪，单调变换秩不变）。")
    L.append("若改为「去残差」（正交化掉 rev20+mom250），属于改含义，需重新记账。\n")

    L.append("## 五、已知局限\n")
    L.append("1. `transfer_intensity` 与 `dep_rate` 覆盖率仅约 19.8%（只有制造业披露附注）")
    L.append("2. `gm_vol` 回看窗 240 日 = 4×H，属倍长，IC 预期低于等长因子")
    L.append("3. `dep_rate` 驱动强度标为「弱」，即使 IC 显著也需更严的样本外验证")
    L.append("4. 全部 5 个因子都是横截面参考系，未做时序参考系版本")
    return "\n".join(L)


def main():
    log("校验 10 个必填字段 …")
    results = []
    for s in SPECS:
        e = validate(s)
        results.append(e)
        log(f"  {s['name']:22s} {'✅' if not e else '❌ ' + e}")

    log("交叉检查 …")
    cros = []
    for s in SPECS:
        e = cross_check(s)
        cros.append(e)
        log(f"  {s['name']:22s} {'✅' if not e else '❌ ' + e}")

    log("写出 …")
    (ROOT / "feature_spec.md").write_text(render(results, cros), encoding="utf-8")
    (ROOT / "CLEAN" / "meta" / "feature_spec.json").write_text(
        json.dumps(SPECS, ensure_ascii=False, indent=2), encoding="utf-8")
    log("  → feature_spec.md / feature_spec.json")


if __name__ == "__main__":
    main()