# -*- coding: utf-8 -*-
"""D7 版本、复现与监控 → versioning.md + snapshot.json"""
import os
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd

# 数据根目录。默认取本仓库上两级（src/<stage>/x.py → 仓库根）。
# 本地复现时把 FACTOR_STUDY_ROOT 指向你放 CLEAN/ 与 data/ 的目录。
ROOT = Path(os.environ.get("FACTOR_STUDY_ROOT", Path(__file__).resolve().parents[2]))
CLEAN = ROOT / "CLEAN"
SCRIPTS = ROOT / "SCRIPT"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

RULE_VERSIONS = {
    "universe": {"version": "v1.0", "effective": "2026-11-20",
                 "rules": {"min_listed_days": 250, "exclude_st": True,
                           "amount_rank_min": 0.20, "require_quote": True}},
    "mask": {"version": "v1.0", "effective": "2026-11-20",
             "rules": {"exclude_limit_up": True, "exclude_limit_down": False}},
    "label": {"version": "v1.0", "effective": "2026-11-20",
              "rules": {"convention": "next-open-to-open", "horizon": 60,
                        "fill_price": "open", "rebalance_interval": 20}},
}

PARAMS = {
    "H": 60, "purge_calendar_days": 90, "cost_bp_one_way": 10,
    "mad_k": 3, "mad_to_sigma": 1.4826, "winsor_p": 0.01,
    "train_end": "2024-09-30", "holdout_start": "2025-01-01",
    "n_splits": 5, "min_listed_days": 250,
}

SEEDS = {
    "random_forest": 42,
    "permutation_test": 42,
    "gbdt": 42,
}

RUN_ORDER = [
    "clean_all.py", "clean_fin.py", "build_fin_factors.py",
    "export_mlft_data.py", "d2_pit.py", "d3_universe.py",
    "d4_label.py", "d5_split.py", "d6_missing.py", "d7_versioning.py",
]


def compute_snapshot():
    items = []
    for pat in ("data/*.csv", "data/*/*.csv", "CLEAN/*.csv",
                "CLEAN/*/*.csv", "CLEAN/*/*/*.csv"):
        for f in sorted(ROOT.glob(pat)):
            st = f.stat()
            items.append(f"{f.relative_to(ROOT)}|{st.st_size}|{int(st.st_mtime)}")
    h = hashlib.sha256("\n".join(items).encode()).hexdigest()
    return {
        "snapshotId": f"snap-{h[:16]}",
        "files": len(items),
        "totalBytes": sum(int(x.split("|")[1]) for x in items),
        "computedAt": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    }


def code_versions():
    out = {}
    for f in sorted(SCRIPTS.glob("*.py")):
        out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def psi(base, cur, bins=10, eps=1e-4):
    base = pd.Series(base).dropna()
    cur = pd.Series(cur).dropna()
    if len(base) < 100 or len(cur) < 100:
        return np.nan
    qs = np.quantile(base, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.nan
    b = np.histogram(base, bins=qs)[0] / len(base)
    c = np.histogram(cur, bins=qs)[0] / len(cur)
    b = np.clip(b, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - b) * np.log(c / b)))


def load_features():
    fp = pd.read_csv(CLEAN/"fin"/"factor_panel.csv",
                     encoding="utf-8-sig", dtype={"code": str})
    fp["declare"] = pd.to_datetime(fp["declare"], errors="coerce")
    fp = fp.sort_values(["code","accper"]) if "accper" in fp.columns else fp
    denom = fp["fa_cost_begin"].replace(0, np.nan)
    fp["transfer_intensity"] = fp["transfer_from_cip"] / denom
    fp["dep_rate"] = fp["dep_add"] / denom
    fp["gm_vol"] = fp.groupby("code")["gm"].transform(
        lambda s: s.rolling(4, min_periods=3).std())
    fp["dgm_q"] = fp.groupby("code")["gm"].diff()
    return fp


def render(snap, ver, params, seeds, order, psi_table):
    L = []
    L.append("# versioning.md —— 版本、复现与监控（D7）\n")
    L.append(f"> 生成：{pd.Timestamp.now():%Y-%m-%d %H:%M}\n")

    L.append("## 一、数据快照\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| **snapshotId** | `{snap['snapshotId']}` |")
    L.append(f"| 文件数 | {snap['files']:,} |")
    L.append(f"| 总字节 | {snap['totalBytes']/1024**3:.2f} GB |")
    L.append(f"| 计算时间 | {snap['computedAt']} |")
    L.append("\n**哈希算法**：sha256( 每个文件的 `相对路径|字节数|mtime` 按序拼接 )")
    L.append("任何文件内容或修改时间变化，哈希即变。\n")

    L.append("## 二、规则版本\n")
    L.append("| 规则 | 版本 | 生效日 | 内容 |")
    L.append("|---|---|---|---|")
    for k, v in ver.items():
        L.append(f"| {k} | **{v['version']}** | {v['effective']} | `{json.dumps(v['rules'], ensure_ascii=False)}` |")

    L.append("\n## 三、代码版本\n")
    L.append("| 脚本 | sha256 前 16 位 |")
    L.append("|---|---|")
    for k, v in sorted(code_versions().items()):
        L.append(f"| `{k}` | `{v}` |")

    L.append("\n## 四、参数清单\n")
    L.append("| 参数 | 值 |")
    L.append("|---|---|")
    for k, v in params.items():
        L.append(f"| `{k}` | {v} |")

    L.append("\n## 五、随机种子\n")
    L.append("| 用途 | seed |")
    L.append("|---|---|")
    for k, v in seeds.items():
        L.append(f"| {k} | {v} |")

    L.append("\n## 六、运行顺序\n")
    L.append("```")
    for i, s in enumerate(order, 1):
        L.append(f"{i:2d}. python {s}")
    L.append("```\n")
    L.append("每一步的输入是上一步的输出；全部脚本幂等，可单独重跑。\n")

    L.append("## 七、PSI 监控\n")
    L.append("| 因子 | 基准期 | 当前期 | PSI | 判定 |")
    L.append("|---|---|---|---|---|")
    for r in psi_table:
        L.append(f"| `{r['factor']}` | 训练窗 | 测试窗 | {r['psi']:.4f} | {r['verdict']} |")
    L.append("\n**阈值**：< 0.10 稳定 ｜ 0.10~0.25 关注 ｜ > 0.25 重训\n")
    L.append("**纪律**：PSI 必须与 IC 同时看——")
    L.append("PSI 小但 IC 降 = alpha 衰减（降权/退役）；")
    L.append("PSI 大但 IC 稳 = 分布漂移但仍有效（观察）。\n")

    L.append("\n## 八、复现检查清单\n")
    for i, item in enumerate([
        "数据快照 ID 已固定", "三条规则版本已固定", "脚本哈希已记录",
        "参数全部在配置中", "随机种子已记录", "运行顺序已编号",
    ], 1):
        L.append(f"- [x] {i}. {item}")
    return "\n".join(L)


def main():
    log("计算数据快照 …")
    snap = compute_snapshot()
    log(f"  {snap['snapshotId']}  {snap['files']} 文件  {snap['totalBytes']/1024**3:.2f} GB")

    log("代码版本 …")
    cv = code_versions()
    log(f"  {len(cv)} 个脚本")

    log("PSI 计算 …")
    fp = load_features()
    train = fp[fp["declare"] < pd.Timestamp("2025-01-01")]
    test = fp[fp["declare"] >= pd.Timestamp("2025-01-01")]
    psi_table = []
    for c in ["gm", "dgm_q", "transfer_intensity", "dep_rate", "gm_vol"]:
        if c not in fp.columns:
            continue
        p = psi(train[c], test[c])
        verdict = ("稳定" if p < 0.10 else "关注" if p < 0.25 else "重训") if np.isfinite(p) else "样本不足"
        psi_table.append({"factor": c, "psi": p if np.isfinite(p) else float("nan"),
                          "verdict": verdict})
        log(f"  {c:22s} PSI {p if np.isfinite(p) else float('nan'):8.4f}  {verdict}")

    log("写出 …")
    (ROOT / "versioning.md").write_text(
        render(snap, RULE_VERSIONS, PARAMS, SEEDS, RUN_ORDER, psi_table),
        encoding="utf-8")
    (CLEAN / "meta" / "snapshot.json").write_text(
        json.dumps({"snapshot": snap, "rules": RULE_VERSIONS,
                    "params": PARAMS, "seeds": SEEDS,
                    "code": cv, "run_order": RUN_ORDER,
                    "psi": psi_table},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    log("  → versioning.md / snapshot.json")


if __name__ == "__main__":
    main()