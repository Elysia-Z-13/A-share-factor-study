# -*- coding: utf-8 -*-
"""M-C 补充：二分类概率的 Brier / ECE（Precision-Recall 校准曲线）
用于如实满足 G-C.7，同时验证「概率是否可用」
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"

GROUP_FULL = ["gm_vol", "dgm_q", "gm"]
SIGN_FULL = {"gm_vol": -1, "dgm_q": +1, "gm": +1}
GROUP_MFG = ["transfer_intensity", "dep_rate"]
SIGN_MFG = {"transfer_intensity": +1, "dep_rate": +1}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def legal_deadline(a):
    y, m = a.year, a.month
    if m == 12: return pd.Timestamp(y + 1, 4, 30)
    if m == 3:  return pd.Timestamp(y, 4, 30)
    if m == 6:  return pd.Timestamp(y, 8, 31)
    if m == 9:  return pd.Timestamp(y, 10, 31)
    return a + pd.Timedelta(days=120)


def load():
    fp = pd.read_csv(CLEAN / "fin" / "factor_panel.csv",
                     encoding="utf-8-sig", dtype={"code": str})
    fp["accper_dt"] = pd.to_datetime(fp["accper"], errors="coerce")
    fp = fp.dropna(subset=["accper_dt"]).sort_values(["code", "accper_dt"])
    denom = fp["fa_cost_begin"].replace(0, np.nan)
    fp["transfer_intensity"] = fp["transfer_from_cip"] / denom
    fp["dep_rate"] = fp["dep_add"] / denom
    fp["gm_vol"] = fp.groupby("code")["gm"].transform(
        lambda s: s.rolling(4, min_periods=3).std())
    fp["dgm_q"] = fp.groupby("code")["gm"].diff()
    fp["deadline"] = fp["accper_dt"].apply(legal_deadline)
    ann = pd.to_datetime(fp.get("declare"), errors="coerce")
    fp["available_at"] = ann.where(ann.notna(), fp["deadline"])

    parts = []
    for y in range(2010, 2025):
        f = CLEAN / "labels" / f"label_{y}.csv"
        if f.exists():
            parts.append(pd.read_csv(f, encoding="utf-8-sig", dtype={"code": str},
                                     usecols=["code", "date", "label"]))
    lab = pd.concat(parts, ignore_index=True)
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab.dropna(subset=["label"]).drop_duplicates(["code", "date"])
    lab_by = {c: g.sort_values("date")[["date", "label"]].reset_index(drop=True)
              for c, g in lab.groupby("code", sort=False)}
    out = []
    for code, g in fp.groupby("code", sort=False):
        g = g.sort_values("available_at").reset_index(drop=True)
        ll = lab_by.get(code)
        if ll is None or ll.empty:
            out.append(g.assign(label=np.nan)); continue
        m = pd.merge_asof(g, ll, left_on="available_at", right_on="date",
                          direction="forward", tolerance=pd.Timedelta(days=15))
        out.append(m.drop(columns=["date"]))
    return pd.concat(out, ignore_index=True).sort_values(
        ["accper_dt", "code"]).reset_index(drop=True)


def eq_score(d, factors, signs):
    return sum(d[f].rank(pct=True) * signs[f] for f in factors) / len(factors)


def ece(prob, y, bins=10):
    """Expected Calibration Error"""
    qs = np.quantile(prob, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    if len(qs) < 3:
        return np.nan
    idx = np.clip(np.searchsorted(qs, prob, side="right") - 1, 0, len(qs) - 2)
    e = 0.0
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(y[m].mean() - prob[m].mean())
    return float(e)


def brier(prob, y):
    return float(np.mean((prob - y) ** 2))


def run_group(df, factors, signs, n_splits=5):
    ud = np.array(sorted(df["accper_dt"].unique()))
    folds = np.array_split(ud, n_splits)
    rows = []
    for k, fold in enumerate(folds, 1):
        te_days = set(fold)
        prior = np.array([d for d in ud if d not in te_days and d < min(te_days)])
        if len(prior) < 10:
            continue
        cut = int(len(prior) * 0.75)
        tr = df[df["accper_dt"].isin(set(prior[:cut]))].dropna(subset=factors + ["label"])
        ca = df[df["accper_dt"].isin(set(prior[cut:]))].dropna(subset=factors + ["label"])
        te = df[df["accper_dt"].isin(te_days)].dropna(subset=factors + ["label"])
        if min(len(tr), len(ca), len(te)) < 200:
            continue

        # 二分类：未来 60 日收益 > 0
        y_tr = (tr["label"] > 0).astype(int).values
        y_ca = (ca["label"] > 0).astype(int).values
        y_te = (te["label"] > 0).astype(int).values

        s_tr = eq_score(tr, factors, signs).values.reshape(-1, 1)
        s_ca = eq_score(ca, factors, signs).values.reshape(-1, 1)
        s_te = eq_score(te, factors, signs).values.reshape(-1, 1)

        clf = LogisticRegression(random_state=42, max_iter=1000).fit(s_tr, y_tr)
        p_raw_ca = clf.predict_proba(s_ca)[:, 1]
        p_raw_te = clf.predict_proba(s_te)[:, 1]

        # Platt：在校准集上再拟合一个逻辑回归（对 logit 做）
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) /
                                 (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        platt = LogisticRegression(random_state=42, max_iter=1000)
        platt.fit(logit(p_raw_ca).reshape(-1, 1), y_ca)
        p_pl_te = platt.predict_proba(logit(p_raw_te).reshape(-1, 1))[:, 1]

        # isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw_ca, y_ca)
        p_iso_te = np.clip(iso.predict(p_raw_te), 1e-6, 1 - 1e-6)

        rows.append({
            "fold": k, "n_test": len(te),
            "base_rate": float(y_te.mean()),
            "brier_raw": brier(p_raw_te, y_te),
            "brier_platt": brier(p_pl_te, y_te),
            "brier_iso": brier(p_iso_te, y_te),
            "ece_raw": ece(p_raw_te, y_te),
            "ece_platt": ece(p_pl_te, y_te),
            "ece_iso": ece(p_iso_te, y_te),
        })
    return rows


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行")

    out = {}
    for gname, factors, signs in [("全市场组", GROUP_FULL, SIGN_FULL),
                                  ("制造业组", GROUP_MFG, SIGN_MFG)]:
        log(f"\n=== {gname}：二分类概率校准 ===")
        rows = run_group(df, factors, signs)
        for r in rows:
            log(f"  折{r['fold']}: 基线率 {r['base_rate']*100:.1f}%  "
                f"Brier raw {r['brier_raw']:.4f} / Platt {r['brier_platt']:.4f} / Iso {r['brier_iso']:.4f}  "
                f"ECE raw {r['ece_raw']:.4f} / Platt {r['ece_platt']:.4f} / Iso {r['ece_iso']:.4f}")
        if rows:
            avg = {k: float(np.mean([r[k] for r in rows]))
                   for k in ("base_rate", "brier_raw", "brier_platt", "brier_iso",
                             "ece_raw", "ece_platt", "ece_iso")}
            log(f"  **平均**: Brier raw {avg['brier_raw']:.4f} → Platt {avg['brier_platt']:.4f} → Iso {avg['brier_iso']:.4f}")
            log(f"          ECE   raw {avg['ece_raw']:.4f} → Platt {avg['ece_platt']:.4f} → Iso {avg['ece_iso']:.4f}")
            out[gname] = {"folds": rows, "avg": avg}

    (CLEAN / "meta" / "brier_ece.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n  → brier_ece.json")


if __name__ == "__main__":
    main()
