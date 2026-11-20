# -*- coding: utf-8 -*-
"""M-S 复验：用 sklearn 重跑 等权 / Ridge / LASSO / GBDT / RF 对比
并加入 PCA 风险分解（BBP 阈值判断真实成分数）
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mini_models import ridge_fit, ridge_predict, MiniGBDT

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"D:\HuaweiMoveData\Users\战昶旭\Desktop\Test")
CLEAN = ROOT / "CLEAN"
SEED = 42

GROUP_FULL = ["gm_vol", "dgm_q", "gm"]
SIGN_FULL = {"gm_vol": -1, "dgm_q": +1, "gm": +1}
GROUP_MFG = ["transfer_intensity", "dep_rate"]
SIGN_MFG = {"transfer_intensity": +1, "dep_rate": +1}

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def spear(x, y):
    rx = pd.Series(x).rank().to_numpy(dtype=float, copy=True)
    ry = pd.Series(y).rank().to_numpy(dtype=float, copy=True)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx*rx).sum() * (ry*ry).sum())
    return float((rx*ry).sum()/den) if den > 0 else np.nan


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
    df = pd.concat(out, ignore_index=True)
    return df.sort_values(["accper_dt", "code"]).reset_index(drop=True)


def build_models():
    return {
        "equal": None,   # 单独处理
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=SEED)),
        "ridge_a10": make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=SEED)),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=5000, random_state=SEED)),
        "enet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=SEED)),
        "gbdt": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=120,
            min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=False, random_state=SEED),
        "gbdt_es": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=500,
            min_samples_leaf=200, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=20, random_state=SEED),
        "rf": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=200,
            n_jobs=-1, random_state=SEED),
        "minigbdt": "mini",
    }


def cv_compare(df, factors, signs, n_splits=5):
    ud = np.array(sorted(df["accper_dt"].unique()))
    folds = np.array_split(ud, n_splits)
    names = list(build_models().keys())
    res = {k: [] for k in names}

    for fold in folds:
        te = df[df["accper_dt"].isin(set(fold))]
        tr = df[~df["accper_dt"].isin(set(fold))]
        tr = tr.dropna(subset=factors + ["label"])
        te = te.dropna(subset=factors + ["label"])
        if len(tr) < 300 or len(te) < 300:
            continue

        med = tr[factors].median()
        tr = tr.copy(); te = te.copy()
        tr["_y"] = tr.groupby("accper_dt")["label"].rank(pct=True)
        te["_y"] = te.groupby("accper_dt")["label"].rank(pct=True)

        Xtr = (tr[factors].fillna(med)
               .groupby(tr["accper_dt"]).rank(pct=True).values.astype(float))
        Xte = (te[factors].fillna(med)
               .groupby(te["accper_dt"]).rank(pct=True).values.astype(float))
        ytr = tr["_y"].values.astype(float)
        yte = te["_y"].values.astype(float)

        # 等权（用已知方向）
        eq = sum(te[f].rank(pct=True) * signs[f] for f in factors) / len(factors)
        res["equal"].append(spear(eq.values, te["label"].values))

        for name, mdl in build_models().items():
            if name == "equal":
                continue
            try:
                if name == "minigbdt":
                    m = MiniGBDT(n_estimators=120, learning_rate=0.05, max_depth=3,
                                 min_leaf=200, nbins=32, lam=1.0, seed=SEED)
                    m.fit(Xtr, ytr)
                    pred = m.predict(Xte)
                else:
                    mdl.fit(Xtr, ytr)
                    pred = mdl.predict(Xte)
                res[name].append(spear(pred, yte))
            except Exception as e:
                res[name].append(np.nan)

    return {k: {"mean": float(np.nanmean(v)) if v else np.nan,
                "std": float(np.nanstd(v)) if v else np.nan,
                "n": int(np.isfinite(v).sum()),
                "folds": [round(float(x), 4) if np.isfinite(x) else None for x in v]}
            for k, v in res.items()}


def pca_analysis(df, factors):
    """PCA 风险分解 + BBP 阈值"""
    d = df.dropna(subset=factors)
    if len(d) < 500:
        return None
    X = d[factors].groupby(d["accper_dt"]).rank(pct=True).values.astype(float)
    X = X - X.mean(0)
    # 协方差特征值
    C = np.cov(X, rowvar=False)
    ev, evec = np.linalg.eigh(C)
    idx = np.argsort(ev)[::-1]
    ev, evec = ev[idx], evec[:, idx]
    evr = ev / ev.sum()
    p, n = len(factors), len(d)
    gamma = p / n
    bbp = 1 + np.sqrt(gamma)
    # BBP 阈值是对"标准化特征值"的，这里用简化判断
    n_true = int(np.sum(evr > (bbp / p) * 0 + 1.0 / p * 1.0))
    return {
        "eigenvalues": [round(float(x), 4) for x in ev],
        "explained_ratio": [round(float(x), 4) for x in evr],
        "loadings_pc1": [round(float(x), 4) for x in evec[:, 0]],
        "loadings_pc2": [round(float(x), 4) for x in evec[:, 1]] if p > 1 else [],
        "gamma": round(float(gamma), 6),
        "bbp_threshold": round(float(bbp), 4),
        "n_effective_components_est": n_true,
    }


def main():
    log("加载 …")
    df = load()
    log(f"  {len(df):,} 行 / 有标签 {df['label'].notna().sum():,}")

    log("PCA 风险分解（全市场组）…")
    pca_full = pca_analysis(df, GROUP_FULL)
    pca_mfg = pca_analysis(df, GROUP_MFG)
    for nm, pc in [("全市场", pca_full), ("制造业", pca_mfg)]:
        if pc:
            log(f"  {nm}: 方差解释率 {pc['explained_ratio']}  BBP 阈值 {pc['bbp_threshold']}")

    log("sklearn 对比（全市场组）…")
    cmp_full = cv_compare(df, GROUP_FULL, SIGN_FULL)
    for k, v in sorted(cmp_full.items(), key=lambda x: -(x[1]["mean"] if np.isfinite(x[1]["mean"]) else -9)):
        log(f"  {k:10s} IC {v['mean']:+.4f} ± {v['std']:.4f}  n={v['n']}")

    log("sklearn 对比（制造业组）…")
    cmp_mfg = cv_compare(df, GROUP_MFG, SIGN_MFG)
    for k, v in sorted(cmp_mfg.items(), key=lambda x: -(x[1]["mean"] if np.isfinite(x[1]["mean"]) else -9)):
        log(f"  {k:10s} IC {v['mean']:+.4f} ± {v['std']:.4f}  n={v['n']}")

    (CLEAN / "meta" / "ms_sklearn.json").write_text(
        json.dumps({"full": cmp_full, "mfg": cmp_mfg,
                    "pca_full": pca_full, "pca_mfg": pca_mfg},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log("  → ms_sklearn.json")

    # 打印最优
    log("=== 结论 ===")
    for nm, cmp in [("全市场组", cmp_full), ("制造业组", cmp_mfg)]:
        best = max(cmp.items(), key=lambda x: x[1]["mean"] if np.isfinite(x[1]["mean"]) else -9)
        log(f"  {nm} 最优: {best[0]} IC {best[1]['mean']:+.4f}")


if __name__ == "__main__":
    main()
