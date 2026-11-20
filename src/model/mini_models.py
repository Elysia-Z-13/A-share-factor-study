# -*- coding: utf-8 -*-
"""极简模型库：Ridge（闭式解）+ 直方图 GBDT（numpy 实现，无 sklearn 依赖）"""
import numpy as np


def ridge_fit(X, y, alpha=1.0):
    """Ridge 闭式解：β = (X'X + αI)^(-1) X'y（不惩罚截距）"""
    n, p = X.shape
    Xa = np.column_stack([np.ones(n), X])
    A = Xa.T @ Xa
    reg = np.eye(p + 1) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(A + reg, Xa.T @ y)
    return beta


def ridge_predict(X, beta):
    return np.column_stack([np.ones(len(X)), X]) @ beta


# ───────────────────────── 直方图 GBDT ─────────────────────────

def make_bins(X, nbins=32):
    """按分位数把每列离散成 0..nbins-1 的桶"""
    edges = []
    Xb = np.zeros(X.shape, dtype=np.int16)
    for j in range(X.shape[1]):
        qs = np.quantile(X[:, j], np.linspace(0, 1, nbins + 1))
        qs = np.unique(qs)
        if len(qs) < 3:
            edges.append(np.array([-np.inf, np.inf]))
            Xb[:, j] = 0
            continue
        edges.append(qs)
        Xb[:, j] = np.clip(np.searchsorted(qs, X[:, j], side="right") - 1,
                           0, len(qs) - 2)
    return Xb, edges


def apply_bins(X, edges):
    Xb = np.zeros(X.shape, dtype=np.int16)
    for j in range(X.shape[1]):
        qs = edges[j]
        Xb[:, j] = np.clip(np.searchsorted(qs, X[:, j], side="right") - 1,
                           0, max(len(qs) - 2, 0))
    return Xb


class _Node:
    __slots__ = ("feat", "thr", "left", "right", "value")
    def __init__(self):
        self.feat = -1; self.thr = -1
        self.left = None; self.right = None
        self.value = 0.0


def _build(Xb, g, idx, depth, max_depth, min_leaf, nbin_list, lam):
    node = _Node()
    node.value = float(g[idx].sum() / (len(idx) + lam))
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        return node

    best_gain, best_f, best_t = 0.0, -1, -1
    gsum = g[idx].sum()
    n = len(idx)

    for j in range(Xb.shape[1]):
        nb = nbin_list[j]
        bj = Xb[idx, j].astype(np.int64)
        cnt = np.bincount(bj, minlength=nb).astype(float)
        s = np.bincount(bj, weights=g[idx], minlength=nb)
        cs_cnt = np.cumsum(cnt)[:-1]
        cs_s = np.cumsum(s)[:-1]
        tot_cnt = cnt.sum(); tot_s = s.sum()
        rs_cnt = tot_cnt - cs_cnt
        rs_s = tot_s - cs_s
        ok = (cs_cnt >= min_leaf) & (rs_cnt >= min_leaf)
        if not ok.any():
            continue
        gain = cs_s**2 / (cs_cnt + lam) + rs_s**2 / (rs_cnt + lam) - tot_s**2 / (tot_cnt + lam)
        gain = np.where(ok, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] > best_gain:
            best_gain, best_f, best_t = float(gain[k]), j, k + 1

    if best_f < 0:
        return node

    mask = Xb[idx, best_f] < best_t
    li, ri = idx[mask], idx[~mask]
    if len(li) < min_leaf or len(ri) < min_leaf:
        return node
    node.feat, node.thr = best_f, best_t
    node.left = _build(Xb, g, li, depth + 1, max_depth, min_leaf, nbin_list, lam)
    node.right = _build(Xb, g, ri, depth + 1, max_depth, min_leaf, nbin_list, lam)
    return node


def _predict_tree(node, Xb):
    out = np.empty(len(Xb))
    stack = [(node, np.arange(len(Xb)))]
    while stack:
        nd, rows = stack.pop()
        if nd.feat < 0 or nd.left is None:
            out[rows] = nd.value
            continue
        m = Xb[rows, nd.feat] < nd.thr
        stack.append((nd.left, rows[m]))
        stack.append((nd.right, rows[~m]))
    return out


class MiniGBDT:
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3,
                 min_leaf=200, nbins=32, lam=1.0, subsample=1.0, seed=42):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.nbins = nbins
        self.lam = lam
        self.subsample = subsample
        self.seed = seed
        self.trees = []
        self.base = 0.0
        self.edges = None
        self.nbin_list = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        Xb, edges = make_bins(X, self.nbins)
        self.edges = edges
        self.nbin_list = [len(e) - 1 for e in edges]
        self.base = float(np.mean(y))
        pred = np.full(len(y), self.base)
        self.trees = []
        n = len(y)
        for _ in range(self.n_estimators):
            g = y - pred                       # 平方损失的负梯度
            if self.subsample < 1.0:
                idx = rng.choice(n, int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)
            t = _build(Xb, g, idx, 0, self.max_depth, self.min_leaf,
                       self.nbin_list, self.lam)
            pred += self.lr * _predict_tree(t, Xb)
            self.trees.append(t)
        return self

    def predict(self, X):
        Xb = apply_bins(X, self.edges)
        pred = np.full(len(X), self.base)
        for t in self.trees:
            pred += self.lr * _predict_tree(t, Xb)
        return pred
