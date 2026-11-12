#!/usr/bin/env python3
"""
Model zoo vs STANDARD erasure-aware decoding on the d=3 color code ([[7,1,3]]),
under the correlated-leakage toy from heralding_correlation_sweep.py.

Every decoder sees ONLY (3 syndrome bits + 7 herald bits) = 10 bits and predicts
whether the canonical correction E0[syndrome] must be multiplied by the logical
L (the "relative logical class"). Failure = wrong class.

Key trick: the noise model factorizes over pairs, so the joint distribution of
(heralds, bit-flips) has only 16^3 * 4 = 16384 atoms. We enumerate it, which
gives EXACT logical error rates for any decoder (a 1024-entry function of the
input) -- no Monte-Carlo noise in evaluation. Only training is sampled, so the
learning curves show pure learner/sample-efficiency effects.

References (no training):
  STANDARD : exact ML with independent per-qubit priors (uses heralds)
  ORACLE   : Bayes-optimal given the observables (argmax P(class | input)),
             the ceiling every learner is chasing.
Learners (sklearn), trained on N sampled shots from the TRUE correlated model.

Speed: every decoder is timed on a 1M-shot batch in its native form. Since the
input space at d=3 is 1024 points, any decoder can also be compiled to a LUT;
that throughput is reported as well (it is the same for all of them).

Usage:
  python model_zoo.py                # full run (parallel), then report
  python model_zoo.py --quick        # smoke test
  python model_zoo.py --report-only  # rebuild figures/ + model_zoo_report.md from CSVs

Outputs: results/*.csv (raw per-seed runs, refs, speed, tie decomposition),
figures/*.png, model_zoo_report.md
"""

import argparse
import csv
import os
import re
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from heralding_correlation_sweep import (
    E0, H, KER, L, N_Q, PAIRS, SINGLE, pair_table, single_table, _logsumexp)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_IN = 1 << (3 + N_Q)  # 1024 distinct decoder inputs
E0_ARR = np.stack([E0[tuple((s >> np.arange(3)) & 1)] for s in range(8)])  # (8,7)
E0_LOG = E0_ARR @ L % 2                                                     # (8,)


def input_index(syn, her):
    """syn (n,3), her (n,7) -> int in [0,1024). bits: syn0..2 then her0..6."""
    w = 1 << np.arange(3 + N_Q)
    return np.concatenate([syn, her], axis=1).astype(np.int64) @ w


def index_bits(idx):
    return ((np.asarray(idx)[:, None] >> np.arange(3 + N_Q)) & 1).astype(np.uint8)


# ---------------------------------------------------------------- exact model
def enumerate_atoms(p_loss, p_pauli, eta, c):
    """All 16384 (prob, input_idx, rel_class) atoms of the true distribution."""
    T = pair_table(p_loss, p_pauli, eta, c); T = T / T.sum()
    S = single_table(p_loss, p_pauli, eta); S = S / S.sum()
    t = np.arange(16); s = np.arange(4)
    g = np.meshgrid(t, t, t, s, indexing="ij")
    k0, k1, k2, ks = (x.ravel() for x in g)
    prob = T.ravel()[k0] * T.ravel()[k1] * T.ravel()[k2] * S.ravel()[ks]
    n = prob.size
    bit = np.zeros((n, N_Q), np.uint8); her = np.zeros((n, N_Q), np.uint8)
    for k, (qa, qb) in zip((k0, k1, k2), PAIRS):
        ha, hb, ba, bb = np.unravel_index(k, (2, 2, 2, 2))
        her[:, qa], her[:, qb], bit[:, qa], bit[:, qb] = ha, hb, ba, bb
    hs, bs = np.unravel_index(ks, (2, 2))
    her[:, SINGLE], bit[:, SINGLE] = hs, bs
    syn = bit @ H.T % 2
    idx = input_index(syn, her)
    rel = (bit @ L % 2) ^ E0_LOG[syn @ (1 << np.arange(3))]
    return prob, idx, rel.astype(np.uint8), T, S


def joint_table(prob, idx, rel):
    """P[input, class] as a (1024, 2) array."""
    J = np.zeros((N_IN, 2))
    np.add.at(J, (idx, rel), prob)
    return J


def exact_ler(J, pred):
    """pred: (1024,) class predictions. Exact LER under J."""
    return float(J[np.arange(N_IN), 1 - pred].sum())


def standard_predictions(T, S):
    """STANDARD decoder as a 1024-entry table: exact ML with independent
    per-qubit priors P(b | own herald). Returns (prediction, exact-tie mask)."""
    llr = standard_llr(T, S)
    tie = np.abs(llr) < 1e-9   # independent model is genuinely indifferent
    return (llr > 0).astype(np.uint8), tie


def standard_llr(T, S):
    """STANDARD's log-likelihood ratio log P(class 1)/P(class 0) per input,
    under its independent per-qubit model. This is its full soft output."""
    eps = 1e-12
    Ph = T.sum(axis=(1, 3))                          # [ha, ba]
    Mpair = Ph / (Ph.sum(axis=1, keepdims=True) + eps)
    Msing = S / (S.sum(axis=1, keepdims=True) + eps)
    LOGM = np.log(np.stack([Mpair] * 6 + [Msing]) + eps)  # (7,2,2)
    bits = index_bits(np.arange(N_IN))
    syn, her = bits[:, :3], bits[:, 3:]
    sidx = syn.astype(int) @ (1 << np.arange(3))
    cand = E0_ARR[sidx][:, None, :] ^ KER[None]      # (1024,16,7)
    rel_c = (cand @ L % 2) ^ E0_LOG[sidx][:, None]
    score = np.zeros(cand.shape[:2])
    for i in range(N_Q):
        score += LOGM[i][her[:, i][:, None], cand[:, :, i]]
    pos = _logsumexp(np.where(rel_c == 1, score, -np.inf), 1)
    neg = _logsumexp(np.where(rel_c == 0, score, -np.inf), 1)
    return pos - neg


def standard_lers(J, pred, tie):
    """STANDARD has exact ties; its LER depends on how they are broken.
    Returns (random tie-break = expected LER, best-possible tie-break)."""
    wrong = J[np.arange(N_IN), 1 - pred] * ~tie
    return (float(wrong.sum() + 0.5 * J[tie].sum()),
            float(wrong.sum() + J[tie].min(axis=1).sum()))


# ------------------------------------------------------------------ learners
class CountLUT:
    """Empirical Bayes: majority class per input; unseen inputs fall back to
    a supplied prior table (STANDARD), so it never does worse than chance."""
    def __init__(self, fallback):
        self.fallback = fallback

    def fit(self, X, y):
        idx = X.astype(np.int64) @ (1 << np.arange(X.shape[1]))
        C = np.zeros((N_IN, 2)); np.add.at(C, (idx, y), 1)
        self.table = np.where(C.sum(1) == 0, self.fallback,
                              (C[:, 1] > C[:, 0]).astype(np.uint8))
        return self

    def predict(self, X):
        return self.table[X.astype(np.int64) @ (1 << np.arange(X.shape[1]))]


def _idx(X):
    return X.astype(np.int64) @ (1 << np.arange(X.shape[1]))


def _counts(X, y):
    C = np.zeros((N_IN, 2)); np.add.at(C, (_idx(X), y), 1)
    return C


class TieBreakLUT:
    """HYBRID 1: follow STANDARD wherever it has a preference; learn a majority
    vote ONLY on inputs where STANDARD is exactly tied. Its ceiling is exactly
    'STANDARD + best tie-break'."""
    def __init__(self, std, tie):
        self.std, self.tie = std, tie

    def fit(self, X, y):
        C = _counts(X, y)
        learned = np.where(C[:, 1] > C[:, 0], 1, np.where(C[:, 0] > C[:, 1], 0, self.std))
        self.table = np.where(self.tie, learned, self.std).astype(np.uint8)
        return self

    def predict(self, X):
        return self.table[_idx(X)]


class PriorLUT:
    """HYBRID 2: Bayesian lookup table with STANDARD's soft posterior as the
    prior, worth `alpha` pseudo-shots per input. Data can override STANDARD
    anywhere, but only with enough evidence:
        P(1 | input) = (n1 + alpha * p_std) / (n + alpha)."""
    def __init__(self, std, llr, alpha):
        self.std, self.alpha = std, alpha
        self.p_std = 1.0 / (1.0 + np.exp(-np.clip(llr, -50, 50)))

    def fit(self, X, y):
        C = _counts(X, y)
        post = (C[:, 1] + self.alpha * self.p_std) / (C.sum(1) + self.alpha)
        self.table = np.where(np.abs(post - 0.5) < 1e-12, self.std,
                              (post > 0.5).astype(np.uint8)).astype(np.uint8)
        return self

    def predict(self, X):
        return self.table[_idx(X)]


class WithLLR:
    """HYBRID 3: any classifier, given STANDARD's LLR as an extra feature. The
    scalable hybrid: needs no table, so it carries over to d=5,7 where the
    input space cannot be enumerated."""
    def __init__(self, base, llr):
        self.base, self.llr = base, llr

    def _aug(self, X):
        return np.hstack([X, self.llr[_idx(X)][:, None]])

    def fit(self, X, y):
        self.base.fit(self._aug(X), y)
        return self

    def predict(self, X):
        return self.base.predict(self._aug(X))


class ResidualXGB:
    """HYBRID 4: gradient boosting that STARTS from STANDARD. STANDARD's LLR is
    the initial margin (XGBoost base_margin), so with no trees the model IS
    STANDARD (soft), and the trees learn only the residual correction. Regularized
    (shallow trees, min_child_weight, L2) so small samples cannot move it far."""
    def __init__(self, llr, seed):
        self.llr = np.clip(llr, -15, 15)
        self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.m = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                               min_child_weight=5, reg_lambda=5.0, tree_method="hist",
                               n_jobs=1, random_state=self.seed, verbosity=0)
        self.m.fit(X, y, base_margin=self.llr[_idx(X)])
        return self

    def predict(self, X):
        return self.m.predict(X, base_margin=self.llr[_idx(X)])


PLAIN = ["lookup table", "naive Bayes", "logistic", "logistic+pairs", "MLP 64x64",
         "random forest", "hist grad boost", "kNN (k=25)"]
HYBRID = ["hybrid: tie-break LUT", "hybrid: prior LUT", "hybrid: HGB + STANDARD LLR",
          "hybrid: residual XGB"]


def make_models(prior, seed):
    """prior: dict with STANDARD's 'std' (hard decision), 'tie' mask and 'llr'."""
    std = prior["std"]
    return {
        "hybrid: tie-break LUT": lambda: TieBreakLUT(std, prior["tie"]),
        "hybrid: prior LUT": lambda: PriorLUT(std, prior["llr"], alpha=10.0),
        "hybrid: HGB + STANDARD LLR": lambda: WithLLR(
            HistGradientBoostingClassifier(max_iter=300, random_state=seed), prior["llr"]),
        "hybrid: residual XGB": lambda: ResidualXGB(prior["llr"], seed),
        "lookup table": lambda: CountLUT(std),
        "naive Bayes": lambda: BernoulliNB(alpha=1.0),
        "logistic": lambda: LogisticRegression(max_iter=500),
        "logistic+pairs": lambda: make_pipeline(
            PolynomialFeatures(2, interaction_only=True, include_bias=False),
            LogisticRegression(max_iter=1000, C=10.0)),
        "MLP 64x64": lambda: MLPClassifier((64, 64), max_iter=200,
                                           early_stopping=True, random_state=seed),
        "random forest": lambda: RandomForestClassifier(
            200, min_samples_leaf=3, n_jobs=1, random_state=seed),
        "hist grad boost": lambda: HistGradientBoostingClassifier(
            max_iter=300, random_state=seed),
        "kNN (k=25)": lambda: KNeighborsClassifier(25, n_jobs=1),
    }


def sample_train(prob, idx, rel, n, rng):
    k = rng.choice(prob.size, size=n, p=prob)
    return index_bits(idx[k]), rel[k]


def throughput(predict, X, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); predict(X); best = min(best, time.perf_counter() - t0)
    return X.shape[0] / best




def standard_ml_shots(X, T, S):
    """STANDARD evaluated per shot (no precompilation): 16-candidate ML sum.
    This is the 'native' cost of the reference decoder for speed comparisons."""
    eps = 1e-12
    Ph = T.sum(axis=(1, 3)); Mpair = Ph / (Ph.sum(1, keepdims=True) + eps)
    Msing = S / (S.sum(1, keepdims=True) + eps)
    LOGM = np.log(np.stack([Mpair] * 6 + [Msing]) + eps)
    sidx = X[:, :3].astype(int) @ (1 << np.arange(3)); her = X[:, 3:]
    cand = E0_ARR[sidx][:, None, :] ^ KER[None]
    rel_c = (cand @ L % 2) ^ E0_LOG[sidx][:, None]
    score = np.zeros(cand.shape[:2])
    for i in range(N_Q):
        score += LOGM[i][her[:, i][:, None], cand[:, :, i]]
    return (_logsumexp(np.where(rel_c == 1, score, -np.inf), 1) >
            _logsumexp(np.where(rel_c == 0, score, -np.inf), 1)).astype(np.uint8)


# ------------------------------------------------------------------- compute
POINTS = [  # (label, p_loss, p_pauli, eta, c)
    ("independent, eta=0.7", 0.08, 0.01, 0.7, 0.0),
    ("correlated, eta=1.0", 0.08, 0.01, 1.0, 0.4),
    ("correlated, eta=0.7", 0.08, 0.01, 0.7, 0.4),
    ("strongly correlated, eta=0.7", 0.08, 0.01, 0.7, 0.6),
]
SPEED_POINT = 2          # speed is ~independent of the noise point; measure once
MODEL_NAMES = PLAIN + HYBRID
TABLE_MODELS = {"lookup table", "hybrid: tie-break LUT", "hybrid: prior LUT"}
RES, FIG = "results", "figures"


def point_setup(p_loss, p_pauli, eta, c):
    prob, idx, rel, T, S = enumerate_atoms(p_loss, p_pauli, eta, c)
    J = joint_table(prob, idx, rel)
    std, tie = standard_predictions(T, S)
    return dict(prob=prob, idx=idx, rel=rel, T=T, S=S, J=J, std=std, tie=tie,
                llr=standard_llr(T, S),
                oracle=(J[:, 1] > J[:, 0]).astype(np.uint8))


def fit_one(ctx, mname, N, seed):
    """Train one model instance. The training set depends only on (seed, N), so
    all models at a given (seed, N) see the SAME data -> paired comparison."""
    X, y = sample_train(ctx["prob"], ctx["idx"], ctx["rel"], N,
                        np.random.default_rng([seed, N]))
    t0 = time.perf_counter()
    if mname not in TABLE_MODELS and len(np.unique(y)) < 2:   # degenerate sample
        pred = np.full(N_IN, y[0], np.uint8)
    else:
        m = make_models(ctx, seed)[mname]().fit(X, y)
        pred = m.predict(index_bits(np.arange(N_IN))).astype(np.uint8)
    return mname, N, seed, exact_ler(ctx["J"], pred), time.perf_counter() - t0


def tie_decomposition():
    """How much of the STANDARD->ORACLE gap is just breaking STANDARD's exact
    ties well, vs. overriding decisions STANDARD was confident about."""
    with open(f"{RES}/ties.csv", "w") as f:
        f.write("p_loss,p_pauli,eta,c,LER_std_random_tie,LER_std_best_tie,LER_oracle,"
                "gain_total,gain_ties_only,frac_gain_from_ties,tied_mass,nontie_override_mass\n")
        for pl in (0.04, 0.08, 0.15):
            for pp in (0.003, 0.01, 0.03):
                for eta in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                    for c in np.round(np.linspace(0, 0.8, 9), 2):
                        x = point_setup(pl, pp, eta, c); J = x["J"]
                        a, b = standard_lers(J, x["std"], x["tie"])
                        o = exact_ler(J, x["oracle"])
                        dis = (~x["tie"]) & (x["oracle"] != x["std"])
                        frac = (a - b) / (a - o) if a - o > 1e-9 else float("nan")
                        f.write(f"{pl},{pp},{eta},{c},{a:.6g},{b:.6g},{o:.6g},{a / o:.5f},"
                                f"{a / b:.5f},{frac:.4f},{J[x['tie']].sum():.4g},"
                                f"{J[dis].sum():.4g}\n")


def compute(Ns, seeds, n_speed, only=None):
    """only: list of model names to (re)run; their rows replace any existing
    rows in results/runs.csv and results/speed.csv, everything else is kept."""
    models = only or MODEL_NAMES
    from joblib import Parallel, delayed
    os.makedirs(RES, exist_ok=True)
    t_start = time.perf_counter()
    refs, runs, speed = [], [], []
    for k, (name, pl, pp, eta, c) in enumerate(POINTS):
        x = point_setup(pl, pp, eta, c)
        ls, lsb = standard_lers(x["J"], x["std"], x["tie"])
        lo = exact_ler(x["J"], x["oracle"])
        refs.append((name, pl, pp, eta, c, ls, lsb, lo, int(x["tie"].sum()),
                     float(x["J"][x["tie"]].sum())))
        jobs = sorted(((m, N, s) for m in models for N in Ns for s in range(seeds)),
                      key=lambda j: -j[1])              # big jobs first
        out = Parallel(n_jobs=-1)(delayed(fit_one)(x, *j) for j in jobs)
        runs += [(name,) + r for r in out]
        print(f"[{time.perf_counter() - t_start:7.0f}s] {name}: {len(jobs)} fits done | "
              f"STANDARD {ls:.5f}  best-tie {lsb:.5f}  ORACLE {lo:.5f}", flush=True)

        if k == SPEED_POINT:   # clean single-process timing, no contention
            Xs, _ = sample_train(x["prob"], x["idx"], x["rel"], n_speed,
                                 np.random.default_rng(7))
            w = 1 << np.arange(10)
            for mname in models:
                Xt, yt = sample_train(x["prob"], x["idx"], x["rel"], Ns[-1],
                                      np.random.default_rng(11))
                m = make_models(x, 0)[mname]()
                t0 = time.perf_counter(); m.fit(Xt, yt); ft = time.perf_counter() - t0
                knn = mname.startswith("kNN")
                rate = throughput(m.predict, Xs[:20_000] if knn else Xs, reps=1 if knn else 3)
                speed.append((mname, "native sklearn/numpy", ft, rate))
            if not only:   # reference rows only on a full run
                speed.append(("STANDARD", "per-shot ML enumeration", 0.0,
                              throughput(lambda X: standard_ml_shots(X, x["T"], x["S"]), Xs)))
                speed.append(("any decoder", "compiled 1024-entry LUT", 0.0,
                              throughput(lambda X: x["oracle"][X.astype(np.int64) @ w], Xs)))
            print(f"[{time.perf_counter() - t_start:7.0f}s] speed benchmark done", flush=True)

    with open(f"{RES}/refs.csv", "w") as f:
        f.write("point,p_loss,p_pauli,eta,c,LER_standard,LER_standard_best_tie,"
                "LER_oracle,tied_inputs,tied_mass\n")
        cw = csv.writer(f, lineterminator="\n")
        for r in refs:
            cw.writerow(list(r[:5]) + [f"{v:.7g}" for v in r[5:8]] + [r[8], f"{r[9]:.5g}"])
    runs = [list(r[:4]) + [f"{r[4]:.7g}", f"{r[5]:.4g}"] for r in runs]
    speed = [[r[0], r[1], f"{r[2]:.4g}", f"{r[3]:.4g}"] for r in speed]
    if only:   # merge: keep every existing row for models we did not rerun
        keep = lambda rows: [list(r.values()) for r in rows if r["model"] not in only]
        runs = keep(_read(f"{RES}/runs.csv")) + runs
        speed = keep(_read(f"{RES}/speed.csv")) + speed
    for path, header, rows in ((f"{RES}/runs.csv", "point,model,N_train,seed,LER,fit_seconds", runs),
                               (f"{RES}/speed.csv", "model,form,fit_seconds_at_Nmax,shots_per_sec", speed)):
        with open(path, "w") as f:
            f.write(header + "\n")
            csv.writer(f, lineterminator="\n").writerows(rows)
    if not only:
        tie_decomposition()
    print(f"[{time.perf_counter() - t_start:7.0f}s] wrote {RES}/*.csv", flush=True)


# -------------------------------------------------------------------- report
# validated reference categorical palette (dataviz skill), fixed order by model
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
# 8 hues for 8 plotted series: flexible plain learners keep their original slots,
# hybrids take the slots freed by the linear models and kNN (which are table-only).
COLOR = {"lookup table": PALETTE[0], "hybrid: tie-break LUT": PALETTE[1],
         "hybrid: prior LUT": PALETTE[2], "hybrid: HGB + STANDARD LLR": PALETTE[3],
         "MLP 64x64": PALETTE[4], "random forest": PALETTE[5],
         "hist grad boost": PALETTE[6], "hybrid: residual XGB": PALETTE[7]}
CURVE_MODELS = list(COLOR)
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
        "savefig.facecolor": SURF, "axes.grid": True, "grid.alpha": 0.25,
        "grid.linewidth": 0.6, "legend.frameon": False})
    return plt


def aggregate(runs):
    """(point, model, N) -> (mean, sd, ci95 half-width, n, mean fit seconds)."""
    from scipy.stats import t as student_t
    g = {}
    for r in runs:
        g.setdefault((r["point"], r["model"], int(r["N_train"])), []).append(
            (float(r["LER"]), float(r["fit_seconds"])))
    agg = {}
    for k, v in g.items():
        a = np.array(v); n = len(a); sd = a[:, 0].std(ddof=1) if n > 1 else 0.0
        ci = student_t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
        agg[k] = (a[:, 0].mean(), sd, ci, n, a[:, 1].mean())
    return agg


def n_to_close(Ns, mu, ls, lo, frac=0.9):
    """Smallest N (log-interpolated) where the mean curve closes `frac` of the gap."""
    if ls - lo < 1e-6:
        return None
    target = ls - frac * (ls - lo)
    for i, (N, m) in enumerate(zip(Ns, mu)):
        if m <= target:
            if i == 0:
                return N
            f = (mu[i - 1] - target) / (mu[i - 1] - m)
            return float(np.exp(np.log(Ns[i - 1]) + f * (np.log(N) - np.log(Ns[i - 1]))))
    return float("inf")


def fig_learning_curve(plt, name, ref, agg, color, models=None, tag=""):
    ls, lsb, lo = ref
    floor = 1e-6
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for m in (models or CURVE_MODELS):
        ks = sorted(k for k in agg if k[0] == name and k[1] == m)
        if not ks:
            continue
        N = np.array([k[2] for k in ks]); mu = np.array([agg[k][0] for k in ks])
        ci = np.array([agg[k][2] for k in ks])
        ex = np.clip(mu - lo, floor, None)
        ax.plot(N, ex, "-o", color=color[m], lw=2, ms=4.5, mec=SURF, mew=1, label=m)
        ax.fill_between(N, np.clip(mu - ci - lo, floor, None),
                        np.clip(mu + ci - lo, floor, None),
                        color=color[m], alpha=0.15, lw=0)
    if ls - lo > 1e-6:
        ax.axhline(ls - lo, color=INK, lw=1.4, ls="--")
        ax.text(N[0], (ls - lo) * 1.08, "STANDARD (random tie-break)", fontsize=9, color=INK)
        if lsb - lo > floor:
            ax.axhline(lsb - lo, color=INK2, lw=1.2, ls="-.")
            ax.text(N[0], (lsb - lo) * 1.08, "STANDARD + best tie-break", fontsize=9,
                    color=INK2)
    else:
        ax.text(N[0], floor * 1.5, "STANDARD = ORACLE here (no gap to close)",
                fontsize=9, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(floor * 0.8, None)
    ax.set_xlabel("training shots N")
    ax.set_ylabel("excess LER over ORACLE  (LER − LER_oracle)")
    ax.set_title(f"{name}: learning curves  (ORACLE LER = {lo:.4f}; "
                 f"band = 95% CI of mean)", color=INK, fontsize=11, loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              title=None if models else "linear models & kNN: see table", title_fontsize=8)
    fig.tight_layout()
    path = f"{FIG}/curve_{tag}{_slug(name)}.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_gap_closed(plt, refs, agg, Nmax):
    pts = [r for r in refs if float(r["LER_standard"]) - float(r["LER_oracle"]) > 1e-6]
    markers = ["o", "s", "D", "^"]; shades = [INK, INK2, "#8a897f", "#b5b4ab"]
    names = [m for m in MODEL_NAMES if (pts[0]["point"], m, Nmax) in agg]
    fig, ax = plt.subplots(figsize=(9, 6.6))
    y = np.arange(len(names))
    for j, r in enumerate(pts):
        ls, lo = float(r["LER_standard"]), float(r["LER_oracle"])
        g, e = [], []
        for m in names:
            mu, sd, ci, *_ = agg[(r["point"], m, Nmax)]
            g.append(100 * (ls - mu) / (ls - lo)); e.append(100 * ci / (ls - lo))
        off = (j - (len(pts) - 1) / 2) * 0.2
        ax.errorbar(g, y + off, xerr=e, fmt=markers[j], color=shades[j], ms=7,
                    mec=SURF, mew=1.2, capsize=2, label=r["point"])
    ax.axvline(100, color=INK, lw=1.2, ls=":"); ax.axvline(0, color=INK, lw=1.2, ls="--")
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlim(max(-120, ax.get_xlim()[0]), 112)
    ax.text(100, -0.7, "ORACLE", fontsize=9, color=INK, ha="center")
    ax.text(0, -0.7, "STANDARD", fontsize=9, color=INK, ha="center")
    ax.set_xlabel(f"% of STANDARD→ORACLE gap closed at N = {Nmax:,}  (±95% CI)")
    ax.set_title("How much of the available gain each model captures", color=INK,
                 fontsize=11, loc="left", pad=18)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    path = f"{FIG}/gap_closed.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_speed(plt, speed, color):
    rows = sorted(speed, key=lambda r: float(r["shots_per_sec"]))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, r in enumerate(rows):
        v = float(r["shots_per_sec"])
        ax.barh(i, v, color=color.get(r["model"], INK2), height=0.6)
        ax.text(v * 1.15, i, f"{v:.2g}/s", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)),
                  [r["model"] + ("" if r["form"].startswith("native") else f"  [{r['form']}]")
                   for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(None, max(float(r["shots_per_sec"]) for r in rows) * 8)
    ax.set_xlabel("inference throughput (shots/s, single process, batch of 10^6)")
    ax.set_title("Decode throughput", color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = f"{FIG}/speed.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fig_ties(plt, ties):
    sel = [r for r in ties if float(r["p_loss"]) == 0.08 and float(r["p_pauli"]) == 0.01]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for k, e in enumerate((0.5, 0.7, 1.0)):
        rr = sorted((r for r in sel if float(r["eta"]) == e), key=lambda r: float(r["c"]))
        c = [float(r["c"]) for r in rr]
        ax.plot(c, [float(r["gain_total"]) for r in rr], "-o", color=PALETTE[k], lw=2, ms=5,
                label=f"η={e}: ORACLE vs STANDARD")
        ax.plot(c, [float(r["gain_ties_only"]) for r in rr], "--", color=PALETTE[k], lw=1.6,
                label=f"η={e}: tie-break only")
    ax.axhline(1, color=INK2, lw=1)
    ax.set_xlabel("correlation strength c"); ax.set_ylabel("gain = LER_STANDARD / LER")
    ax.set_title("Where the gain comes from (p_loss=0.08, p_pauli=0.01)", color=INK,
                 fontsize=11, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path = f"{FIG}/ties_gain.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def report():
    plt = _style()
    os.makedirs(FIG, exist_ok=True)
    refs, runs = _read(f"{RES}/refs.csv"), _read(f"{RES}/runs.csv")
    speed, ties = _read(f"{RES}/speed.csv"), _read(f"{RES}/ties.csv")
    agg = aggregate(runs)
    color = dict(COLOR)
    Ns = sorted({k[2] for k in agg}); Nmax = Ns[-1]
    seeds = max(v[3] for v in agg.values())

    md = []
    w = md.append
    w("# Model zoo vs standard erasure-aware decoding — d=3 color code\n")
    w(f"_Generated by `model_zoo.py` on {time.strftime('%Y-%m-%d')}. "
      f"{len(MODEL_NAMES)} learners × {len(Ns)} training sizes × {seeds} seeds × "
      f"{len(refs)} noise points = {len(runs):,} trained models._\n")
    w("<!-- FINDINGS -->\n")
    w("## Setup\n")
    w("- **Code / noise:** [[7,1,3]] color code, phenomenological toy from "
      "`heralding_correlation_sweep.py`: per-qubit loss (heralded w.p. η) + Pauli, plus "
      "correlated leakage (a lost qubit flips its fixed partner w.p. c).")
    w("- **Decoder input:** 3 syndrome bits + 7 herald bits (1024 possible inputs). "
      "**Output:** logical class relative to the canonical correction.")
    w("- **Exact scoring:** the noise distribution factorizes into 16,384 atoms, so every "
      "LER below is computed *exactly*, with no Monte-Carlo noise in evaluation. Error bars "
      "reflect only training-set randomness (95% CI of the mean over seeds; all models at a "
      "given (seed, N) share the same training set).")
    w("- **References:** *STANDARD* = exact ML with independent per-qubit priors given "
      "heralds. Its independent model has **exact ties** on some inputs; the honest baseline "
      "breaks them at random (expected LER). *STANDARD + best tie-break* breaks every tie "
      "the right way (what tie-breaking alone can buy). *ORACLE* = Bayes-optimal given the "
      "observables, the ceiling for any decoder.\n")

    w("## Reference decoders (exact)\n")
    w("| point | η | c | STANDARD | STANDARD + best tie | ORACLE | ceiling gain | "
      "share of gain from ties | tied inputs (prob. mass) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in refs:
        ls, lsb, lo = (float(r[k]) for k in
                       ("LER_standard", "LER_standard_best_tie", "LER_oracle"))
        share = f"{100 * (ls - lsb) / (ls - lo):.0f}%" if ls - lo > 1e-6 else "—"
        w(f"| {r['point']} | {r['eta']} | {r['c']} | {ls:.4f} | {lsb:.4f} | {lo:.4f} | "
          f"{ls / lo:.2f}× | {share} | {r['tied_inputs']} ({float(r['tied_mass']):.3f}) |")
    w("")

    w("## Where the gain comes from\n")
    w(f"![tie decomposition]({fig_ties(plt, ties)})\n")
    w("Solid lines: full ORACLE gain over STANDARD. Dashed: gain from fixing STANDARD's "
      "tie-breaks only, without overriding any decision it is confident about. Where solid "
      "and dashed coincide, *all* of the value of modeling correlation is in resolving "
      "inputs the independent model is exactly indifferent about.\n")
    w("Share of the gain explained by tie-breaking at η=0.7, across the parameter grid "
      "(full grid in `results/ties.csv`):\n")
    cs = ("0.2", "0.4", "0.6", "0.8")
    w("| p_loss | p_pauli | " + " | ".join(f"c={c}" for c in cs) + " |")
    w("|---|---|" + "---|" * len(cs))
    for pl in ("0.04", "0.08", "0.15"):
        for pp in ("0.003", "0.01", "0.03"):
            cells = []
            for c in cs:
                rr = [r for r in ties if r["p_loss"] == pl and r["p_pauli"] == pp
                      and float(r["eta"]) == 0.7 and float(r["c"]) == float(c)]
                cells.append(f"{100 * float(rr[0]['frac_gain_from_ties']):.0f}%" if rr else "")
            w(f"| {pl} | {pp} | " + " | ".join(cells) + " |")
    w("")

    w("## Learning curves\n")
    w("Y-axis is **excess LER over the ORACLE** on a log scale, so every curve's distance to "
      "the Bayes ceiling is visible, including models far worse than STANDARD. "
      "Points pinned at the bottom have learned the ORACLE exactly (excess ≤ 1e-6).\n")
    for r in refs:
        name = r["point"]
        ref = tuple(float(r[k]) for k in
                    ("LER_standard", "LER_standard_best_tie", "LER_oracle"))
        ls, lsb, lo = ref
        w(f"### {name}\n")
        w(f"![{name}]({fig_learning_curve(plt, name, ref, agg, color)})\n")
        w(f"| model | LER at N={Nmax:,} (±95% CI) | gain vs STANDARD | gap closed | "
          f"N to close 90% of gap | best N | mean fit time at N={Nmax:,} |")
        w("|---|---|---|---|---|---|---|")
        for m in MODEL_NAMES:
            if (name, m, Nmax) not in agg:
                continue
            ks = sorted(k for k in agg if k[0] == name and k[1] == m)
            N = [k[2] for k in ks]; mu = [agg[k][0] for k in ks]
            mu_max, _, ci, _, ft = agg[(name, m, Nmax)]
            gap = f"{100 * (ls - mu_max) / (ls - lo):.1f}%" if ls - lo > 1e-6 else "—"
            n90 = n_to_close(N, mu, ls, lo)
            n90s = "—" if n90 is None else ("not reached" if n90 == float("inf")
                                            else f"{n90:,.0f}")
            best = N[int(np.argmin(mu))]
            w(f"| {m} | {mu_max:.5f} ± {ci:.5f} | {ls / mu_max:.3f}× | {gap} | {n90s} | "
              f"{best:,} | {ft:.2f}s |")
        w("")

    w("## Hybrid decoders\n")
    w("Hybrids start from STANDARD and learn only where it falls short. "
      "*tie-break LUT* keeps STANDARD wherever it has a preference and learns a majority "
      "vote only on its exactly-tied inputs (ceiling = STANDARD + best tie-break). "
      "*prior LUT* is a Bayesian table with STANDARD's soft posterior as a prior worth "
      "α=10 pseudo-shots per input. *HGB + STANDARD LLR* is gradient boosting given "
      "STANDARD's log-likelihood ratio as an extra feature; it is the only hybrid that "
      "needs no table, so it is the one that carries over to d=5,7.\n")
    w("Percent of the STANDARD→ORACLE gap closed vs. training shots (mean over seeds; "
      "negative = worse than STANDARD):\n")
    cols = [min(Ns, key=lambda n: abs(np.log(n / t))) for t in (100, 1e3, 1e4, 1e5, 3e5)]
    show = HYBRID + ["lookup table", "random forest", "hist grad boost"]
    for r in refs:
        ls, lo = float(r["LER_standard"]), float(r["LER_oracle"])
        if ls - lo < 1e-6:
            continue
        w(f"**{r['point']}**\n")
        w("| model | " + " | ".join(f"N={n:,}" for n in cols) + " |")
        w("|---|" + "---|" * len(cols))
        for m in show:
            if (r["point"], m, Nmax) not in agg:
                continue
            w(f"| {m} | " + " | ".join(
                f"{100 * (ls - agg[(r['point'], m, n)][0]) / (ls - lo):.1f}%" for n in cols) + " |")
        w("")

    if len(refs) > 1:
        w("## Gap closed, all correlated points\n")
        w(f"![gap closed]({fig_gap_closed(plt, refs, agg, Nmax)})\n")

    w("## Speed\n")
    w(f"![speed]({fig_speed(plt, speed, color)})\n")
    w("| decoder | form | throughput (shots/s) | fit time at N_max (single process) |")
    w("|---|---|---|---|")
    for r in sorted(speed, key=lambda r: -float(r["shots_per_sec"])):
        w(f"| {r['model']} | {r['form']} | {float(r['shots_per_sec']):.3g} | "
          f"{float(r['fit_seconds_at_Nmax']):.2f}s |")
    w("\nAt d=3 the input space is only 1024 points, so **every** decoder here, learned or "
      "not, can be compiled to a lookup table and run at the LUT rate. Native-form speed "
      "differences are an implementation detail at d=3; they start to matter only at larger "
      "distance, where the input space is too big to tabulate.\n")

    w("## Reproduce\n")
    w("```\n.venv/Scripts/python model_zoo.py                # full run + report\n"
      ".venv/Scripts/python model_zoo.py --only hybrid  # rerun just the hybrids, merge\n"
      ".venv/Scripts/python model_zoo.py --report-only  # rebuild figures + this doc\n```\n")
    text = "\n".join(md)
    try:   # keep hand-written findings across regenerations
        with open("model_zoo_findings.md", encoding="utf-8") as f:
            text = text.replace("<!-- FINDINGS -->", f.read())
    except FileNotFoundError:
        pass
    with open("model_zoo_report.md", "w", encoding="utf-8") as f:
        f.write(text)
    hybrid_report(plt, refs, agg, speed, Ns, seeds)
    print("wrote model_zoo_report.md, hybrid_report.md and figures/*.png")


HYBRID_COMPARE = HYBRID + ["lookup table", "hist grad boost"]   # hybrids + plain twins
ALPHAS = [0.3, 1, 3, 10, 30, 100, 300]


def alpha_sweep(Ns, seeds):
    """Prior strength for the prior-LUT hybrid. Same paired training sets as the
    zoo; the counts are computed once per (seed, N) and reused for every alpha."""
    rows = []
    for name, pl, pp, eta, c in POINTS:
        x = point_setup(pl, pp, eta, c)
        p_std = 1.0 / (1.0 + np.exp(-np.clip(x["llr"], -50, 50)))
        for N in Ns:
            for s in range(seeds):
                X, y = sample_train(x["prob"], x["idx"], x["rel"], N,
                                    np.random.default_rng([s, N]))
                C = _counts(X, y)
                for a in ALPHAS:
                    post = (C[:, 1] + a * p_std) / (C.sum(1) + a)
                    pred = np.where(np.abs(post - 0.5) < 1e-12, x["std"],
                                    (post > 0.5).astype(np.uint8)).astype(np.uint8)
                    rows.append((name, a, N, s, exact_ler(x["J"], pred)))
    with open(f"{RES}/alpha_sweep.csv", "w") as f:
        f.write("point,alpha,N_train,seed,LER\n")
        cw = csv.writer(f, lineterminator="\n")
        for r in rows:
            cw.writerow(list(r[:4]) + [f"{r[4]:.7g}"])
    print(f"wrote {RES}/alpha_sweep.csv ({len(rows):,} rows)")


def fig_alpha(plt, name, ls, lo, A, Ns):
    """% gap closed vs N, one line per alpha (sequential single-hue ramp)."""
    ramp = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for a, col in zip(ALPHAS, ramp):
        mu = [100 * (ls - A[(name, a, n)]) / (ls - lo) for n in Ns]
        ax.plot(Ns, mu, "-o", color=col, lw=2, ms=4, mec=SURF, mew=1, label=f"α = {a:g}")
    ax.axhline(100, color=INK, lw=1.2, ls=":")
    ax.set_xscale("log"); ax.set_ylim(40, 101)
    ax.set_xlabel("training shots N"); ax.set_ylabel("% of STANDARD→ORACLE gap closed")
    ax.set_title(f"{name}: prior LUT, effect of prior strength α (mean over seeds)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    path = f"{FIG}/alpha_{_slug(name)}.png"; fig.savefig(path, dpi=150); plt.close(fig)
    return path


def hybrid_report(plt, refs, agg, speed, Ns, seeds):
    """Standalone report on the hybrid decoders -> hybrid_report.md."""
    Nmax = Ns[-1]
    cols = [min(Ns, key=lambda n: abs(np.log(n / t))) for t in (100, 300, 1e3, 3e3, 1e4, 1e5, 3e5)]
    md = []; w = md.append
    w("# Hybrid decoders — d=3 color code\n")
    w(f"_Generated by `model_zoo.py` on {time.strftime('%Y-%m-%d')}. Companion to "
      f"[model_zoo_report.md](model_zoo_report.md), which has the full setup and the plain "
      f"learners. {len(HYBRID)} hybrids × {len(Ns)} training sizes × {seeds} seeds × "
      f"{len(refs)} noise points, exact LER, paired training sets._\n")
    w("<!-- HYBRID FINDINGS -->\n")

    w("## The four hybrids\n")
    w("Each hybrid starts from STANDARD (exact ML decoding under an independent-noise model "
      "that uses the heralds) and learns only where STANDARD falls short.\n")
    w("| hybrid | what it learns | data can override STANDARD… | limit with infinite data | "
      "needs a table? |")
    w("|---|---|---|---|---|")
    w("| **tie-break LUT** | majority vote per input, *only* on inputs where STANDARD is "
      "exactly tied | never (ties only) | STANDARD + best tie-break | yes |")
    w("| **prior LUT** (α=10) | posterior (n₁ + α·p_STD) / (n + α) per input | anywhere, "
      "once counts outweigh α pseudo-shots | ORACLE | yes |")
    w("| **HGB + STANDARD LLR** | gradient boosting on the 10 input bits plus STANDARD's "
      "log-likelihood ratio | anywhere | ORACLE (in principle) | **no** |")
    w("| **residual XGB** | boosted trees *starting from* STANDARD's LLR as the initial "
      "margin; regularized (depth 4, min_child_weight 5, λ=5) | anywhere, once trees "
      "accumulate enough evidence | ORACLE (in principle; slowed by regularization) | **no** |")
    w("")
    w("Both table hybrids were checked against the exact distribution before the run. "
      "Given exact counts, tie-break LUT lands exactly on the STANDARD + best tie-break LER "
      "and prior LUT exactly on the ORACLE LER.\n")
    w("The plain twins used for comparison are the **lookup table** (the same as prior LUT "
      "with α→0, falling back to STANDARD only on unseen inputs) and **hist grad boost** "
      "(HGB + STANDARD LLR without the LLR feature).\n")

    w("## Who wins at each training size\n")
    w("Lowest mean LER among **all 11** decoders (8 plain + 3 hybrid) at each training "
      "size. ✱ marks a hybrid.\n")
    w("| point | " + " | ".join(f"N={n:,}" for n in cols) + " |")
    w("|---|" + "---|" * len(cols))
    for r in refs:
        cells = []
        for n in cols:
            cand = [(agg[(r["point"], m, n)][0], m) for m in MODEL_NAMES
                    if (r["point"], m, n) in agg]
            lo_v = min(cand)[0]
            best = [m for v, m in cand if v - lo_v < 1e-9]       # show exact co-winners
            cells.append(" = ".join(("✱ " if b in HYBRID else "") + b.replace("hybrid: ", "")
                                    for b in best))
        w(f"| {r['point']} | " + " | ".join(cells) + " |")
    w("")

    for r in refs:
        name = r["point"]
        ref = tuple(float(r[k]) for k in
                    ("LER_standard", "LER_standard_best_tie", "LER_oracle"))
        ls, lsb, lo = ref
        w(f"## {name}\n")
        w(f"η = {r['eta']}, c = {r['c']}. STANDARD {ls:.4f} · STANDARD + best tie-break "
          f"{lsb:.4f} · ORACLE {lo:.4f}.\n")
        w(f"![{name}]({fig_learning_curve(plt, name, ref, agg, COLOR, HYBRID_COMPARE, 'hybrid_')})\n")
        if ls - lo > 1e-6:
            w("% of the STANDARD→ORACLE gap closed (mean ± 95% CI over seeds; negative = "
              "worse than STANDARD):\n")
            w("| model | " + " | ".join(f"N={n:,}" for n in cols) + " |")
            w("|---|" + "---|" * len(cols))
            for m in HYBRID_COMPARE:
                cells = []
                for n in cols:
                    mu, _, ci, *_ = agg[(name, m, n)]
                    cells.append(f"{100 * (ls - mu) / (ls - lo):.1f} ± {100 * ci / (ls - lo):.1f}")
                w(f"| {m} | " + " | ".join(cells) + " |")
            tie_cap = 100 * (ls - lsb) / (ls - lo)
            w(f"\nTie-break LUT's ceiling here is **{tie_cap:.1f}%** of the gap, the share "
              "explained by STANDARD's ties.\n")
        else:
            w("No gap to close here (STANDARD = ORACLE). This point tests whether a "
              "hybrid *does harm* when there is no correlation to learn. Excess LER over "
              "ORACLE (×10⁻⁴):\n")
            w("| model | " + " | ".join(f"N={n:,}" for n in cols) + " |")
            w("|---|" + "---|" * len(cols))
            for m in HYBRID_COMPARE:
                w(f"| {m} | " + " | ".join(
                    f"{max(0.0, 1e4 * (agg[(name, m, n)][0] - lo)):.2f}" for n in cols) + " |")
            w("")

    if os.path.exists(f"{RES}/alpha_sweep.csv"):
        A = {}
        for row in _read(f"{RES}/alpha_sweep.csv"):
            A.setdefault((row["point"], float(row["alpha"]), int(row["N_train"])), []).append(
                float(row["LER"]))
        A = {k: float(np.mean(v)) for k, v in A.items()}
        w("## Prior strength α (prior LUT)\n")
        w("α is how many pseudo-shots STANDARD's opinion is worth at each input. Small α "
          "trusts the data sooner; large α holds onto STANDARD longer. Same paired training "
          "sets as above; % of gap closed, mean over seeds.\n")
        for r in refs:
            ls, lo = float(r["LER_standard"]), float(r["LER_oracle"])
            if ls - lo < 1e-6:
                continue
            w(f"**{r['point']}**\n")
            w("| α | " + " | ".join(f"N={n:,}" for n in cols) + " |")
            w("|---|" + "---|" * len(cols))
            best = {n: min(ALPHAS, key=lambda a: A[(r["point"], a, n)]) for n in cols}
            for a in ALPHAS:
                w(f"| {a:g} | " + " | ".join(
                    ("**" if best[n] == a else "") +
                    f"{100 * (ls - A[(r['point'], a, n)]) / (ls - lo):.1f}" +
                    ("**" if best[n] == a else "") for n in cols) + " |")
            w("")
            if float(r["c"]) >= 0.6:
                w(f"![alpha sweep]({fig_alpha(plt, r['point'], ls, lo, A, Ns)})\n")
        w("Bold = best α at that training size.\n")

    w("## Speed\n")
    w("| decoder | throughput (shots/s) | fit time at N=300k |")
    w("|---|---|---|")
    for r in sorted(speed, key=lambda r: -float(r["shots_per_sec"])):
        if r["model"] in HYBRID_COMPARE or r["form"] != "native sklearn/numpy":
            label = r["model"] + ("" if r["form"].startswith("native") else f" [{r['form']}]")
            w(f"| {label} | {float(r['shots_per_sec']):.3g} | "
              f"{float(r['fit_seconds_at_Nmax']):.2f}s |")
    w("\nThe table hybrids decode at lookup-table speed. HGB + STANDARD LLR is measured "
      "with STANDARD's LLR precomputed as a 1024-entry table. At d≥5 the LLR is not "
      "tabulable, so each shot also pays for one STANDARD decode.\n")

    w("## Reproduce\n")
    w("```\n.venv/Scripts/python model_zoo.py --only hybrid  # rerun hybrids (~2 min), merge\n"
      ".venv/Scripts/python model_zoo.py --report-only  # rebuild both reports\n```\n")
    text = "\n".join(md)
    try:
        with open("hybrid_findings.md", encoding="utf-8") as f:
            text = text.replace("<!-- HYBRID FINDINGS -->", f.read())
    except FileNotFoundError:
        pass
    with open("hybrid_report.md", "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run for smoke testing")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only", default=None,
                    help="comma-separated model names to (re)run and merge into results/, "
                         "or 'hybrid' for all hybrids")
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="run only the prior-LUT alpha sweep (~1 min), then report")
    a = ap.parse_args()
    if a.alpha_sweep:
        Ns = sorted({int(round(x, -1 if x < 1e3 else -2))
                     for x in np.logspace(2, np.log10(3e5), 15)})
        alpha_sweep(Ns, a.seeds)
    elif not a.report_only:
        if a.quick:
            Ns, seeds, n_speed = [300, 3_000, 30_000], 2, 100_000
        else:
            # quarter-decade grid, 100 .. 300k
            Ns = sorted({int(round(x, -1 if x < 1e3 else -2))
                         for x in np.logspace(2, np.log10(3e5), 15)})
            seeds, n_speed = a.seeds, 1_000_000
        only = (HYBRID if a.only == "hybrid" else a.only.split(",")) if a.only else None
        compute(Ns, seeds, n_speed, only)
    report()


if __name__ == "__main__":
    main()
