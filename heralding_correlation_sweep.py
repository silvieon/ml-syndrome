#!/usr/bin/env python3
"""
3D sweep: logical error rate over (heralding efficiency eta) x (correlation
strength c), comparing a STANDARD erasure-aware decoder against a
CORRELATION-AWARE decoder on the distance-3 color code ([[7,1,3]] Steane).

This is the toy that isolates the thesis's actual claim. The earlier demo showed
that on INDEPENDENT noise a smarter decoder gains ~nothing over standard
erasure-aware decoding. Here we add a tunable correlated-leakage mechanism and
ask: once errors are correlated, how much does MODELING the correlation buy you,
and how does that depend on heralding quality?

Model (exact, factorized over a fixed matching of the 7 qubits):
  qubits are matched into pairs (0,1) (2,3) (4,5) with 6 a singleton -- a stand-in
  for gate-induced leakage propagation between gate partners.
  Per qubit: loss w.p. p_loss (heralded w.p. eta); a lost qubit takes a random
  Pauli (bit=1 w.p. 0.5), else an unheralded bit-flip w.p. p_pauli.
  Correlation: when a qubit is lost, it flips its PARTNER w.p. c -- even if the
  partner was not itself lost. That cross-term is what standard erasure decoding,
  which assumes independence, cannot see -- and it persists even at eta=1.

Decoders:
  STANDARD   : exact ML with per-qubit independent priors (heralded -> 0.5).
               Its independent model has EXACT TIES between logical classes on
               many inputs; we report it with RANDOM tie-breaking (its expected
               LER), the only implementation-independent choice.
  STANDARD + best tie-break : every tie broken the right way -- what fixing
               STANDARD's ties alone can buy.
  ORACLE     : Bayes-optimal given (syndrome, heralds) under the true correlated
               model (= the old CORR-AWARE decoder).

LERs are EXACT: the noise factorizes into 16,384 atoms, enumerated via the
machinery in model_zoo.py -- no Monte-Carlo noise. The gain is split into
  gain_total = LER_std / LER_oracle
  share_from_ties = (LER_std - LER_std_best_tie) / (LER_std - LER_oracle)

Outputs: heralding_correlation_sweep.png and .csv.
  python heralding_correlation_sweep.py             # exact 15x15 sweep
  python heralding_correlation_sweep.py --mc-check  # MC cross-check of one point
"""

import argparse

import itertools
import numpy as np

H = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
], dtype=np.uint8)
N_Q, N_S = 7, 3
PAIRS = [(0, 1), (2, 3), (4, 5)]
SINGLE = 6


def _kernel(M):
    out = []
    for x in range(2 ** M.shape[1]):
        e = np.array([(x >> i) & 1 for i in range(M.shape[1])], dtype=np.uint8)
        if not (M @ e % 2).any():
            out.append(e)
    return np.array(out, dtype=np.uint8)


def _rowspace(M):
    sp = {tuple(np.zeros(M.shape[1], np.uint8))}
    for r in M:
        sp |= {tuple((np.array(v) ^ r) % 2) for v in sp}
    return sp


KER = _kernel(H)
L = sorted((e for e in KER if tuple(e) not in _rowspace(H)),
           key=lambda e: int(e.sum()))[0]
E0 = {}
for _x in range(2 ** N_Q):
    _e = np.array([(_x >> i) & 1 for i in range(N_Q)], dtype=np.uint8)
    E0.setdefault(tuple(H @ _e % 2), _e)


def pair_table(p_loss, p_pauli, eta, c):
    """Exact joint P(ha,hb,ba,bb) for a correlated pair, by enumerating latents."""
    T = np.zeros((2, 2, 2, 2))  # [ha, hb, ba, bb]
    # per-qubit latent: (state, own_bit) with state in {NOLOSS, LOSS_H, LOSS_S}
    states = [("N", 1 - p_loss), ("H", p_loss * eta), ("S", p_loss * (1 - eta))]
    for (sa, pa), (sb, pb) in itertools.product(states, states):
        la, lb = sa != "N", sb != "N"
        ha, hb = int(sa == "H"), int(sb == "H")
        for own_a, own_b in itertools.product((0, 1), (0, 1)):
            pA = 0.5 if la else (p_pauli if own_a == 1 else 1 - p_pauli)
            pB = 0.5 if lb else (p_pauli if own_b == 1 else 1 - p_pauli)
            # propagation coins: a->b and b->a, each active only if source lost
            for prop_ab, prop_ba in itertools.product((0, 1), (0, 1)):
                pr = pa * pb * pA * pB
                pr *= (c if prop_ab else (1 - c)) if la else (1 if prop_ab == 0 else 0)
                pr *= (c if prop_ba else (1 - c)) if lb else (1 if prop_ba == 0 else 0)
                if pr == 0:
                    continue
                ba = own_a ^ (prop_ba if lb else 0)
                bb = own_b ^ (prop_ab if la else 0)
                T[ha, hb, ba, bb] += pr
    return T


def single_table(p_loss, p_pauli, eta):
    S = np.zeros((2, 2))  # [h, b]
    for st, ps in [("N", 1 - p_loss), ("H", p_loss * eta), ("S", p_loss * (1 - eta))]:
        lost = st != "N"
        h = int(st == "H")
        for b in (0, 1):
            pb = 0.5 if lost else (p_pauli if b == 1 else 1 - p_pauli)
            S[h, b] += ps * pb
    return S


def sample_pairs(T, S, n, rng):
    bit = np.zeros((n, N_Q), np.uint8)
    her = np.zeros((n, N_Q), np.uint8)
    flatT = T.reshape(-1) / T.sum()
    idxT = rng.choice(16, size=(n, len(PAIRS)), p=flatT)
    for k, (qa, qb) in enumerate(PAIRS):
        ha, hb, ba, bb = np.unravel_index(idxT[:, k], (2, 2, 2, 2))
        her[:, qa], her[:, qb] = ha, hb
        bit[:, qa], bit[:, qb] = ba, bb
    flatS = S.reshape(-1) / S.sum()
    idxS = rng.choice(4, size=n, p=flatS)
    hS, bS = np.unravel_index(idxS, (2, 2))
    her[:, SINGLE], bit[:, SINGLE] = hS, bS
    return bit, her


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return m.squeeze(axis) + np.log(np.exp(a - m).sum(axis=axis))


def ler_point(p_loss, p_pauli, eta, c, n, rng):
    T, S = pair_table(p_loss, p_pauli, eta, c), single_table(p_loss, p_pauli, eta)
    bit, her = sample_pairs(T, S, n, rng)
    syn = bit @ H.T % 2
    true_log = bit @ L % 2
    e0 = np.stack([E0[tuple(s)] for s in syn])
    cand = e0[:, None, :] ^ KER[None, :, :]          # (n,16,7)
    cand_log = cand @ L % 2
    eps = 1e-12

    # STANDARD: per-qubit marginal conditioned on own herald
    def marg_from_pair(T):
        Ph = T.sum(axis=(1, 3))           # over partner -> [ha, ba]  P(ha,ba)
        M = np.zeros((2, 2))
        for h in (0, 1):
            z = Ph[h].sum()
            M[h] = Ph[h] / (z + eps)       # P(ba | ha)
        return M
    Mpair = marg_from_pair(T)
    Msing = S / (S.sum(axis=1, keepdims=True) + eps)
    LOGM = np.stack([Mpair, Mpair, Mpair, Mpair, Mpair, Mpair, Msing])  # (7,2,2)
    LOGM = np.log(LOGM + eps)
    std = np.zeros((n, cand.shape[1]))
    for i in range(N_Q):
        std += LOGM[i][her[:, i][:, None], cand[:, :, i]]

    # CORR-AWARE: true pair-joint conditional P(ba,bb | ha,hb)
    denom = T.sum(axis=(2, 3), keepdims=True) + eps
    Ccond = np.log(T / denom + eps)        # [ha,hb,ba,bb]
    Scond = np.log(Msing + eps)
    corr = np.zeros((n, cand.shape[1]))
    for (qa, qb) in PAIRS:
        corr += Ccond[her[:, qa][:, None], her[:, qb][:, None],
                      cand[:, :, qa], cand[:, :, qb]]
    corr += Scond[her[:, SINGLE][:, None], cand[:, :, SINGLE]]

    def ler(score):
        neg = np.where(cand_log == 0, score, -np.inf)
        pos = np.where(cand_log == 1, score, -np.inf)
        pred = (_logsumexp(pos, 1) > _logsumexp(neg, 1)).astype(np.uint8)
        return float(np.mean(pred != true_log))

    return ler(std), ler(corr)


def exact_point(p_loss, p_pauli, eta, c):
    """Exact (LER_std_random_tie, LER_std_best_tie, LER_std_worst_tie, LER_oracle)."""
    import model_zoo as mz   # lazy: model_zoo imports this module at top level
    x = mz.point_setup(p_loss, p_pauli, eta, c)
    J, tie = x["J"], x["tie"]
    ls, lsb = mz.standard_lers(J, x["std"], tie)
    wrong = (J[np.arange(mz.N_IN), 1 - x["std"]] * ~tie).sum()
    return ls, lsb, float(wrong + J[tie].max(axis=1).sum()), mz.exact_ler(J, x["oracle"])


def mc_check(p_loss=0.08, p_pauli=0.01, eta=0.7, c=0.4, n=1_000_000):
    """Cross-check the exact numbers against the original Monte-Carlo decoders.
    The MC STANDARD breaks ties toward absolute class 0, so it must land inside
    the exact [best-tie, worst-tie] interval, not on the random-tie value."""
    ls, lsb, lsw, lo = exact_point(p_loss, p_pauli, eta, c)
    s, co = ler_point(p_loss, p_pauli, eta, c, n, np.random.default_rng(1))
    se = lambda q: (q * (1 - q) / n) ** 0.5
    print(f"point p_loss={p_loss} p_pauli={p_pauli} eta={eta} c={c}, MC n={n:,}")
    print(f"  ORACLE   exact {lo:.5f} | MC corr-aware {co:.5f} +- {se(co):.5f}")
    print(f"  STANDARD exact random-tie {ls:.5f}, tie-break interval [{lsb:.5f}, {lsw:.5f}]"
          f" | MC (ties -> class 0) {s:.5f} +- {se(s):.5f}")
    ok = abs(co - lo) < 4 * se(co) and lsb - 4 * se(s) <= s <= lsw + 4 * se(s)
    print("  check", "PASSED" if ok else "FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-check", action="store_true")
    ap.add_argument("--grid", type=int, default=15)
    a = ap.parse_args()
    if a.mc_check:
        return mc_check()

    p_loss, p_pauli = 0.08, 0.01
    etas = np.linspace(0.5, 1.0, a.grid)
    cs = np.linspace(0.0, 0.8, a.grid)
    shape = (len(cs), len(etas))
    STD, BEST, ORC = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    rows = []
    for i, c in enumerate(cs):
        for j, eta in enumerate(etas):
            ls, lsb, _, lo = exact_point(p_loss, p_pauli, eta, c)
            STD[i, j], BEST[i, j], ORC[i, j] = ls, lsb, lo
            share = (ls - lsb) / (ls - lo) if ls - lo > 1e-9 else float("nan")
            rows.append((eta, c, ls, lsb, lo, ls / lo, ls / lsb, share))
    GAIN = STD / ORC
    SHARE = np.where(STD - ORC > 1e-9, (STD - BEST) / np.maximum(STD - ORC, 1e-12), np.nan)

    with open("heralding_correlation_sweep.csv", "w") as f:
        f.write("eta,c,LER_standard,LER_standard_best_tie,LER_oracle,"
                "gain_total,gain_ties_only,share_from_ties\n")
        for r in rows:
            f.write("{:.4f},{:.4f},{:.6g},{:.6g},{:.6g},{:.4f},{:.4f},{:.4f}\n".format(*r))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    # single-hue sequential ramp (dataviz reference blue, light -> dark)
    blues = LinearSegmentedColormap.from_list(
        "ref_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURF, "savefig.facecolor": SURF})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    ext = [etas[0], etas[-1], cs[0], cs[-1]]
    panels = [
        (GAIN, "A. Gain of the Bayes-optimal decoder over STANDARD",
         "gain = LER_STANDARD / LER_ORACLE", [1.25, 1.5, 2, 3, 4], "{:.2g}×"),
        (100 * SHARE, "B. Share of that gain from fixing STANDARD's ties",
         "% of gap explained by tie-breaking alone", [70, 80, 90, 99], "{:.0f}%"),
    ]
    for ax, (Z, title, cbl, levels, fmt) in zip(axes, panels):
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap=blues, extent=ext,
                       interpolation="nearest")
        # contour ink flips with the ramp so labels stay legible on light and dark cells
        lc = [INK if im.norm(v) < 0.45 else SURF for v in levels]
        cs_ = ax.contour(etas, cs, Z, levels=levels, colors=lc, linewidths=1.2)
        ax.clabel(cs_, fmt=lambda v, f=fmt: f.format(v), fontsize=8.5)
        if np.isnan(Z[0]).all():
            ax.text(np.mean(ext[:2]), cs[0], "c=0: no gap, share undefined", ha="center",
                    va="bottom", fontsize=8, color=INK2)
        ax.set_xlabel("heralding efficiency η"); ax.set_ylabel("correlation strength c")
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(cbl)
        cb.outline.set_edgecolor(INK2)
    fig.suptitle(f"d=3 color code, exact LER  |  p_loss={p_loss}, p_pauli={p_pauli}  |  "
                 f"STANDARD = independent-noise ML with random tie-breaking",
                 color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig("heralding_correlation_sweep.png", dpi=170)

    print("gain (LER_std / LER_oracle) | share of gain from ties; rows=c, cols=eta")
    sel = range(0, len(etas), max(1, len(etas) // 5))
    print("        " + "  ".join(f"eta={etas[j]:.2f}      " for j in sel))
    for i, c in enumerate(cs):
        print(f"c={c:.2f} | " + "  ".join(
            f"{GAIN[i, j]:5.2f}x {100 * SHARE[i, j]:5.1f}%" if np.isfinite(SHARE[i, j])
            else f"{GAIN[i, j]:5.2f}x    —  " for j in sel))
    print("\nwrote heralding_correlation_sweep.png and .csv")


if __name__ == "__main__":
    main()
