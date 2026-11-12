#!/usr/bin/env python3
"""
Thesis-in-miniature: how much does a BETTER loss-aware decoder buy you over the
STANDARD erasure-aware baseline, as heralding degrades?

Runs TODAY with numpy only. Smallest 2D color code = [[7,1,3]] Steane code
(the distance-3 triangular color code), one CSS sector, EXACT max-likelihood
decoding (brute force over the 16 syndrome-consistent patterns) so any gap is
information, not decoder-quality artifact.

Noise per qubit:
    * physical loss/leakage with prob p_loss  -> qubit becomes a random Pauli
      (bit=1 w.p. 0.5 in this sector)
    * of those losses, a fraction  eta  are HERALDED (flagged); 1-eta are silent
    * otherwise, unheralded Pauli bit-flip with prob p_pauli

Three decoders, same shots, differing only in the per-qubit prior P(bit=1):
    IGNORANT  : ignores heralds entirely -> marginal prior everywhere
    STANDARD  : heralded qubit -> 0.5 ; everything else -> p_pauli
                (what erasure-conversion + an off-the-shelf decoder does)
    BETTER    : heralded qubit -> 0.5 ; unheralded -> p_eff, the Bayes-correct
                background that accounts for MISSED (silent) losses

Why it matters for the thesis:
  - at eta = 1 (perfect heralding) STANDARD and BETTER must coincide: there is
    nothing a smarter decoder can know that the herald didn't already say.
  - as eta drops, missed losses masquerade as a raised background error rate;
    BETTER models that and should pull ahead of STANDARD. That gap -- small,
    and only opening under imperfect heralding -- is the HONEST niche the real
    thesis has to defend (on independent noise it stays small; correlated
    leakage at circuit level is where it should grow, which is Phase 2 work).
"""

import numpy as np

# ----- Steane / Hamming [7,4,3] parity-check matrix ------------------------
H = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
], dtype=np.uint8)
N_Q, N_S = H.shape[1], H.shape[0]


def gf2_rowspace(M):
    space = {tuple(np.zeros(M.shape[1], dtype=np.uint8))}
    for r in M:
        space |= {tuple((np.array(v) ^ r) % 2) for v in space}
    return space


def gf2_kernel(M):
    n = M.shape[1]
    out = []
    for x in range(2 ** n):
        e = np.array([(x >> i) & 1 for i in range(n)], dtype=np.uint8)
        if not (M @ e % 2).any():
            out.append(e)
    return np.array(out, dtype=np.uint8)


KER = gf2_kernel(H)                                   # 16 syndrome-consistent generators
ROWSPACE = gf2_rowspace(H)                            # 8 stabilizers
_logicals = sorted((e for e in KER if tuple(e) not in ROWSPACE), key=lambda e: int(e.sum()))
L = _logicals[0]                                      # weight-3 logical indicator


def _particular_solutions():
    sol = {}
    for x in range(2 ** N_Q):
        e = np.array([(x >> i) & 1 for i in range(N_Q)], dtype=np.uint8)
        s = tuple(H @ e % 2)
        sol.setdefault(s, e)
        if len(sol) == 2 ** N_S:
            break
    return sol


E0 = _particular_solutions()


def _logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return m.squeeze(axis) + np.log(np.exp(a - m).sum(axis=axis))


def simulate(p_loss, p_pauli, eta, n_shots, rng):
    """Return (LER_ignorant, LER_standard, LER_better) for one noise point."""
    lost = rng.random((n_shots, N_Q)) < p_loss
    heralded = lost & (rng.random((n_shots, N_Q)) < eta)
    bit = np.where(lost,
                   rng.random((n_shots, N_Q)) < 0.5,
                   rng.random((n_shots, N_Q)) < p_pauli).astype(np.uint8)

    syndrome = bit @ H.T % 2
    true_logical = bit @ L % 2

    e0 = np.stack([E0[tuple(s)] for s in syndrome])
    cand = e0[:, None, :] ^ KER[None, :, :]               # (shots, 16, 7)
    cand_logical = cand @ L % 2                            # (shots, 16)

    eps = 1e-12
    p_marg = p_loss * 0.5 + (1 - p_loss) * p_pauli
    # Bayes-correct background for an UNHERALDED qubit (accounts for missed losses):
    denom = 1.0 - p_loss * eta
    p_eff = (p_loss * (1 - eta) * 0.5 + (1 - p_loss) * p_pauli) / max(denom, eps)

    priors = {
        "ignorant": np.full((n_shots, N_Q), p_marg, dtype=float),
        "standard": np.where(heralded, 0.5, p_pauli),
        "better":   np.where(heralded, 0.5, p_eff),
    }

    def ler(pr):
        lp1, lp0 = np.log(pr + eps), np.log(1 - pr + eps)
        logp = (cand * lp1[:, None, :] + (1 - cand) * lp0[:, None, :]).sum(axis=2)
        neg = np.where(cand_logical == 0, logp, -np.inf)
        pos = np.where(cand_logical == 1, logp, -np.inf)
        pred = (_logsumexp(pos, 1) > _logsumexp(neg, 1)).astype(np.uint8)
        return float(np.mean(pred != true_logical))

    return ler(priors["ignorant"]), ler(priors["standard"]), ler(priors["better"])


def main():
    rng = np.random.default_rng(0xC010C0DE)
    n_shots, p_pauli = 60_000, 0.01
    print("[[7,1,3]] Steane color code | one CSS sector | EXACT ML decoding")
    print(f"p_pauli = {p_pauli} | shots/point = {n_shots:,}")
    print("columns: LER for IGNORANT / STANDARD erasure-aware / BETTER (Bayes background)\n")
    for eta in (1.0, 0.8, 0.5):
        print(f"=== heralding efficiency eta = {eta:.2f} "
              f"({'perfect' if eta == 1 else 'imperfect'}) ===")
        print(f"{'p_loss':>7} | {'ignorant':>9} | {'standard':>9} | {'better':>9} | {'better/std':>10}")
        print("-" * 58)
        for p_loss in (0.0, 0.02, 0.05, 0.10, 0.20):
            ig, st, be = simulate(p_loss, p_pauli, eta, n_shots, rng)
            ratio = (st / be) if be > 0 else float("inf")
            print(f"{p_loss:>7.2f} | {ig:>9.5f} | {st:>9.5f} | {be:>9.5f} | {ratio:>10.2f}")
        print()
    print("Read: STANDARD >> IGNORANT always (heralds are worth a lot -> use them).")
    print("BETTER == STANDARD at eta=1 (nothing left to model). As eta falls, BETTER")
    print("edges ahead -- that is the only gap the thesis gets to claim on independent")
    print("noise, and it is small. The case for a learned decoder therefore rests on")
    print("imperfect heralding + CORRELATED circuit-level leakage, which d=3 toy noise")
    print("cannot show. That is precisely what the stim pipeline is built to test.")


if __name__ == "__main__":
    main()
