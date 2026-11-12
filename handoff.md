# Project handoff — loss-aware color-code decoding (QEC thesis)

Context dump for continuing in a fresh Claude Code session. Everything below is
state as of 2026-11-11. Working dir: this folder (`qec_thesis/`).

## What this is

An undergrad-thesis-scale research project to pitch to **Prof. Steve Girvin**
(worked with him before — warm reintroduction, not a cold call). Deliverable to
him is a short write-up + preliminary results, **sent only once prelim results
are in hand**. Author: account holder (Yale, background in quantum info,
cybersecurity, cryptography, ML, agentic AI, distributed systems; formal
verification is the weakest area, so the project deliberately avoids it).

## Central research question (the sharpened one — use this framing)

> How much does a learned / correlation-aware decoder recover over **standard
> erasure-aware** decoding of a color code, as heralding efficiency degrades and
> loss mixes with Pauli and **correlated** leakage noise?

Baseline is deliberately **standard erasure-aware**, NOT loss-ignorant. Erasure
conversion is the whole point of neutral-atom / trapped-ion platforms, so beating
a loss-ignorant decoder is not a contribution. The contribution has to be the gap
over a decoder that already uses the heralds.

Platform framing: neutral-atom / trapped-ion (slow-clock, ms cycle), where
loss/leakage are converted to **heralded erasures**. The slow clock means latency
is NOT the binding constraint (you have ms), so the thesis competes on
**accuracy under realistic noise**, not speed.

## Scientific findings so far (honest)

1. **Heralds are worth a lot.** Standard erasure-aware ≫ loss-ignorant always.
   (steane_erasure_demo.py)
2. **On INDEPENDENT noise, a smarter decoder buys ~nothing over standard.**
   At heralding η=1 they're identical; even at η=0.5 the Bayes-optimal
   "better" decoder ties standard (ratio ≈ 1.0). So the contribution is NOT
   there. (steane_erasure_demo.py)
3. **Correlation is where the gap opens.** With a tunable correlated-leakage
   mechanism, gain = LER_standard / LER_corr rises with correlation strength c
   (≈1 at c=0 up to ~2–2.6× at high c), and critically **stays >1 even at η=1**
   (perfect heralding) because correlated leakage crosses the erased/clean
   boundary that standard erasure decoding assumes independent.
   (heralding_correlation_sweep.py) — fine structure is MC-noisy at 60k shots;
   trend is robust.

Takeaway for the write-up: the case for a learned/correlation-aware decoder rests
on **imperfect heralding + correlated circuit-level leakage**. Say this plainly;
it's the honest scope and it's what makes the circuit-level pipeline worth building.

## Honest scope / caveats (put these in the write-up)

- This is fault-tolerance INFRASTRUCTURE research. It is NOT a contribution to any
  quantum-attack-on-crypto timeline (the project started from a Bitcoin/Shor
  question, but slow-clock machines only do at-rest attacks and are years out —
  decouple entirely).
- Current results are a **d=3 phenomenological toy** (= [[7,1,3]] Steane) with a
  FIXED, invented qubit pairing standing in for gate-induced leakage propagation.
  It demonstrates the mechanism; it does NOT quantify the real color code. The
  honest version is the stim circuit-level pipeline (d=3,5,7) with a physical
  leakage-propagation model.
- A specific decoder has a short half-life; frame the lasting contribution as the
  **characterization curve** (gain vs η, loss:Pauli ratio, correlation, distance).

## Files

- `steane_erasure_demo.py` — numpy only, RUNS. Exact ML decoding of d=3 color
  code under heralded loss + Pauli. Three decoders (ignorant / standard / Bayes
  "better") with heralding η knob. Shows findings 1 & 2.
- `heralding_correlation_sweep.py` — numpy + matplotlib, RUNS. 3D sweep of
  LER over (η, correlation c); standard vs correlation-aware; writes
  `heralding_correlation_sweep.png` + `.csv`. Shows finding 3. NOTE: has a
  redundant `pA` line in `pair_table` to clean up (harmless). Turn shots up to
  ~1M/pt and grid to ~15×15 with multi-seed error bars before the figure goes
  to Girvin.
- `phase0_color_code.py` — stim + chromobius + sinter. Standard triangular
  color-code memory threshold under circuit-level depolarizing noise.
  **UNTESTED** (couldn't install stim here). `--selftest` is the real
  verification gate. Most likely snag: chromobius detector annotations on the
  generated circuit.
- `requirements.txt`, `README.md` — deps + the research framing and references.

## Environment constraints (important for Claude Code)

- Cloud sandbox has **no PyPI access** — pip installs fail (NameResolutionError
  on pypi.org). Installed: numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2,
  networkx 3.7, **sklearn 1.9.1**. MISSING: stim, chromobius, sinter, pymatching,
  **torch**, xgboost, lightgbm.
- So: the stim pipeline and any torch GNN can only run where PyPI/conda is
  reachable (local machine or an env with network). The numpy/sklearn toys run
  anywhere.

## Immediate next task (was in progress when this dump was requested)

Compare the standard decoder against a **varied sample of models optimizing
accuracy** — a model zoo that sees only (syndrome bits + herald bits) and
predicts the logical class, scored against the true correlated distribution:
- Reference (no training): STANDARD erasure-aware; CORR-AWARE oracle = Bayes
  ceiling given the observables.
- Learners (sklearn is available): empirical lookup table (count-based Bayes,
  converges to the oracle), LogisticRegression, MLPClassifier, RandomForest,
  GradientBoosting, kNN. Add a train-set-size learning curve so you can see which
  learners approach the oracle and by how much they beat STANDARD.
- Output: LER-per-model bar chart + learning curve + CSV. torch GNN is the
  circuit-level version later (needs network to install).

## Model-zoo results (2026-11-11) — see model_zoo_report.md

- Env is now LOCAL Windows (GTX 1080 Ti, 11 GB, sm_61). Project venv at `.venv/`
  (numpy, scipy, matplotlib, sklearn). PyPI reachable here.
- `model_zoo.py`: exact LER (enumerates all 16,384 noise atoms; no MC eval noise),
  8 sklearn learners x 15 N (100..300k) x 10 seeds x 4 noise points; ~31 min on 8 cores.
  `--report-only` rebuilds `figures/` + `model_zoo_report.md` from `results/*.csv`;
  hand-written findings live in `model_zoo_findings.md` (spliced in).
- KEY FINDING: STANDARD has exact logical-class ties on ~580/1024 inputs. At c<=0.4,
  ~100% of the ORACLE gain is just breaking those ties well; only at c>=0.6 does
  correlation override confident STANDARD decisions (10-40% of gain). Report three
  baselines going forward: STANDARD (random tie), STANDARD + best tie, ORACLE/learned.
- RF / HGB / lookup table reach 99%+ of the gap; logistic & naive Bayes are WORSE than
  STANDARD. Speed is irrelevant at d=3 (everything compiles to a 1024-entry LUT).
- HYBRIDS added (`--only hybrid` reruns just them, ~2 min): tie-break LUT (best at small N
  for c<=0.4, capped at the tie share — plateaus at 83% for c=0.6), prior LUT (alpha=10;
  converges to ORACLE, slower than plain LUT at strong c), HGB + STANDARD-LLR feature
  (best large-N at strong c, but worse than STANDARD at N=100). Next: residual boosting
  from STANDARD's LLR as base_margin (XGBoost/LightGBM) — the scalable d=5,7 hybrid.
- TODO: (a) weighted-unique-row training (1024 rows + sample_weight) => ~100x faster runs;
  (b) alpha sweep for prior LUT; (c) make heralding_correlation_sweep.py report the
  random-tie baseline explicitly. GPU only pays off for the Phase-2 torch GNN (check that
  the torch build includes sm_61, e.g. cu126 wheels).

## Session 2026-11-11 (later): all four next-steps landed

- `heralding_correlation_sweep.py` now EXACT (imports model_zoo lazily), 15x15 grid in
  ~5 s, plots gain + share-from-ties. `--mc-check` cross-checks vs old MC (passes).
  Max gain 5.3x at eta=1, c=0.8.
- Phase 0 WORKS: `color_code_circuit.py` = own triangular 6.6.6 memory generator with
  Chromobius 4th-coord annotations (stim's built-in lacks them). CX order MUST be
  cyclic around the hexagon or Chromobius fails at d>=5 ("Failed to decompose").
  Bare-ancilla => circuit distance (d+1)/2 (same as stim's built-in); upgrade to
  Gidney-Jones superdense before scaling claims. Threshold ~0.35-0.45%.
  Chromobius accepts 4th coord -1 = ignore detector (useful for heralds/flags).
- Residual XGB hybrid (base_margin = STANDARD LLR): fixes small-N failure, but
  regularization caps it at 96.5-99%; needs N-dependent regularization / CV.
- alpha sweep (`--alpha-sweep`): best alpha is regime-dependent (large at moderate c,
  small at strong c) -> pick by CV.
- `girvin_writeup_draft.md`: first draft of the pitch write-up.
- NEXT: Phase 1 proper — HERALDED_ERASE in color_code_circuit (hook exists:
  `p_erase`) + per-shot herald-conditioned DEM reweighting for erasure-aware
  Chromobius; then correlated leakage.

## Roadmap

1. Phase 0 (done): tooling + Pauli baseline + η/correlation toys. ✅
2. Model-zoo accuracy comparison at d=3 (next, sklearn — runs here). ◀ START HERE
3. Phase 1: add HERALDED_ERASE to the color-code circuit; circuit-level
   standard erasure-aware baseline at d=3,5,7 (needs stim).
4. Phase 2: physical correlated leakage + imperfect heralding; GNN decoder on
   the color-code lattice (needs torch). Primary output = characterization curve.
5. Phase 3: accuracy/latency frontier note (ms neutral-atom budget).

## Key references

- Gidney & Jones, *New circuits and an open source decoder for the color code*,
  arXiv:2312.08813 (Chromobius).
- Lee, Li et al., *Color code decoder with improved scaling for circuit-level
  noise*, arXiv:2404.07482.
- Sahay & Brown, *Decoder for the triangular color code by matching on a Möbius
  strip*, PRX Quantum 3, 010310 (2022).
- Ma et al., *mid-circuit erasure conversion*, arXiv:2305.05493.
- Pecorari et al., *Loss-biased fault-tolerant QEC*, arXiv:2604.21876
  (loss biasing; ν ~ d/4 → d/2 → d).
- *Tracking and Decoding Rydberg Leakage Error*, arXiv:2411.04664.
- Gidney, *Stim*, Quantum 5, 497 (2021); Higgott, *PyMatching*, 2022.