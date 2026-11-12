## Summary

- **A hybrid is the best decoder at nearly every training size and noise point.** In
  the "who wins" table below, a hybrid has the lowest LER in 26 of 28 cells. The
  exceptions are two mid-N cells at strong correlation. Four hybrids were tested; the
  fourth, residual XGB, was added after the first three to fix HGB + LLR's small-data
  failure (see below).

- **Moderate correlation (c = 0.4): fixing STANDARD's ties is all you need.**
  - The tie-break LUT never touches a decision STANDARD is confident about. Even so, it
    is the best decoder from ~3k shots onward, and it closes 92–95% of the gap with only
    ~1k shots.
  - The best plain learners need ~10× more data to match it. At N≈1k, the plain lookup
    table closes 88% and gradient boosting closes 80–82%.
  - At N=100, the table hybrids already close 72–83% of the gap. Plain gradient boosting
    is at about −200%: worse than STANDARD by twice the size of the gap.

- **Strong correlation (c = 0.6): the tie-break ceiling binds, and the scalable hybrid
  takes over.**
  - The tie-break LUT stops at 83.0%, exactly the tie share for this point.
  - From ~3k shots onward, **HGB + STANDARD LLR** is the best of all 11 decoders, ending
    at 99.6% of the gap.
  - The prior LUT converges to the ORACLE but lags in the middle (91% at 10k vs 95% for
    HGB + LLR). Its α=10 prior holds back the overrides this regime needs.

- **The hybrids do no harm when there is nothing to learn.** At the independent-noise
  point (STANDARD = ORACLE), both table hybrids have *zero* excess LER at every training
  size. Plain learners pay for having to relearn STANDARD from data: the lookup table
  and gradient boosting are still 0.15–0.36×10⁻⁴ above the ORACLE at 300k shots.

- **HGB + STANDARD LLR's weakness is small data.** At N=100 it is worse than STANDARD
  (−107% / −81% / −27% of the gap). It still beats plain gradient boosting (−211% /
  −201% / −97%), so the LLR feature helps. But boosting doesn't fall back on the feature
  it was given.

- **Residual XGB (boosting that *starts* at STANDARD) fixes that weakness, at a price.**
  - At N=100 it is at −6% / +16% / +22% of the gap, versus −107% / −81% / −27% for
    HGB + LLR. It has zero excess LER at the independent point.
  - But it never wins a cell. The regularization that keeps it near STANDARD with little
    data also caps it at 96.5–99% of the gap at 300k shots. HGB + LLR reaches 99.4–99.7%.
  - The fix is regularization that loosens as N grows (early stopping on a validation
    split, or min_child_weight and λ scaled with N). This is still the right design for
    d=5,7: it is the only scalable hybrid that is safe with little data.

- **Prior strength α should be chosen per regime, not fixed.**
  - α→∞ is the tie-break LUT and α→0 is the plain lookup table; the prior LUT
    interpolates between them.
  - At moderate correlation, large α (30–100) is best at every N. Holding onto STANDARD
    is correct because only its ties need fixing.
  - At strong correlation, small α (≤1) is best from N≈300 onward. α=10 trails by 3–6
    points of the gap at 1k–100k shots.
  - So the best α encodes how much correlation is present. Selecting it by
    cross-validation on the training set (no peeking at the exact LER) would get the
    best of both. The same logic applies to residual XGB's regularization strength.

**What this means for the thesis.** The decoder question splits by regime:

- Where STANDARD's degeneracies explain the whole gap, a minimal "learn the ties" fix is
  both sufficient and the most data-efficient option.
- Where correlation is strong enough to overrule confident decisions, a learned model
  anchored on STANDARD's soft output wins.

The scalable path for d=5,7 is a learned model anchored on STANDARD: residual boosting,
or a GNN that takes STANDARD's output as input. It should be judged against all three
baselines: STANDARD, STANDARD + best tie-break, and ORACLE where it can be computed.
