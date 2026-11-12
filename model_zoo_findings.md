## Findings

1. **At moderate correlation, the whole ORACLE-over-STANDARD gain comes from STANDARD's exact ties.**
   STANDARD's independent-noise model is *exactly indifferent* between the two logical
   classes on ~580 of 1024 inputs (8–11% of probability mass). These are the
   capacity-exceeding d=3 events: an erasure plus a flip lying on a common weight-3
   logical. With c ≤ 0.4, breaking those ties correctly recovers 90–100% of the gap. The
   correlation-aware ORACLE never overrides a decision STANDARD was confident about. Only
   at strong correlation (c ≥ 0.6) does it start overriding confident decisions, which
   accounts for 10–40% of the gain (see the share table below).

2. **The earlier sweep's numbers hold up, but its baseline is arbitrary.**
   `heralding_correlation_sweep.py` breaks STANDARD's ties deterministically toward
   absolute class 0. At η=0.7, c=0.4 that gives STANDARD LER ≈ 0.085–0.089 (60k–400k MC
   shots), close to the random-tie-break value of 0.0878, so its reported gains are not
   materially inflated there. But the rule is arbitrary. A different but equally valid
   STANDARD implementation can land anywhere between 0.0571 (best tie-break) and roughly
   0.09. Before figures go to Girvin, the sweep should report the random-tie-break
   baseline explicitly, alongside the best-tie-break line.

3. **Flexible learners reach the Bayes ceiling; linear ones can't beat STANDARD.**
   - The lookup table, random forest, hist gradient boosting, MLP and kNN close 96–99.6% of
     the gap at N = 300k.
   - The lookup table, random forest and gradient boosting reach 90% of the gap within
     ~1.3k–10k training shots. The MLP and kNN need ~10–20k.
   - Logistic regression and naive Bayes are *worse than STANDARD* at every N. Decoding is
     a parity-like, interaction-heavy function that additive models cannot represent.
     Adding pairwise interaction features ("logistic+pairs") helps but plateaus at 65–84%
     of the gap.

4. **Speed is not a differentiator at d=3.** With 1024 possible inputs, any decoder can be
   compiled to a lookup table that runs at ~3×10⁷ shots/s. In native form the learners
   span 340 shots/s (kNN at N = 300k) to 2.5×10⁷ shots/s (logistic regression). STANDARD's
   per-shot ML enumeration runs at 5×10⁵ shots/s. Among the flexible learners, hist
   gradient boosting has the best accuracy, fit time and throughput trade-off (fit about
   3 s, about 6×10⁵ shots/s).

5. **Hybrids help most when data is scarce, and the right hybrid depends on correlation
   strength** (see "Hybrid decoders" below).
   - **Tie-break LUT** (STANDARD everywhere except its exact ties) is the most
     sample-efficient decoder at moderate correlation. It closes 92–95% of the gap with
     ~1k shots and 97–100% with ~10k, ahead of every plain learner. But it is capped at
     the tie share by construction. At c=0.6 it levels off at 83%, exactly on the
     "STANDARD + best tie-break" line. That cap is finding 1, made visible in a decoder.
   - **Prior LUT** (STANDARD's posterior as a 10-pseudo-shot prior) matches the
     tie-break LUT at small N and converges to the ORACLE everywhere (99–99.9% at 300k).
     At strong correlation, though, the prior *resists* the overrides the data calls for.
     There it trails the plain lookup table at mid N (91% vs 94% at 10k), so α trades
     small-N safety against large-N speed.
   - **HGB + STANDARD LLR** (gradient boosting with STANDARD's soft output as a feature)
     is the best large-N model at strong correlation (95% of the gap at 10k, 99.6% at
     300k). However, it is *worse than STANDARD* at N=100: gradient boosting doesn't fall
     back to the feature it was given. This is the hybrid that scales to d=5,7. The fix
     to try there is boosting *from* STANDARD's LLR as the initial margin (residual
     boosting, `base_margin` in XGBoost/LightGBM), not just next to it.
   - All hybrids decode at plain-model speed: about 3×10⁷ shots/s for the LUTs, and
     about 9×10⁵ for HGB+LLR with the LLR tabulated. At larger d the LLR costs one
     STANDARD decode per shot.

**Implication for the thesis.** The honest claim is not "a learned decoder beats
standard erasure decoding by 2×." It is narrower:

- At moderate correlation, all of the value lies in a *small, identifiable set of
  degenerate syndromes*, where correlation information decides what the independent
  model cannot.
- At strong correlation, the learned decoder also starts overruling confident STANDARD
  decisions.

The hybrid results make the same point
constructively: a decoder that only fixes STANDARD's ties gets most of the gain with very
little data, and full correlation modeling is needed only to go past that.

The circuit-level d=5,7 version should report three baselines: STANDARD (random
tie-break), STANDARD + best tie-break, and ORACLE or learned. That way the decomposition
carries over.
