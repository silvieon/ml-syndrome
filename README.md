# Loss-aware color-code decoding on neutral-atom / trapped-ion noise — Phase 0

**Central question (the one the committee will test):**
*How much does a learned, loss-aware decoder recover over standard erasure-aware
decoding of a color code, as heralding degrades and loss mixes with Pauli and
correlated leakage noise?*

Note the baseline deliberately: **standard erasure-aware**, not loss-ignorant.
Erasure conversion is the whole point of these platforms, so beating a
loss-ignorant decoder is not a contribution. The real target is the gap over a
decoder that already uses the heralds — which the toy below shows is *small on
independent noise* and must be earned at circuit level with correlations.

---

## What runs today (numpy only)

```bash
python3 -I steane_erasure_demo.py
```

Exact maximum-likelihood decoding of the distance-3 triangular color code
(= [[7,1,3]] Steane), one CSS sector, under heralded loss + Pauli noise with a
tunable **heralding efficiency η**. Three decoders: loss-ignorant, standard
erasure-aware, Bayes-optimal "better". Demonstrates:

- heralds are worth a lot (STANDARD ≫ IGNORANT);
- at η=1, a smarter decoder can gain nothing (BETTER == STANDARD);
- the BETTER-over-STANDARD gap only opens under imperfect heralding, and on
  independent noise it stays small — so the thesis's contribution must come
  from **correlated circuit-level leakage**, which this toy cannot show.

That last point is a *result*, not a disclaimer: it tells you where to aim.

## What runs once the stack is installed

```bash
pip install -r requirements.txt           # needs PyPI reachable
python3 phase0_color_code.py --selftest   # ~seconds: confirms stim+chromobius+sinter wired
python3 phase0_color_code.py              # full sweep -> threshold PNG + CSV
```

Standard triangular-color-code memory threshold under circuit-level depolarizing
noise, decoded by Chromobius via Sinter. This de-risks the tooling and gives the
Pauli-only baseline curves. If distances d=3,5,7 show a threshold crossing around
p ≈ 0.2–0.5%, the stack is good.

## Roadmap (where this goes)

1. **Phase 0 (here):** tooling + Pauli baseline + the η toy. ✅ runnable
2. **Phase 1:** add `HERALDED_ERASE` channels to the color-code circuit; reproduce
   standard erasure-aware decoding at circuit level (Chromobius/PyMatching with
   erasure weights). Establish the honest baseline at d=3,5,7.
3. **Phase 2:** inject *correlated* leakage (deterministic leakage propagation /
   depolarizing-leakage model) and *imperfect* heralding; this is where a learned
   decoder (GNN on the color-code lattice) should find room. Primary output is the
   characterization curve: gain vs (η, loss:Pauli ratio, distance).
4. **Phase 3:** latency/accuracy frontier; note the ms-scale neutral-atom budget.

## Honest scope

- This is fault-tolerance *infrastructure* research. It is **not** a contribution
  to any quantum-attack-on-crypto timeline — slow-clock machines only ever do
  at-rest attacks and are years out. Decouple the motivation from Bitcoin.
- A specific decoder has a short half-life; frame the lasting contribution as the
  characterization curve, which survives a better decoder or better hardware.

## Key references

- Gidney & Jones, *New circuits and an open source decoder for the color code*,
  arXiv:2312.08813 (Chromobius; github.com/quantumlib/chromobius).
- Lee, Li & co., *Color code decoder with improved scaling for correcting
  circuit-level noise*, arXiv:2404.07482.
- Sahay & Brown, *Decoder for the triangular color code by matching on a Möbius
  strip*, PRX Quantum 3, 010310 (2022).
- Wu, Kolkowitz, Puri, Thompson, erasure conversion in alkaline-earth Rydberg
  arrays (erasure-conversion origin); Ma et al., arXiv:2305.05493 (mid-circuit
  erasure conversion, experiment).
- Pecorari, Brennen, Kondov, Pupillo, *Loss-biased fault-tolerant quantum error
  correction*, arXiv:2604.21876 (loss biasing; ν ~ d/4 → d/2 → d).
- *Tracking and Decoding Rydberg Leakage Error*, arXiv:2411.04664 (leakage
  propagation / decoding model).
- Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum 5, 497 (2021).
- Higgott, *PyMatching*, ACM Trans. Quantum Comput. 3 (2022).
