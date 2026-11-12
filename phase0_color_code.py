#!/usr/bin/env python3
"""
Phase 0 (de-risk): triangular color code memory threshold under circuit-level
depolarizing noise, decoded with Chromobius via Sinter.

This is the STANDARD, reliable baseline -- no erasure yet. Its job is to prove
the tooling works end to end and to give you the Pauli-only logical-error curves
that every later experiment is measured against. Run this first; if it produces
a sensible threshold crossing around p ~ 0.2-0.5% across distances, your stack
is good and you can move to the erasure study.

Requires: stim, chromobius, sinter, numpy, matplotlib   (see requirements.txt)
Does NOT run in an environment without those installed; see README for install.

Usage:
    python3 phase0_color_code.py --selftest          # tiny, ~seconds, sanity only
    python3 phase0_color_code.py                     # full sweep, saves PNG + CSV
    python3 phase0_color_code.py -d 3 5 7 --shots 200000
"""

import argparse
import sys


def build_circuit(distance, rounds, p):
    # stim's built-in color_code:memory_xyz has only 3 detector coordinates, but
    # Chromobius needs a 4th (basis+color), so we use our own annotated generator.
    # NOTE: like stim's built-in circuit, it is a bare-ancilla circuit whose
    # circuit distance is ~(d+1)/2 because of hook errors (see color_code_circuit.py).
    from color_code_circuit import memory_circuit
    return memory_circuit(distance, rounds=rounds, p=p)


def make_tasks(distances, ps):
    import sinter
    tasks = []
    for d in distances:
        for p in ps:
            tasks.append(sinter.Task(
                circuit=build_circuit(d, rounds=d, p=p),
                json_metadata={"d": d, "r": d, "p": p},
            ))
    return tasks


def run(distances, ps, max_shots, max_errors, workers):
    import sinter
    import chromobius  # noqa: F401  (registers the sinter decoder)

    stats = sinter.collect(
        num_workers=workers,
        tasks=make_tasks(distances, ps),
        decoders=["chromobius"],
        custom_decoders=chromobius.sinter_decoders(),
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )
    return stats


def save_outputs(stats, path_stub):
    import matplotlib.pyplot as plt

    # CSV
    with open(f"{path_stub}.csv", "w") as f:
        f.write("d,p,shots,errors,logical_error_rate\n")
        for s in stats:
            d = s.json_metadata["d"]
            p = s.json_metadata["p"]
            ler = s.errors / s.shots if s.shots else float("nan")
            f.write(f"{d},{p},{s.shots},{s.errors},{ler:.6g}\n")

    plot_csv(f"{path_stub}.csv", f"{path_stub}.png")


def plot_csv(csv_path, png_path):
    """LER vs p per distance (threshold = where curves cross), from the CSV."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]          # dataviz reference slots 1-3
    by_d = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            by_d.setdefault(int(r["d"]), []).append(
                (float(r["p"]), int(r["shots"]), int(r["errors"])))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "axes.facecolor": SURF, "figure.facecolor": SURF})
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for d, col in zip(sorted(by_d), PALETTE):
        pts = sorted(by_d[d])
        p = [x[0] for x in pts]
        ler = [e / s for _, s, e in pts]
        err = [1.96 * ((l * (1 - l)) / s) ** 0.5 for l, (_, s, _) in zip(ler, pts)]
        ax.errorbar(p, ler, yerr=err, fmt="-o", color=col, lw=2, ms=6, mec=SURF,
                    mew=1.2, capsize=2, label=f"d={d}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("physical error rate p (circuit-level depolarizing)")
    ax.set_ylabel("logical error rate per shot (d rounds)")
    ax.set_title("Triangular color code memory, Chromobius decoder  (±95% binomial CI)",
                 loc="left", color=INK, fontsize=11)
    ax.text(0.99, 0.02, "bare-ancilla circuit: circuit distance (d+1)/2 from hook errors",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
    ax.legend(frameon=False); ax.grid(True, which="both", alpha=0.25, lw=0.6)
    fig.tight_layout(); fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--max-errors", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selftest", action="store_true",
                    help="tiny run (d=3, few points, few shots) to check the stack")
    args = ap.parse_args(argv)

    try:
        import stim, sinter, chromobius  # noqa: F401
    except ImportError as e:
        print(f"[missing dependency] {e}\n"
              f"Install with:  pip install stim chromobius sinter numpy matplotlib\n"
              f"(PyPI must be reachable; see README.)", file=sys.stderr)
        return 2

    if args.selftest:
        ps = [0.001, 0.003, 0.006]
        stats = run([3], ps, max_shots=5_000, max_errors=200, workers=args.workers)
        for s in stats:
            print(s.json_metadata, "LER=",
                  (s.errors / s.shots if s.shots else float('nan')))
        print("selftest OK: stim+chromobius+sinter are wired correctly.")
        return 0

    import numpy as np
    ps = list(np.geomspace(0.0008, 0.01, 8))
    stats = run(args.distances, ps, args.shots, args.max_errors, args.workers)
    save_outputs(stats, "phase0_color_code_threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
