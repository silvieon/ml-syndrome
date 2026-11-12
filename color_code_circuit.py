#!/usr/bin/env python3
"""
Triangular 6.6.6 color-code memory circuits, annotated for Chromobius.

Why our own generator: stim's built-in `color_code:memory_xyz` labels detectors
with 3 coordinates, but Chromobius needs a 4th coordinate giving each detector's
basis and color (RedX=0, GreenX=1, BlueX=2, RedZ=3, GreenZ=4, BlueZ=5). We also
need full control of the circuit to insert heralded erasures and correlated
leakage in Phase 1/2.

Lattice: points (i, j) of a triangular lattice with i, j >= 0, i + j <= m,
m = 3(d-1)/2. Points with (i - j) % 3 == 1 are plaquette centers; the rest are
data qubits. Each plaquette acts on its in-region neighbours (weight 6 in the
bulk, 4 on the boundary). Gives [[ (3d^2+1)/4, 1, d ]]: n = 7, 19, 37 for
d = 3, 5, 7.

Circuit (Z-basis memory, `rounds` rounds):
  per round: data depolarization -> X checks (RX ancilla, 6 CX layers, MX)
             -> Z checks (R ancilla, 6 CX layers, M)
  one ancilla per check; CX layers are conflict-free by construction (each
  layer is one lattice direction) and go cyclically around the hexagon (see
  X_ORDER). Logical Z = Z on every data qubit.

  CAVEAT: bare-ancilla readout -> hook errors -> circuit distance ~(d+1)/2
  (2, 3, 4 for d = 3, 5, 7), same as stim's built-in color-code circuit. Fine
  for comparing decoders on a fixed circuit; for threshold/scaling claims,
  upgrade to Gidney-Jones superdense circuits (arXiv:2312.08813).

Noise (same knobs as phase0_color_code.py / stim's generated circuits):
  DEPOLARIZE2(p) after every CX, DEPOLARIZE1(p) on data before each round,
  flip(p) before every measurement and after every reset.

Optional heralded erasure (Phase 1): `p_erase` applies stim's HERALDED_ERASE to
every data qubit before each round. Each herald is a measurement-record bit;
it is NOT a detector, so the detector error model (DEM) stays decodable by
Chromobius, and the heralds come out of stim as extra measurement bits for
the erasure-aware decoder to use. Build the DEM with
`circuit.detector_error_model(approximate_disjoint_errors=True)` (stim requires it
for HERALDED_ERASE). Decoding that DEM directly treats erasures as Pauli noise,
i.e. the herald-IGNORANT baseline; erasure-aware decoding needs per-shot
reweighting from the herald bits (Phase 1, not implemented yet).
"""

import math

import numpy as np

NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]   # neighbour directions
# CX layer order (indices into NB). MUST go cyclically around the hexagon:
# then every hook error (ancilla fault partway through the sequence) acts on a
# contiguous arc of the plaquette, whose symptoms are a same-color pair or a
# rainbow triple -- both decomposable by Chromobius. Orders that touch opposite
# vertices early (e.g. 0,1,2,...) create 4-symptom GGBB errors that Chromobius
# rejects ("Failed to decompose") at d>=5. All 12 cyclic orders were checked to
# compile at d=3,5,7; circuit distance is ~(d+1)/2 for every order (bare-ancilla
# hook limit, same as stim's built-in color_code:memory_xyz).
X_ORDER = [0, 2, 5, 1, 3, 4]
Z_ORDER = [0, 2, 5, 1, 3, 4]


class Lattice:
    def __init__(self, d):
        if d < 3 or d % 2 == 0:
            raise ValueError("d must be odd and >= 3")
        self.d = d
        m = 3 * (d - 1) // 2
        pts = [(i, j) for i in range(m + 1) for j in range(m + 1 - i)]
        is_center = lambda p: (p[0] - p[1]) % 3 == 1
        self.data = [p for p in pts if not is_center(p)]
        idx = {p: k for k, p in enumerate(self.data)}
        self.plaqs = []          # (center, {direction_k: data_index})
        for i in range(-2, m + 3):
            for j in range(-2, m + 3):
                if not is_center((i, j)):
                    continue
                nbrs = {k: idx[(i + a, j + b)] for k, (a, b) in enumerate(NB)
                        if (i + a, j + b) in idx}
                if len(nbrs) >= 4:
                    self.plaqs.append(((i, j), nbrs))
        self.colors = self._three_color()
        self.n = len(self.data)
        assert self.n == (3 * d * d + 1) // 4, (d, self.n)

    def _three_color(self):
        """Proper 3-coloring of plaquettes (adjacent = share a data qubit)."""
        P = len(self.plaqs)
        sets = [set(q.values()) for _, q in self.plaqs]
        adj = [[b for b in range(P) if b != a and sets[a] & sets[b]] for a in range(P)]
        color = [-1] * P
        for start in range(P):
            if color[start] >= 0:
                continue
            color[start] = 0
            stack = [start]
            while stack:
                a = stack.pop()
                for b in adj[a]:
                    if color[b] < 0:
                        used = {color[c] for c in adj[b] if color[c] >= 0}
                        color[b] = min({0, 1, 2} - used)
                        stack.append(b)
        for a in range(P):
            assert all(color[a] != color[b] for b in adj[a]), "not 3-colorable"
        return color

    def xy(self, p):
        """Cartesian coordinates of lattice point p (for QUBIT_COORDS / detectors)."""
        return (p[0] + p[1] / 2, p[1] * math.sqrt(3) / 2)

    def check_matrix(self):
        H = np.zeros((len(self.plaqs), self.n), np.uint8)
        for r, (_, q) in enumerate(self.plaqs):
            H[r, list(q.values())] = 1
        return H


def memory_circuit(d, rounds=None, p=0.001, p_erase=0.0, x_order=None, z_order=None):
    """Z-basis memory experiment on the distance-d triangular color code."""
    import stim
    lat = Lattice(d)
    rounds = d if rounds is None else rounds
    orders = {"X": x_order or X_ORDER, "Z": z_order or Z_ORDER}
    n, P = lat.n, len(lat.plaqs)
    data = list(range(n))
    xanc = list(range(n, n + P))
    zanc = list(range(n + P, n + 2 * P))
    c = stim.Circuit()
    for q, pt in enumerate(lat.data):
        c.append("QUBIT_COORDS", [q], list(lat.xy(pt)))
    for k, (ctr, _) in enumerate(lat.plaqs):
        x, y = lat.xy(ctr)
        c.append("QUBIT_COORDS", [xanc[k]], [x, y, 0])
        c.append("QUBIT_COORDS", [zanc[k]], [x, y, 1])

    def det_coords(k, t, basis):
        x, y = lat.xy(lat.plaqs[k][0])
        return [x, y, t, lat.colors[k] + (3 if basis == "Z" else 0)]

    c.append("R", data)
    if p > 0:
        c.append("X_ERROR", data, p)
    n_meas = 0           # running count of measurement-record entries
    last_x, last_z = None, None
    for r in range(rounds):
        c.append("TICK")
        if p > 0:
            c.append("DEPOLARIZE1", data, p)
        if p_erase > 0:
            c.append("HERALDED_ERASE", data, p_erase)
            n_meas += n
        for basis, anc in (("X", xanc), ("Z", zanc)):
            c.append("RX" if basis == "X" else "R", anc)
            if p > 0:
                c.append("Z_ERROR" if basis == "X" else "X_ERROR", anc, p)
            for direction in orders[basis]:
                pairs = []
                for k, (_, q) in enumerate(lat.plaqs):
                    if direction in q:
                        pairs += [anc[k], q[direction]] if basis == "X" else [q[direction], anc[k]]
                if pairs:
                    c.append("TICK")
                    c.append("CX", pairs)
                    if p > 0:
                        c.append("DEPOLARIZE2", pairs, p)
            if p > 0:
                c.append("Z_ERROR" if basis == "X" else "X_ERROR", anc, p)
            c.append("MX" if basis == "X" else "M", anc)
            first = n_meas
            n_meas += P
            prev = last_x if basis == "X" else last_z
            for k in range(P):
                cur = stim.target_rec(first + k - n_meas)
                if prev is not None:
                    c.append("DETECTOR", [cur, stim.target_rec(prev + k - n_meas)],
                             det_coords(k, r, basis))
                elif basis == "Z":     # first-round Z checks are deterministic on |0...0>
                    c.append("DETECTOR", [cur], det_coords(k, r, basis))
            if basis == "X":
                last_x = first
            else:
                last_z = first

    c.append("TICK")
    if p > 0:
        c.append("X_ERROR", data, p)
    c.append("M", data)
    first_data = n_meas
    n_meas += n
    for k, (_, q) in enumerate(lat.plaqs):
        recs = [stim.target_rec(first_data + i - n_meas) for i in q.values()]
        recs.append(stim.target_rec(last_z + k - n_meas))
        c.append("DETECTOR", recs, det_coords(k, rounds, "Z"))
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(first_data + i - n_meas) for i in data], 0)
    return c


def selfcheck(ds=(3, 5, 7)):
    """Structural checks: code parameters, deterministic detectors, circuit distance."""
    for d in ds:
        lat = Lattice(d)
        H = lat.check_matrix()
        assert (H.sum(axis=1) % 2 == 0).all()                   # X and Z checks commute
        circ = memory_circuit(d, p=0.001)
        dem = circ.detector_error_model(decompose_errors=False)  # raises if non-deterministic
        errs = circ.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=6,
            dont_explore_edges_with_degree_above=6,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True)
        print(f"d={d}: n={lat.n}, plaquettes={len(lat.plaqs)}, detectors={dem.num_detectors}, "
              f"circuit distance={len(errs)}")


if __name__ == "__main__":
    selfcheck()
