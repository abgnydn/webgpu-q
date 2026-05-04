// ─────────────────────────────────────────────────────────────
// orbital.ts — evaluate the STO-3G H 1s wavefunction at any
// real-space point (atomic units, Bohr).
//
// φ_1s(r) = Σ_i c_i · (2 α_i / π)^{3/4} · exp(-α_i |r|²)
//
// Same exponents and contraction coefficients as the Hamiltonian
// builder — so the orbitals we render are exactly the orbitals the
// VQE optimizes against.
// ─────────────────────────────────────────────────────────────

import { STO3G_H_1S } from "../chemistry/integrals.js";

const ALPHA = STO3G_H_1S.alpha;
const C = STO3G_H_1S.c;

/** Pre-computed normalization factors N_i = (2 α_i / π)^{3/4}. */
const NORMS = ALPHA.map((a) => Math.pow(2 * a / Math.PI, 0.75));

/** φ_1s evaluated at offset (rx, ry, rz) from the atomic center. */
export function phi1s(dx: number, dy: number, dz: number): number {
  const r2 = dx * dx + dy * dy + dz * dz;
  let s = 0;
  for (let i = 0; i < ALPHA.length; i++) {
    s += C[i]! * NORMS[i]! * Math.exp(-ALPHA[i]! * r2);
  }
  return s;
}
