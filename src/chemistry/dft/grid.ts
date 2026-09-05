// ─────────────────────────────────────────────────────────────
// grid.ts — molecular grid for DFT numerical integration.
//
// Per atom A:
//   • Radial: Mura-Knowles 1996 mapping r = -ξ_A · ln(1 - x³),
//     x ∈ (0, 1), uniform spacing in x. Per-atom Becke radial
//     scale ξ_A from atomic radii (kept as a small per-element
//     table; defaults to 1.0 Bohr for atoms not listed).
//   • Angular (default): Lebedev-Laikov order-110 octahedral-
//     symmetric quadrature — 110 points, exact for spherical
//     harmonics up to L = 17. Replaced the older Gauss-Legendre
//     × uniform-φ product (12 × 24 = 288 points, exact only to
//     L = 11) for a 2.6× point reduction at strictly better
//     accuracy. Available orders: 50, 110, 302 — see
//     `lebedev.ts`. Falls back to the product rule when
//     `nLebedev: null` is passed explicitly.
//   • Becke 1988 partition with 3-iteration smooth step:
//        s(μ) = ½ (1 − p₃(p₃(p₃(μ))))
//        p₃(x) = (3x − x³)/2
//     so each atom-centered shell carries a smooth weight that
//     vanishes near other nuclei, summing to 1 across atoms.
//
// All grid points are produced in BOHR (matching the integral
// pipeline), and each point's weight already includes the r²
// Jacobian + Becke factor — so any integrand g(r) integrates as
// just Σ w_i g(r_i) without further bookkeeping.
// ─────────────────────────────────────────────────────────────

import type { Nucleus } from "../cg-molecular.js";
import type { AtomSymbol } from "../atoms.js";
import { lebedev, type LebedevOrder } from "./lebedev.js";

export interface MolecularGrid {
  /** Cartesian coordinates of grid points (Bohr), length nPoints each. */
  readonly x: Float64Array;
  readonly y: Float64Array;
  readonly z: Float64Array;
  /** Combined weights: r² · radial · angular · Becke. Σ w_i g(r_i) ≈ ∫ g d³r. */
  readonly w: Float64Array;
}

export interface GridOpts {
  /** Radial points per atom. Default 50. */
  readonly nRadial?: number;
  /**
   * Lebedev order for angular quadrature. Default 110 (exact for
   * L ≤ 17). Available: 50, 110, 302 (`LEBEDEV_AVAILABLE_ORDERS`).
   * Pass `null` to use the legacy Gauss-Legendre × uniform-φ
   * product rule (controlled by `nTheta` and `nPhi`).
   */
  readonly nLebedev?: LebedevOrder | null;
  /** Polar (cos θ) Gauss-Legendre nodes per radial shell, used only
   *  when `nLebedev: null`. Default 12. */
  readonly nTheta?: number;
  /** Azimuthal uniform nodes per shell, used only when
   *  `nLebedev: null`. Default 24. */
  readonly nPhi?: number;
}

/**
 * Build the full molecular grid: union of per-atom shells with Becke
 * partition weights applied. Atoms are identified by `nuclei[i].pos`
 * (Bohr) and by `symbols[i]` for Becke-radius lookup.
 */
export function molecularGrid(
  nuclei: readonly Nucleus[],
  symbols: readonly AtomSymbol[],
  opts: GridOpts = {},
): MolecularGrid {
  const nRad = opts.nRadial ?? 50;
  const useLebedev = opts.nLebedev !== null;       // explicit null disables
  const lebedevOrder: LebedevOrder = (opts.nLebedev ?? 110);
  const nTh  = opts.nTheta  ?? 12;
  const nPh  = opts.nPhi    ?? 24;
  if (nuclei.length !== symbols.length) {
    throw new Error(`molecularGrid: nuclei.length (${nuclei.length}) ≠ symbols.length (${symbols.length})`);
  }

  // Pre-compute per-atom local grids in BOHR, anchored at the nucleus.
  const radial = nuclei.map((_, i) => muraKnowlesRadial(nRad, BECKE_XI[symbols[i]!] ?? 1.0));

  // Build the angular grid once — same Lebedev (or product) rule
  // for every radial shell, every atom.
  let angX: Float64Array, angY: Float64Array, angZ: Float64Array, angW: Float64Array;
  if (useLebedev) {
    const ang = lebedev(lebedevOrder);
    angX = ang.x; angY = ang.y; angZ = ang.z; angW = ang.w;
  } else {
    const { x: cosTh, w: wTh } = gaussLegendreNodes(nTh);
    const dPhi = (2 * Math.PI) / nPh;
    const total = nTh * nPh;
    angX = new Float64Array(total); angY = new Float64Array(total);
    angZ = new Float64Array(total); angW = new Float64Array(total);
    let k = 0;
    for (let ti = 0; ti < nTh; ti++) {
      const cT = cosTh[ti]!;
      const sT = Math.sqrt(Math.max(0, 1 - cT * cT));
      const wT = wTh[ti]!;
      for (let pi = 0; pi < nPh; pi++) {
        const phi = pi * dPhi;
        angX[k] = sT * Math.cos(phi);
        angY[k] = sT * Math.sin(phi);
        angZ[k] = cT;
        angW[k] = wT * dPhi;
        k++;
      }
    }
  }
  const nAng = angX.length;

  // Total points = nuclei × nRad × nAng.
  const total = nuclei.length * nRad * nAng;
  const X = new Float64Array(total);
  const Y = new Float64Array(total);
  const Z = new Float64Array(total);
  const W = new Float64Array(total);

  // Pre-compute inter-atomic distances for Becke partition.
  const nA = nuclei.length;
  const Rab = new Float64Array(nA * nA);
  for (let a = 0; a < nA; a++) {
    for (let b = a + 1; b < nA; b++) {
      const dx = nuclei[a]!.pos[0] - nuclei[b]!.pos[0];
      const dy = nuclei[a]!.pos[1] - nuclei[b]!.pos[1];
      const dz = nuclei[a]!.pos[2] - nuclei[b]!.pos[2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      Rab[a * nA + b] = r;
      Rab[b * nA + a] = r;
    }
  }

  let idx = 0;
  const beckeP = new Float64Array(nA);    // p_A scratch
  const distToAtom = new Float64Array(nA); // |r - R_A| scratch

  for (let aIdx = 0; aIdx < nA; aIdx++) {
    const Rx = nuclei[aIdx]!.pos[0];
    const Ry = nuclei[aIdx]!.pos[1];
    const Rz = nuclei[aIdx]!.pos[2];
    const { r: rRad, w: wRad } = radial[aIdx]!;
    for (let ri = 0; ri < nRad; ri++) {
      const r = rRad[ri]!;
      const wR = wRad[ri]!;          // already includes r² Jacobian
      for (let aj = 0; aj < nAng; aj++) {
        const ux = angX[aj]!, uy = angY[aj]!, uz = angZ[aj]!;
        const wA = angW[aj]!;
        const px = Rx + r * ux;
        const py = Ry + r * uy;
        const pz = Rz + r * uz;
        // Compute Becke weight for this atom at this point.
        for (let b = 0; b < nA; b++) {
          const dx = px - nuclei[b]!.pos[0];
          const dy = py - nuclei[b]!.pos[1];
          const dz = pz - nuclei[b]!.pos[2];
          distToAtom[b] = Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        for (let A = 0; A < nA; A++) {
          let prod = 1;
          for (let B = 0; B < nA; B++) {
            if (A === B) continue;
            const mu = (distToAtom[A]! - distToAtom[B]!) / Rab[A * nA + B]!;
            prod *= beckeStep(mu);
          }
          beckeP[A] = prod;
        }
        let pSum = 0;
        for (let A = 0; A < nA; A++) pSum += beckeP[A]!;
        const wB = pSum > 0 ? beckeP[aIdx]! / pSum : 0;

        X[idx] = px;
        Y[idx] = py;
        Z[idx] = pz;
        W[idx] = wR * wA * wB;
        idx++;
      }
    }
  }

  return { x: X, y: Y, z: Z, w: W };
}

// ── Becke 1988 radial grid (M3 mapping × Gauss-Chebyshev 2nd kind) ──
//
// Substitution r = ξ · (1 + x)/(1 − x), x ∈ (−1, 1).
//   dr/dx = 2ξ/(1 − x)²
//   ∫₀^∞ f(r) r² dr = ∫_{−1}^1 f(r(x)) · r²(x) · 2ξ/(1−x)² dx
// Apply Gauss-Chebyshev second-kind quadrature on the x integral,
// with the standard 1/√(1−x²) factor folded into the weights:
//   x_i = cos(i π/(n+1)),   w_i^GC = π/(n+1) · sin²(i π/(n+1))
//   ∫_{−1}^1 g(x) dx ≈ Σ_i w_i^GC / √(1−x_i²) · g(x_i)
//                    = Σ_i (π/(n+1) · sin θ_i) · g(x_i)
// (since sin²/√(1−cos²) = sin²/|sin| = sin for i ∈ {1..n}.)
// Total radial weight at node i:
//   W_i = (π/(n+1) · sin θ_i) · 2ξ/(1−x_i)² · r²(x_i),    θ_i = i π/(n+1).
// Function name retained as a stable export — the M3-style mapping is
// in the Mura-Knowles family (different parameter choice).
function muraKnowlesRadial(n: number, xi: number): { r: Float64Array; w: Float64Array } {
  const r = new Float64Array(n);
  const w = new Float64Array(n);
  for (let i = 1; i <= n; i++) {
    const theta = (i * Math.PI) / (n + 1);
    const x = Math.cos(theta);
    const sinTh = Math.sin(theta);
    const ri = xi * (1 + x) / (1 - x);
    r[i - 1] = ri;
    const drDx = (2 * xi) / ((1 - x) * (1 - x));
    w[i - 1] = (Math.PI / (n + 1)) * sinTh * drDx * ri * ri;
  }
  return { r, w };
}

// ── Gauss-Legendre nodes on [−1, 1] ─────────────────────────
//
// Newton iteration on the Legendre polynomial roots. Standard
// Numerical Recipes recipe (Press et al.).
function gaussLegendreNodes(n: number): { x: Float64Array; w: Float64Array } {
  const x = new Float64Array(n);
  const w = new Float64Array(n);
  const half = (n + 1) >> 1;
  for (let i = 0; i < half; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let z1 = 0;
    let pp = 0;
    for (let it = 0; it < 100; it++) {
      let p1 = 1, p2 = 0, p3 = 0;
      for (let j = 0; j < n; j++) {
        p3 = p2;
        p2 = p1;
        p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1);
      }
      pp = (n * (z * p1 - p2)) / (z * z - 1);
      z1 = z;
      z = z1 - p1 / pp;
      if (Math.abs(z - z1) < 1e-14) break;
    }
    x[i] = -z;
    x[n - 1 - i] = z;
    const wi = 2 / ((1 - z * z) * pp * pp);
    w[i] = wi;
    w[n - 1 - i] = wi;
  }
  return { x, w };
}

// ── Becke 3-iteration smooth step ────────────────────────────
//
// p₃(x) = (3x − x³) / 2,   s(μ) = ½ (1 − p₃(p₃(p₃(μ))))
// At μ = 0 (equidistant from two atoms) s = 1/2; at μ = ±1 it
// saturates to 0 / 1 with vanishing derivatives. Three iterations
// is the standard choice (Becke 1988).
function beckeStep(mu: number): number {
  const p1 = 0.5 * (3 * mu - mu * mu * mu);
  const p2 = 0.5 * (3 * p1 - p1 * p1 * p1);
  const p3 = 0.5 * (3 * p2 - p2 * p2 * p2);
  return 0.5 * (1 - p3);
}

// ── Per-atom Becke radial scaling (angstrom) ────────────────
//
// ξ_A is the Bragg-Slater radius of atom A in ANGSTROM (the
// header previously said Bohr; every tabulated value is and always
// was the Angstrom number — C is 0.70, not 1.32); it
// sets the "natural distance" at which the radial mapping puts
// half its quadrature points. Values from Becke (1988) /
// Treutler-Ahlrichs (1995). Atoms not listed default to 1.0.
const BECKE_XI: Partial<Record<AtomSymbol, number>> = {
  H:  0.5,    // tighter than the default (Becke recommends ξ_H = 0.35;
              // 0.5 is a compromise that handles diatomic H₂ better)
  // Corrected 2026-08-10 against pyscf.dft.radi.BRAGG_RADII (the same
  // table Becke's scheme is defined on). He was labelled "Bragg-Slater
  // radius for He" but carried 0.35 against the tabulated 1.40; C, N and
  // O had all been filled with 0.65 where the real values differ.
  // Noble gases are anomalously large in this parameterization — that is
  // the table, not a typo.
  He: 1.40,   // was 0.35
  Li: 1.45,
  Be: 1.05,
  B:  0.85,
  C:  0.70,   // was 0.65
  N:  0.65,
  O:  0.60,   // was 0.65
  F:  0.50,   // was MISSING → silently fell back to 1.0, misplacing the
              // radial quadrature points for every fluorine DFT grid
  Ne: 1.50,
  // Row 3, same source (pyscf.dft.radi.BRAGG_RADII, converted with
  // pyscf.data.nist.BOHR). P / S / Cl all land on exactly 1.00 in that
  // table — not a copy-paste slip. Na and Ar tie at 1.80 for the same
  // reason the row-2 noble gases are large here.
  Na: 1.80,
  Mg: 1.50,
  Al: 1.25,
  Si: 1.10,
  P:  1.00,
  S:  1.00,
  Cl: 1.00,
  Ar: 1.80,
};
