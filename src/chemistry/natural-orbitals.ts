// ─────────────────────────────────────────────────────────────
// natural-orbitals.ts — natural orbital occupation numbers (NOON)
// and natural orbital coefficients from a density matrix.
//
// Natural orbitals diagonalize the 1-particle density matrix D in
// the overlap-metric eigenproblem
//
//     D · S · c = n · c                (generalized eigenvalue)
//
// Symmetrize by multiplying through by Y = S^(1/2) (Löwdin-orthogonal
// metric, so Tr(Y D Y) = Tr(D S) = N_electrons):
//
//     Y D Y · (Y c) = n · (Y c)         standard symmetric eigenproblem
//     n_p, c̃_p = Y c_p                  eigenpairs
//     c_p = Y^(-1) · c̃_p = X · c̃_p     back-transform (X = S^(-1/2))
//
// Eigenvalues n_p ∈ [0, 2] are the natural orbital occupation
// numbers. For a single Slater determinant (HF / KS) they're exactly
// {2, 2, ..., 2, 0, 0, ..., 0}. For correlated wave functions (MP2,
// CCSD relaxed 1-PDM, CI) and for spin-contaminated UHF the values
// are fractional and carry physical information:
//
//   - n_p ∈ {0, 2} ± 1e-12     → essentially single-reference
//   - n_HOMO, n_LUMO ~ {1.8, 0.2}  → mild correlation
//   - n_HOMO, n_LUMO ~ {1, 1}    → strong / multi-reference character
//                                  (HEHE = "half-electron half-electron")
//
// For UHF / UKS, computing NOON on D_total = D_α + D_β makes the
// spin-contamination visible: spurious fractional NOONs flag where
// the unrestricted determinant differs from RHF.
//
// API:
//   `naturalOrbitalOccupations(D, S, X?)` — D in AO basis (closed-shell
//   convention 2·CC^T or unrestricted-total D_α + D_β). Returns
//   occupations (sorted descending) + natural-orbital coefficients
//   (AO basis, column p = NO p). X = S^(-1/2) optional; we recompute
//   from S if not provided.
// ─────────────────────────────────────────────────────────────

import { eigsymmetric } from "../manybody/dense-eig.js";

export interface NaturalOrbitalResult {
  /** NOON values in descending order. */
  readonly occupations: Float64Array;
  /** Natural orbital coefficients in the AO basis (n × n, column p = NO p). */
  readonly C_NO: Float64Array;
  /** Number of "strongly occupied" NOs (n_p > occThreshold; default 1.95). */
  readonly nStronglyOccupied: number;
  /** Number of "active" NOs (occThreshold ≥ n_p ≥ virtThreshold;
   *  default [0.05, 1.95]). High count signals multi-reference. */
  readonly nActive: number;
  /** Sum of NOON values = number of electrons (sanity check). */
  readonly totalElectrons: number;
}

export interface NOONOpts {
  /** Pre-computed orthogonalizing X = S^(-1/2). If omitted, computed
   *  from S via canonical orthogonalization. */
  readonly X?: Float64Array;
  /** n_p > occThreshold → strongly occupied. Default 1.95. */
  readonly occThreshold?: number;
  /** n_p < virtThreshold → strongly virtual. Default 0.05. */
  readonly virtThreshold?: number;
}

export function naturalOrbitalOccupations(
  D: Float64Array,
  S: Float64Array,
  opts: NOONOpts = {},
): NaturalOrbitalResult {
  const occT = opts.occThreshold ?? 1.95;
  const virtT = opts.virtThreshold ?? 0.05;
  // Determine basis size from D.
  const n = Math.round(Math.sqrt(D.length));
  if (n * n !== D.length) {
    throw new Error(`naturalOrbitalOccupations: D length ${D.length} not square`);
  }
  if (S.length !== n * n) {
    throw new Error(`naturalOrbitalOccupations: S length ${S.length} ≠ n² = ${n * n}`);
  }

  // Build Y = S^(1/2) and X = S^(-1/2) from S eigen-decomposition.
  const { Y, X } = canonicalYX(S, n, opts.X);

  // D̃ = Y · D · Y. Symmetric (since D and Y are symmetric).
  const Dtilde = transformSym(D, Y, n);

  // Diagonalize. Eigenvalues come ascending; we want descending NOON.
  const eig = eigsymmetric(Dtilde, n);
  const occAsc = eig.values;
  const Vasc = eig.vectors;

  const occupations = new Float64Array(n);
  for (let i = 0; i < n; i++) occupations[i] = occAsc[n - 1 - i]!;

  // C_NO[μ, p] = X · c̃_p (back-transform to non-orthogonal AO basis),
  // with c̃_p the p-th column of Vasc reversed.
  const C_NO = new Float64Array(n * n);
  for (let p = 0; p < n; p++) {
    const srcCol = n - 1 - p;
    for (let mu = 0; mu < n; mu++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += X[mu * n + k]! * Vasc[srcCol * n + k]!;
      }
      C_NO[mu * n + p] = s;
    }
  }

  let nStrong = 0, nActiveCount = 0, totalE = 0;
  for (let i = 0; i < n; i++) {
    const v = occupations[i]!;
    totalE += v;
    if (v > occT) nStrong++;
    else if (v > virtT) nActiveCount++;
  }
  return {
    occupations,
    C_NO,
    nStronglyOccupied: nStrong,
    nActive: nActiveCount,
    totalElectrons: totalE,
  };
}

function transformSym(M: Float64Array, X: Float64Array, n: number): Float64Array {
  const tmp = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[k * n + i]! * M[k * n + j]!;
      tmp[i * n + j] = s;
    }
  }
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += tmp[i * n + k]! * X[k * n + j]!;
      out[i * n + j] = s;
    }
  }
  return out;
}

/** Build symmetric Y = S^(1/2) and X = S^(-1/2) from S = U·diag(σ)·U^T.
 *  Y[μ,ν] = Σ_p U[μ,p] · σ_p^(1/2) · U[ν,p]; X uses σ^(-1/2) instead. */
function canonicalYX(S: Float64Array, n: number, Xopt?: Float64Array): {
  Y: Float64Array;
  X: Float64Array;
} {
  const eig = eigsymmetric(S, n);
  const sigma = eig.values;
  const U = eig.vectors;       // column-major: U[p*n+k] = k-th component of eigenvector p.
  const Y = new Float64Array(n * n);
  const X = Xopt ?? new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let ySum = 0, xSum = 0;
      for (let p = 0; p < n; p++) {
        const sig = sigma[p]!;
        if (sig <= 0) throw new Error(`canonicalYX: non-positive S eigenvalue ${sig} at index ${p}`);
        const upMu = U[p * n + mu]!;
        const upNu = U[p * n + nu]!;
        ySum += upMu * Math.sqrt(sig) * upNu;
        if (!Xopt) xSum += upMu / Math.sqrt(sig) * upNu;
      }
      Y[mu * n + nu] = ySum;
      if (!Xopt) X[mu * n + nu] = xSum;
    }
  }
  return { Y, X };
}
