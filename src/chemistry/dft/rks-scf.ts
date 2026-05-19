// ─────────────────────────────────────────────────────────────
// rks-scf.ts — Restricted Kohn-Sham DFT SCF (closed-shell).
// Tier 2 stage 2: pure LDA (Slater + VWN5).
//
// Differences from runRHFSCF:
//   • No exact exchange. Fock = h + J(D) + V_xc[ρ].
//   • Each iteration rebuilds the AO V_xc matrix by numerical
//     integration on the molecular grid:
//       V_xc[μν] = Σ_p w_p · v_xc(ρ(r_p)) · φ_μ(r_p) φ_ν(r_p).
//     ρ(r_p) = Σ_μν D_μν φ_μ(r_p) φ_ν(r_p).
//   • Energy:
//       E = Σ_μν D_μν h_μν + ½ Σ_μν D_μν J_μν + E_xc[ρ] + V_nn
//       E_xc = Σ_p w_p · ε_xc(ρ_p) · ρ_p.
//
// DIIS extrapolation is shared with HF (same FDS−SDF commutator
// criterion); it just operates on the DFT Fock matrix.
//
// API mirrors `runRHFSCF` so callers can swap drop-in for a
// quick HF→DFT comparison.
// ─────────────────────────────────────────────────────────────

import { type MolecularIntegrals } from "../cg-molecular.js";
import { eigsymmetric } from "../../manybody/dense-eig.js";
import { molecularGrid, type GridOpts, type MolecularGrid } from "./grid.js";
import {
  evalBasisOnGrid, evalBasisGradOnGrid,
  evalDensityOnGrid, evalDensityAndGradient,
  type BasisValuesOnGrid, type BasisGradientsOnGrid,
} from "./density.js";
import { evalXC, hfExchangeMixOf, type FunctionalKind } from "./functional.js";
import type { AtomSymbol } from "../atoms.js";

export interface RKSResult {
  /** Total DFT energy (Hartree), including V_nn. */
  readonly energy: number;
  /** Electronic energy = energy − Vnn. */
  readonly electronicEnergy: number;
  /** Exchange-correlation energy E_xc (Hartree). */
  readonly exchangeCorrelationEnergy: number;
  /** AO → MO coefficient matrix (row-major, column i = MO i). */
  readonly C_MO: Float64Array;
  /** Orbital energies, ascending. */
  readonly orbitalEnergies: Float64Array;
  /** AO-basis density matrix. */
  readonly D: Float64Array;
  /** Number of doubly occupied orbitals. */
  readonly nOccupied: number;
  /** Number of SCF iterations. */
  readonly iter: number;
  /** True if energy + residual tolerances were met. */
  readonly converged: boolean;
  /** Per-iteration energy trace. */
  readonly history: readonly number[];
  /** Number of grid points used. */
  readonly nGridPoints: number;
}

export interface RKSOpts {
  /** XC functional. Default "lda-svwn" (Slater + VWN5). */
  readonly functional?: FunctionalKind;
  /** Hard cap on SCF iterations. Default 100. */
  readonly maxIter?: number;
  /** Energy tolerance for convergence (Ha). Default 1e-7. */
  readonly energyTol?: number;
  /** DIIS error-vector tolerance (||e||_∞). Default 1e-5. */
  readonly residualTol?: number;
  /** Use DIIS extrapolation. Default true. */
  readonly useDIIS?: boolean;
  /** DIIS subspace size. Default 8. */
  readonly diisHistory?: number;
  /** Damping for the legacy non-DIIS path. Default 0.5. */
  readonly damping?: number;
  /** Numerical-grid options (radial, angular). Defaults in grid.ts. */
  readonly grid?: GridOpts;
  /**
   * Virtual-orbital level shift (Hartree). When > 0, adds
   * `levelShift · P_v` to the orthogonal-basis Fock matrix each
   * iteration, where P_v projects onto the previous virtual
   * subspace. Raises virtual-orbital eigenvalues, stabilizing
   * occupation across stretched-bond / near-degenerate KS problems
   * where the HOMO-LUMO gap is small. Strip on return so
   * Koopmans IPs / EAs from `orbitalEnergies` remain physical.
   * Default 0 (no shift).
   */
  readonly levelShift?: number;
}

/**
 * Run closed-shell Restricted Kohn-Sham DFT (LDA: Slater + VWN5)
 * SCF on the given molecular integrals.
 *
 * The integrals must come from `computeMolecularIntegrals` over a
 * shell list whose centers match `nucleiSymbols` (same order as
 * `integrals.nuclei`) — needed so the Becke partition can pick the
 * right per-atom radial scale.
 */
export function runRKSDFT(
  integrals: MolecularIntegrals,
  nElectrons: number,
  nucleiSymbols: readonly AtomSymbol[],
  opts: RKSOpts = {},
): RKSResult {
  if (nElectrons % 2 !== 0) {
    throw new Error(`runRKSDFT: open-shell systems require UKS (got nElectrons=${nElectrons}, odd)`);
  }
  const nOcc = nElectrons / 2;
  const n = integrals.n;
  if (nOcc > n) {
    throw new Error(`runRKSDFT: ${nElectrons} electrons in ${n} spatial orbitals — too many`);
  }

  const functional: FunctionalKind = opts.functional ?? "lda-svwn";
  const isGGA = functional !== "lda-svwn";
  const hfMix = hfExchangeMixOf(functional);
  const maxIter = opts.maxIter ?? 100;
  const eTol = opts.energyTol ?? 1e-7;
  const rTol = opts.residualTol ?? 1e-5;
  const useDIIS = opts.useDIIS ?? true;
  const diisMaxHistory = opts.diisHistory ?? 8;
  const damping = opts.damping ?? 0.5;
  const levelShift = opts.levelShift ?? 0;

  const { S_AO, h_AO, eri_AO, X, Vnn } = integrals;

  // ── Build molecular grid + precompute basis values ─────────
  const grid: MolecularGrid = molecularGrid(integrals.nuclei, nucleiSymbols, opts.grid ?? {});
  // Spherical-d transform on the grid: when integrals were built with
  // `spherical: true`, basis evaluators return Cartesian-sized arrays
  // that need T applied to match the post-transform density matrix.
  const T = integrals.sphericalT;
  const nCart = integrals.shells.length;
  const basisCart = evalBasisOnGrid(integrals.shells, grid);
  const basisVals: BasisValuesOnGrid = T
    ? { phi: applyTransformOnGrid(basisCart.phi, T, nCart, n, grid.x.length), n, nGrid: grid.x.length }
    : basisCart;
  const basisGradCart = isGGA ? evalBasisGradOnGrid(integrals.shells, grid) : null;
  const basisGrad: BasisGradientsOnGrid | null = isGGA && basisGradCart
    ? T
      ? {
          phix: applyTransformOnGrid(basisGradCart.phix, T, nCart, n, grid.x.length),
          phiy: applyTransformOnGrid(basisGradCart.phiy, T, nCart, n, grid.x.length),
          phiz: applyTransformOnGrid(basisGradCart.phiz, T, nCart, n, grid.x.length),
          n, nGrid: grid.x.length,
        }
      : basisGradCart
    : null;
  const nGrid = grid.x.length;

  // ── Initial guess: diagonalize core h ──────────────────────
  const hPrime = transformSymmetric(h_AO, X, n);
  const initial = solveFock(hPrime, X, n);
  let C_MO = initial.C_MO;
  let eps = initial.eps;
  let cPrimePrev = initial.cPrime;
  let D = densityFromC(C_MO, nOcc, n);
  let E_old = Infinity;
  let E_xc_last = 0;
  const history: number[] = [];
  let converged = false;
  let iter = 0;

  const diisF: Float64Array[] = [];
  const diisE: Float64Array[] = [];

  // Scratch buffers reused across iterations.
  const epsXc = new Float64Array(nGrid);
  const vRho   = new Float64Array(nGrid);
  const vGamma = isGGA ? new Float64Array(nGrid) : null;

  for (iter = 1; iter <= maxIter; iter++) {
    // ── J(D) (and optional K for hybrid) ────────────────────
    const J  = buildJ(D, eri_AO, n);
    const K  = hfMix > 0 ? buildK(D, eri_AO, n) : null;

    // ── Density (and ∇ρ for GGA) ────────────────────────────
    let rho: Float64Array;
    let gradX: Float64Array | null = null;
    let gradY: Float64Array | null = null;
    let gradZ: Float64Array | null = null;
    let gamma: Float64Array | null = null;
    if (isGGA && basisGrad) {
      const dg = evalDensityAndGradient(D, basisVals, basisGrad);
      rho = dg.rho;
      gradX = dg.gradX;
      gradY = dg.gradY;
      gradZ = dg.gradZ;
      gamma = dg.gamma;
    } else {
      rho = evalDensityOnGrid(D, basisVals);
    }

    // ── Functional eval ─────────────────────────────────────
    evalXC(functional, rho, gamma, epsXc, vRho, vGamma);

    // ── V_xc AO matrix ─────────────────────────────────────
    const Vxc = isGGA && basisGrad && gamma && gradX && gradY && gradZ && vGamma
      ? buildVxcGGA(vRho, vGamma, gradX, gradY, gradZ, basisVals, basisGrad, grid.w)
      : buildVxcLDA(vRho, basisVals, grid.w);

    // ── Fock = h + J + V_xc  (− hfMix · ½ K for hybrid) ────
    const F = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) F[i] = h_AO[i]! + J[i]! + Vxc[i]!;
    if (K && hfMix > 0) {
      for (let i = 0; i < n * n; i++) F[i]! -= 0.5 * hfMix * K[i]!;
    }

    // ── Energy: Σ D h + ½ Σ D J + E_xc + V_nn − ¼ hfMix Σ D K + V_nn ─
    let Eone = 0, Ej = 0, Exc = 0, Ek = 0;
    for (let i = 0; i < n * n; i++) {
      Eone += D[i]! * h_AO[i]!;
      Ej   += D[i]! * J[i]!;
      if (K) Ek += D[i]! * K[i]!;
    }
    for (let p = 0; p < nGrid; p++) Exc += grid.w[p]! * epsXc[p]! * rho[p]!;
    const Ehfx = -0.25 * hfMix * Ek;            // hybrid HF-exchange energy
    const E = Eone + 0.5 * Ej + Exc + Ehfx + Vnn;
    history.push(E);
    E_xc_last = Exc + Ehfx;

    // ── DIIS error vector + extrapolation ──────────────────
    let F_use: Float64Array = F;
    let errMax = 0;
    if (useDIIS) {
      const e = buildDIISError(F, D, S_AO, X, n);
      for (let i = 0; i < n * n; i++) {
        const a = Math.abs(e[i]!);
        if (a > errMax) errMax = a;
      }
      diisF.push(F);
      diisE.push(e);
      if (diisF.length > diisMaxHistory) {
        diisF.shift();
        diisE.shift();
      }
      if (diisF.length >= 2) {
        const c = solveDIISCoeffs(diisE, n);
        if (c !== null) {
          const F_ext = new Float64Array(n * n);
          for (let k = 0; k < diisF.length; k++) {
            const ck = c[k]!;
            const Fk = diisF[k]!;
            for (let i = 0; i < n * n; i++) F_ext[i]! += ck * Fk[i]!;
          }
          F_use = F_ext;
        }
      }
    }

    // ── Diagonalize → new C, eps, D ────────────────────────
    const FPrime = transformSymmetric(F_use, X, n);

    // ── Optional virtual-orbital level shift ──────────────
    // F'_lifted = F' + levelShift · Σ_{p ≥ nOcc} c'_p ⊗ c'_p^T
    // Using the PREVIOUS iteration's c'_v as the virtual subspace.
    // Raises virtual KS eigenvalues by `levelShift`, stabilizing
    // occupation for stretched-bond / near-degenerate cases.
    if (levelShift > 0) {
      for (let p = nOcc; p < n; p++) {
        for (let i = 0; i < n; i++) {
          const cpi = cPrimePrev[p * n + i]!;
          if (cpi === 0) continue;
          const shifted_cpi = levelShift * cpi;
          for (let j = 0; j < n; j++) {
            FPrime[i * n + j]! += shifted_cpi * cPrimePrev[p * n + j]!;
          }
        }
      }
    }

    const sol = solveFock(FPrime, X, n);
    C_MO = sol.C_MO;
    eps = sol.eps;
    cPrimePrev = sol.cPrime;
    const D_new = densityFromC(C_MO, nOcc, n);

    let dNorm = 0;
    for (let i = 0; i < n * n; i++) {
      const d = D_new[i]! - D[i]!;
      dNorm += d * d;
    }
    dNorm = Math.sqrt(dNorm);
    if (useDIIS) {
      D = D_new;
    } else {
      for (let i = 0; i < n * n; i++) D[i] = damping * D_new[i]! + (1 - damping) * D[i]!;
    }

    const residOk = useDIIS ? errMax < rTol : dNorm < rTol;
    if (Math.abs(E - E_old) < eTol && residOk && iter >= 2) {
      converged = true;
      E_old = E;
      break;
    }
    E_old = E;
  }

  // ── Strip level-shift from reported virtual eigenvalues. ──
  // The lifted F's virtual eigenvalues are eps_true + shift. At
  // convergence the eigenvectors are correct (the shift only changed
  // eigenvalues, not eigenvectors), so removing the shift restores
  // the physical Koopmans IPs / EAs. Energies are field-functional
  // expressions of D and don't depend on eps, so they stay correct.
  if (levelShift > 0) {
    const epsOut = new Float64Array(eps);
    for (let i = nOcc; i < n; i++) epsOut[i]! -= levelShift;
    eps = epsOut;
  }

  return {
    energy: E_old,
    electronicEnergy: E_old - Vnn,
    exchangeCorrelationEnergy: E_xc_last,
    C_MO,
    orbitalEnergies: eps,
    D,
    nOccupied: nOcc,
    iter,
    converged,
    history,
    nGridPoints: nGrid,
  };
}

// ── Helpers ─────────────────────────────────────────────────

function transformSymmetric(M: Float64Array, X: Float64Array, n: number): Float64Array {
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

function solveFock(FPrime: Float64Array, X: Float64Array, n: number): {
  C_MO: Float64Array;
  /** Orthogonal-basis eigenvectors, column-major: cPrime[c * n + k]
   *  is the k-th component of the c-th eigenvector. C_MO = X · cPrime. */
  cPrime: Float64Array;
  eps: Float64Array;
} {
  const eig = eigsymmetric(FPrime, n);
  const C_MO = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += X[r * n + k]! * eig.vectors[c * n + k]!;
      C_MO[r * n + c] = s;
    }
  }
  return { C_MO, cPrime: eig.vectors, eps: eig.values };
}

function densityFromC(C: Float64Array, nOcc: number, n: number): Float64Array {
  const D = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let i = 0; i < nOcc; i++) s += C[mu * n + i]! * C[nu * n + i]!;
      D[mu * n + nu] = 2 * s;
    }
  }
  return D;
}

/** Pure Coulomb J — no exchange (DFT separates J and v_xc). */
export function buildJ(D: Float64Array, eri_AO: Float64Array, n: number): Float64Array {
  const J = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          const Dls = D[la * n + si]!;
          if (Dls === 0) continue;
          s += Dls * eri_AO[((mu * n + nu) * n + la) * n + si]!;
        }
      }
      J[mu * n + nu] = s;
    }
  }
  return J;
}

/** HF exchange K[μν] = Σ_{λσ} D_{λσ} (μλ|νσ) — needed for hybrid functionals. */
export function buildK(D: Float64Array, eri_AO: Float64Array, n: number): Float64Array {
  const K = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          const Dls = D[la * n + si]!;
          if (Dls === 0) continue;
          s += Dls * eri_AO[((mu * n + la) * n + nu) * n + si]!;
        }
      }
      K[mu * n + nu] = s;
    }
  }
  return K;
}

/**
 * LDA V_xc:  V[μν] = Σ_p w_p · v_ρ(ρ_p) · φ_μ(r_p) φ_ν(r_p).
 * Cost O(n² · nGrid). Standard hot loop.
 */
export function buildVxcLDA(
  vRho: Float64Array,
  basis: { phi: Float64Array; n: number; nGrid: number },
  weights: Float64Array,
): Float64Array {
  const { phi, n, nGrid } = basis;
  const V = new Float64Array(n * n);
  for (let p = 0; p < nGrid; p++) {
    const c = weights[p]! * vRho[p]!;
    if (c === 0) continue;
    const off = p * n;
    for (let mu = 0; mu < n; mu++) {
      const phimu = phi[off + mu]!;
      if (phimu === 0) continue;
      const cmu = c * phimu;
      for (let nu = 0; nu <= mu; nu++) {
        V[mu * n + nu]! += cmu * phi[off + nu]!;
      }
    }
  }
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < mu; nu++) {
      V[nu * n + mu] = V[mu * n + nu]!;
    }
  }
  return V;
}

/**
 * GGA V_xc:
 *   V[μν] = Σ_p w_p {
 *             v_ρ φ_μ φ_ν
 *           + 2 v_γ [(∇ρ · ∇φ_μ) φ_ν + φ_μ (∇ρ · ∇φ_ν)]
 *           }
 *
 * Cost O(n² · nGrid) — same as LDA, just more flops per (μν, p) pair.
 * Per-grid scratch (∇ρ · ∇φ_μ) is computed inline to save on memory.
 */
function buildVxcGGA(
  vRho: Float64Array,
  vGamma: Float64Array,
  gradX: Float64Array,
  gradY: Float64Array,
  gradZ: Float64Array,
  basis: BasisValuesOnGrid,
  basisGrad: BasisGradientsOnGrid,
  weights: Float64Array,
): Float64Array {
  const { phi, n, nGrid } = basis;
  const { phix, phiy, phiz } = basisGrad;
  const V = new Float64Array(n * n);
  // Per-point scratch for (∇ρ · ∇φ_μ).
  const dot = new Float64Array(n);
  for (let p = 0; p < nGrid; p++) {
    const w = weights[p]!;
    const cRho = w * vRho[p]!;
    const cGam2 = 2 * w * vGamma[p]!;
    if (cRho === 0 && cGam2 === 0) continue;
    const off = p * n;
    const rx = gradX[p]!, ry = gradY[p]!, rz = gradZ[p]!;
    // Precompute ∇ρ · ∇φ_μ for every μ at this grid point.
    for (let mu = 0; mu < n; mu++) {
      dot[mu] = rx * phix[off + mu]! + ry * phiy[off + mu]! + rz * phiz[off + mu]!;
    }
    for (let mu = 0; mu < n; mu++) {
      const phimu = phi[off + mu]!;
      const dmu = dot[mu]!;
      // Pure-density contribution + GGA cross-term ∇ρ · ∇φ_μ · φ_ν.
      const aMu = cRho * phimu + cGam2 * dmu;
      // Symmetric counterpart: φ_μ · ∇ρ · ∇φ_ν inside the inner loop.
      for (let nu = 0; nu <= mu; nu++) {
        const phinu = phi[off + nu]!;
        const dnu = dot[nu]!;
        V[mu * n + nu]! += aMu * phinu + cGam2 * phimu * dnu;
      }
    }
  }
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < mu; nu++) {
      V[nu * n + mu] = V[mu * n + nu]!;
    }
  }
  return V;
}

function buildDIISError(F: Float64Array, D: Float64Array, S: Float64Array, X: Float64Array, n: number): Float64Array {
  const FD  = matmul(F, D, n);
  const FDS = matmul(FD, S, n);
  const SD  = matmul(S, D, n);
  const SDF = matmul(SD, F, n);
  const eAO = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) eAO[i] = FDS[i]! - SDF[i]!;
  return transformGeneral(eAO, X, n);
}

function transformGeneral(M: Float64Array, X: Float64Array, n: number): Float64Array {
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

function matmul(A: Float64Array, B: Float64Array, n: number): Float64Array {
  const C = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i * n + k]!;
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) {
        C[i * n + j]! += aik * B[k * n + j]!;
      }
    }
  }
  return C;
}

function solveDIISCoeffs(diisE: Float64Array[], n: number): Float64Array | null {
  const m = diisE.length;
  const dim = m + 1;
  const A = new Float64Array(dim * dim);
  for (let i = 0; i < m; i++) {
    for (let j = i; j < m; j++) {
      let dot = 0;
      const ei = diisE[i]!;
      const ej = diisE[j]!;
      for (let k = 0; k < n * n; k++) dot += ei[k]! * ej[k]!;
      A[i * dim + j] = dot;
      A[j * dim + i] = dot;
    }
    A[i * dim + m] = -1;
    A[m * dim + i] = -1;
  }
  A[m * dim + m] = 0;
  const b = new Float64Array(dim);
  b[m] = -1;
  for (let col = 0; col < dim; col++) {
    let piv = col;
    let pivAbs = Math.abs(A[col * dim + col]!);
    for (let r = col + 1; r < dim; r++) {
      const a = Math.abs(A[r * dim + col]!);
      if (a > pivAbs) { pivAbs = a; piv = r; }
    }
    if (pivAbs < 1e-14) return null;
    if (piv !== col) {
      for (let c = 0; c < dim; c++) {
        const t = A[col * dim + c]!;
        A[col * dim + c] = A[piv * dim + c]!;
        A[piv * dim + c] = t;
      }
      const tb = b[col]!; b[col] = b[piv]!; b[piv] = tb;
    }
    const pivVal = A[col * dim + col]!;
    for (let r = 0; r < dim; r++) {
      if (r === col) continue;
      const f = A[r * dim + col]! / pivVal;
      if (f === 0) continue;
      for (let c = col; c < dim; c++) {
        A[r * dim + c]! -= f * A[col * dim + c]!;
      }
      b[r]! -= f * b[col]!;
    }
  }
  const c = new Float64Array(m);
  for (let i = 0; i < m; i++) c[i] = b[i]! / A[i * dim + i]!;
  return c;
}


/**
 * Apply the Cartesian → spherical-d transform T (row-major n_sph × n_cart)
 * to a per-grid AO array. φ_sph[g, p] = Σ_μ T[p, μ]·φ_cart[g, μ].
 */
function applyTransformOnGrid(
  phiCart: Float64Array,
  T: Float64Array,
  nCart: number,
  nSph: number,
  nGrid: number,
): Float64Array {
  const out = new Float64Array(nGrid * nSph);
  for (let g = 0; g < nGrid; g++) {
    const offC = g * nCart;
    const offS = g * nSph;
    for (let p = 0; p < nSph; p++) {
      let s = 0;
      const Trow = p * nCart;
      for (let mu = 0; mu < nCart; mu++) s += T[Trow + mu]! * phiCart[offC + mu]!;
      out[offS + p] = s;
    }
  }
  return out;
}
