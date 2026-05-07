// ─────────────────────────────────────────────────────────────
// tda-dft.ts — TDA-DFT (Tamm-Dancoff, only A) and full TDDFT
// (Casida, A + B) singlet excitation energies. Tier 2 stages
// 9 + 9b.
//
// Generalizes runCIS to Kohn-Sham reference + HF mixing + the
// LDA XC kernel. The closed-shell singlet matrix elements:
//
//   A_{ia,jb} = (ε_a − ε_i) δ_ij δ_ab
//             + 2·(ia|jb)                    (Hartree / J)
//             − hfMix·(ij|ab)                (HF exchange × hfMix)
//             + 2·(ia|f_xc|jb)               (XC kernel; LDA only)
//   B_{ia,jb} = 2·(ia|jb)                    (same Coulomb piece)
//             − hfMix·(ib|aj)                (B's exchange — different
//                                              index permutation than A's)
//             + 2·(ia|f_xc|jb)               (LDA kernel commutes; same
//                                              integrand as A's kernel)
//
// `hfMix` follows from the SCF method:
//   • HF                  → hfMix = 1, no XC kernel  (= CIS / TDHF)
//   • LDA (lda-svwn)      → hfMix = 0, XC kernel from evalXCKernelLDA
//   • B3LYP-style hybrids → hfMix = 0.20, XC kernel ... NOT YET (GGA)
//
// `runTDA` solves only A · X = ω · X (Tamm-Dancoff approximation).
// `runTDDFT` solves the full Casida problem
//   (A − B)·(A + B) · Z = ω² · Z,
// which factors cleanly via the matrix square root M ≡ S·(A+B)·S
// where S = (A−B)^(1/2). M is real symmetric → eigsymmetric on M
// gives ω² directly. For pure DFT (hfMix = 0) the (A − B) block
// is diagonal and S is just diag(√(ε_a − ε_i)).
//
// Scope shipped this commit:
//   ✓ HF + LDA TDA singlet (stage 9; reproduces runCIS for HF).
//   ✓ HF + LDA full TDDFT singlet (stage 9b).
//   ✗ GGA / hybrid kernels (BVWN5, BLYP, B3VWN5, B3LYP5) require
//     f_ρρ + f_ργ + f_γγ tensors and basis-Hessian-style
//     integrals — the runners throw a clear "needs GGA TDA"
//     message for those. Follow-up.
//   ✗ Triplet TDA / TDDFT — closed-shell triplet kernel
//     f_xc^triplet differs from f_xc^singlet. Deferred; runCIS
//     still ships triplet for HF.
//
// (ia|f_xc|jb) is computed on the molecular grid:
//   (ia|f_xc|jb) = Σ_p w_p · f_xc(ρ_p) · ψ_ia(r_p) · ψ_jb(r_p)
// with ψ_pq(r) ≡ φ_p^MO(r) · φ_q^MO(r). We pre-build all ψ_ia
// values on the grid (n_occ · n_virt · nGrid floats) and contract
// against ψ_jb with the f_xc-weighted measure — O((n_occ · n_virt)²
// · nGrid). For STO-3G H₂O this is ~1.7M ops; trivial.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { AtomSymbol } from "./atoms.js";
import { transformERIToMO } from "./mp2.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { molecularGrid, type GridOpts } from "./dft/grid.js";
import { evalBasisOnGrid, evalDensityOnGrid } from "./dft/density.js";
import { evalXCKernelLDA, type FunctionalKind, hfExchangeMixOf } from "./dft/functional.js";
import type { HFLike } from "./cis.js";

/** SCF method to use as the TDA reference. "hf" → CIS; otherwise a DFT functional. */
export type TDAMethod = "hf" | FunctionalKind;

export interface TDAOpts {
  /** SCF method (default "hf"). */
  readonly method?: TDAMethod;
  /** Per-atom symbols, in the same order as `integrals.nuclei`. Required
   *  for DFT methods (used to rebuild the molecular grid for the XC kernel). */
  readonly nucleiSymbols?: readonly AtomSymbol[];
  /** Numerical-grid options (must match the SCF). */
  readonly grid?: GridOpts;
  /** Number of lowest singlet roots to keep. Default: all. */
  readonly nRoots?: number;
}

export interface TDAResult {
  readonly nOccupied: number;
  readonly nVirtual: number;
  /** Singlet excitation energies (Hartree, ascending). */
  readonly singletEnergies: Float64Array;
  /** Singlet amplitudes c_{ia} per root, row-major
   *  amps[r·(nOcc·nVirt) + i·nVirt + a]. */
  readonly singletAmplitudes: Float64Array;
  /** HF-exchange mixing fraction used (0 for pure DFT, 1 for HF). */
  readonly hfMix: number;
  /** Whether the LDA XC kernel was used. */
  readonly usedXCKernel: boolean;
}

interface TDABuildResult {
  readonly nOcc: number;
  readonly nVirt: number;
  readonly dim: number;
  readonly hfMix: number;
  readonly usedXCKernel: boolean;
  /** A matrix (dim × dim, row-major). */
  readonly A: Float64Array;
  /** B matrix (dim × dim, row-major) — full TDDFT only. Null for TDA-only paths. */
  readonly B: Float64Array | null;
}

/**
 * Build the singlet A (and optionally B) Casida matrices on top
 * of an HF or KS reference. Shared by `runTDA` (A only) and
 * `runTDDFT` (A + B).
 */
function buildTDABlocks(
  integrals: MolecularIntegrals,
  hf: HFLike,
  opts: TDAOpts,
  buildB: boolean,
): TDABuildResult {
  const method: TDAMethod = opts.method ?? "hf";
  const n = integrals.n;
  const nOcc = hf.nOccupied;
  const nVirt = n - nOcc;
  const dim = nOcc * nVirt;
  if (nVirt <= 0) throw new Error(`buildTDABlocks: no virtual orbitals (n=${n}, nOcc=${nOcc})`);

  const isHF = method === "hf";
  const isLDA = method === "lda-svwn";
  if (!isHF && !isLDA) {
    throw new Error(
      `buildTDABlocks: method '${method}' requires the GGA / hybrid XC kernel ` +
      `(f_ρρ + f_ργ + f_γγ + basis Hessians). Only 'hf' and 'lda-svwn' ` +
      `are supported in this commit — see tda-dft.ts header.`,
    );
  }
  const hfMix = isHF ? 1.0 : hfExchangeMixOf(method as FunctionalKind);
  const usedXCKernel = !isHF;

  // ── MO-basis ERI + chemist-notation accessor. ──────────────
  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);
  const eri = (p: number, q: number, r: number, s: number): number =>
    eri_MO[((p * n + q) * n + r) * n + s]!;

  // ── XC kernel piece (DFT only): build (ia|f_xc|jb). ────────
  let K_xc: Float64Array | null = null;
  if (usedXCKernel) {
    if (!opts.nucleiSymbols) {
      throw new Error("buildTDABlocks: DFT method requires opts.nucleiSymbols for grid rebuild.");
    }
    const grid = molecularGrid(integrals.nuclei, opts.nucleiSymbols, opts.grid ?? {});
    const basis = evalBasisOnGrid(integrals.shells, grid);
    const rho = evalDensityOnGrid(hf.D, basis);
    const fxc = evalXCKernelLDA(rho);
    const nGrid = grid.x.length;
    const phi = basis.phi;
    const phiMO = new Float64Array(nGrid * n);
    for (let g = 0; g < nGrid; g++) {
      const off = g * n;
      for (let p = 0; p < n; p++) {
        let s = 0;
        for (let mu = 0; mu < n; mu++) s += phi[off + mu]! * hf.C_MO[mu * n + p]!;
        phiMO[off + p] = s;
      }
    }
    const psi = new Float64Array(dim * nGrid);
    for (let i = 0; i < nOcc; i++) {
      for (let a = 0; a < nVirt; a++) {
        const aMO = nOcc + a;
        const ia = i * nVirt + a;
        for (let g = 0; g < nGrid; g++) {
          psi[ia * nGrid + g] = phiMO[g * n + i]! * phiMO[g * n + aMO]!;
        }
      }
    }
    const psiW = new Float64Array(dim * nGrid);
    for (let ia = 0; ia < dim; ia++) {
      for (let g = 0; g < nGrid; g++) {
        psiW[ia * nGrid + g] = psi[ia * nGrid + g]! * grid.w[g]! * fxc[g]!;
      }
    }
    K_xc = new Float64Array(dim * dim);
    for (let ia = 0; ia < dim; ia++) {
      for (let jb = 0; jb < dim; jb++) {
        let s = 0;
        for (let g = 0; g < nGrid; g++) {
          s += psiW[ia * nGrid + g]! * psi[jb * nGrid + g]!;
        }
        K_xc[ia * dim + jb] = s;
      }
    }
  }

  // ── Assemble A (and optionally B). ─────────────────────────
  const A = new Float64Array(dim * dim);
  const B = buildB ? new Float64Array(dim * dim) : null;
  const eps = hf.orbitalEnergies;
  for (let i = 0; i < nOcc; i++) {
    for (let a = 0; a < nVirt; a++) {
      const ai = nOcc + a;
      const eDiag = eps[ai]! - eps[i]!;
      const row = i * nVirt + a;
      for (let j = 0; j < nOcc; j++) {
        for (let b = 0; b < nVirt; b++) {
          const bj = nOcc + b;
          const col = j * nVirt + b;
          const diag = (i === j && a === b) ? eDiag : 0;
          const coul = 2 * eri(i, ai, j, bj);                 // 2(ia|jb) — same in A and B
          const exchA = hfMix !== 0 ? hfMix * eri(i, j, ai, bj) : 0;   // (ij|ab)
          const kxc   = K_xc ? 2 * K_xc[row * dim + col]! : 0;
          A[row * dim + col] = diag + coul - exchA + kxc;
          if (B) {
            // B's exchange piece uses (ib|aj) — different index permutation.
            const exchB = hfMix !== 0 ? hfMix * eri(i, bj, ai, j) : 0;
            B[row * dim + col] = coul - exchB + kxc;
          }
        }
      }
    }
  }

  return { nOcc, nVirt, dim, hfMix, usedXCKernel, A, B };
}

/**
 * Run TDA on an HF or DFT reference. Singlet sector only — see
 * the file header for the deferred triplet / GGA pieces.
 *
 * For `method = "hf"`, this is identical to `runCIS(...).singlet`
 * up to floating-point noise. For `method = "lda-svwn"`, it adds
 * the LDA XC kernel contribution. Other DFT functionals throw.
 */
export function runTDA(
  integrals: MolecularIntegrals,
  hf: HFLike,
  opts: TDAOpts = {},
): TDAResult {
  const built = buildTDABlocks(integrals, hf, opts, false);
  const { nOcc, nVirt, dim, hfMix, usedXCKernel, A } = built;
  const nRoots = Math.min(opts.nRoots ?? dim, dim);

  const eig = eigsymmetric(A, dim);
  const energies = new Float64Array(nRoots);
  const amps = new Float64Array(nRoots * dim);
  for (let r = 0; r < nRoots; r++) {
    energies[r] = eig.values[r]!;
    for (let k = 0; k < dim; k++) {
      amps[r * dim + k] = eig.vectors[r * dim + k]!;
    }
  }
  return {
    nOccupied: nOcc, nVirtual: nVirt,
    singletEnergies: energies, singletAmplitudes: amps,
    hfMix, usedXCKernel,
  };
}

export interface TDDFTResult {
  readonly nOccupied: number;
  readonly nVirtual: number;
  /** Singlet excitation energies (Hartree, ascending). */
  readonly singletEnergies: Float64Array;
  /** Per-root Z = (A−B)^(1/2)·X amplitudes (row-major). For pure
   *  DFT (hfMix = 0), (A−B) is diagonal so Z = sqrt(ε)·X exactly. */
  readonly singletAmplitudes: Float64Array;
  readonly hfMix: number;
  readonly usedXCKernel: boolean;
}

/**
 * Run full closed-shell singlet TDDFT (Casida, RPA): solves
 *   (A − B)·(A + B) Z = ω² Z
 * by forming the symmetric matrix
 *   M = (A − B)^(1/2) · (A + B) · (A − B)^(1/2)
 * and diagonalizing it. Eigenvalues are ω²; the excitation
 * energies are √ω².
 *
 * For `method = "hf"` this is RPA / TDHF.
 * For `method = "lda-svwn"` this is full LDA TDDFT.
 * Other DFT functionals throw the same way as `runTDA`.
 */
export function runTDDFT(
  integrals: MolecularIntegrals,
  hf: HFLike,
  opts: TDAOpts = {},
): TDDFTResult {
  const built = buildTDABlocks(integrals, hf, opts, true);
  const { nOcc, nVirt, dim, hfMix, usedXCKernel, A, B } = built;
  if (!B) throw new Error("runTDDFT: internal — B matrix missing");
  const nRoots = Math.min(opts.nRoots ?? dim, dim);

  // M_minus = A − B; M_plus = A + B.
  const Mminus = new Float64Array(dim * dim);
  const Mplus  = new Float64Array(dim * dim);
  for (let k = 0; k < dim * dim; k++) {
    Mminus[k] = A[k]! - B[k]!;
    Mplus[k]  = A[k]! + B[k]!;
  }

  // S = (A − B)^(1/2). For pure DFT (hfMix = 0) M_minus is exactly
  // the orbital-energy diagonal — sqrt is element-wise. Otherwise
  // diagonalize and rebuild.
  const S = matrixSqrtSymmetric(Mminus, dim);

  // M = S · M_plus · S, real symmetric.
  const tmp = new Float64Array(dim * dim);
  matmul(S, Mplus, tmp, dim);
  const M = new Float64Array(dim * dim);
  matmul(tmp, S, M, dim);
  // Symmetrize against any FP-noise asymmetry.
  for (let i = 0; i < dim; i++) {
    for (let j = i + 1; j < dim; j++) {
      const v = 0.5 * (M[i * dim + j]! + M[j * dim + i]!);
      M[i * dim + j] = v;
      M[j * dim + i] = v;
    }
  }

  const eig = eigsymmetric(M, dim);
  const energies = new Float64Array(nRoots);
  const amps = new Float64Array(nRoots * dim);
  for (let r = 0; r < nRoots; r++) {
    const om2 = eig.values[r]!;
    if (om2 < 0) {
      // Triplet / singlet instability would land here — the system's
      // RPA reference is unstable. Keep the imaginary frequency as
      // a NEGATIVE energy so callers can spot it.
      energies[r] = -Math.sqrt(-om2);
    } else {
      energies[r] = Math.sqrt(om2);
    }
    for (let k = 0; k < dim; k++) amps[r * dim + k] = eig.vectors[r * dim + k]!;
  }
  return {
    nOccupied: nOcc, nVirtual: nVirt,
    singletEnergies: energies, singletAmplitudes: amps,
    hfMix, usedXCKernel,
  };
}

// ── Linear-algebra helpers ─────────────────────────────────────

/** Real-symmetric square root via eigendecomposition. */
function matrixSqrtSymmetric(M: Float64Array, n: number): Float64Array {
  const eig = eigsymmetric(M, n);
  const sqrtL = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lam = eig.values[i]!;
    if (lam < -1e-9) {
      throw new Error(
        `matrixSqrtSymmetric: negative eigenvalue ${lam.toExponential(3)} ` +
        `— input matrix isn't positive semidefinite (closed-shell instability?)`,
      );
    }
    sqrtL[i] = Math.sqrt(Math.max(lam, 0));
  }
  // M^(1/2) = U · diag(√λ) · U^T. eig.vectors is column-major:
  // vectors[col·n + row] is the row-th component of column-th eigvec.
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += eig.vectors[k * n + i]! * sqrtL[k]! * eig.vectors[k * n + j]!;
      }
      out[i * n + j] = s;
      if (i !== j) out[j * n + i] = s;
    }
  }
  return out;
}

/** Plain n³ row-major matrix multiplication: C = A · B. */
function matmul(A: Float64Array, B: Float64Array, C: Float64Array, n: number): void {
  C.fill(0);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i * n + k]!;
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) {
        C[i * n + j]! += aik * B[k * n + j]!;
      }
    }
  }
}
