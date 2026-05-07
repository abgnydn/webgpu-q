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
import { dipole_cg } from "./integrals-cg.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { molecularGrid, type GridOpts } from "./dft/grid.js";
import {
  evalBasisOnGrid, evalBasisGradOnGrid,
  evalDensityOnGrid, evalDensityAndGradient,
} from "./dft/density.js";
import {
  evalXC, evalXCKernel, evalXCKernelLDA,
  type FunctionalKind, hfExchangeMixOf,
} from "./dft/functional.js";
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
  /** Dimensionless oscillator strengths f_n per root. Sum over
   *  all roots is bounded by the number of valence electrons
   *  (Thomas-Reiche-Kuhn sum rule, partial within the n_occ ·
   *  n_virt singles manifold). */
  readonly oscillatorStrengths: Float64Array;
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
  const isGGA = !isHF && !isLDA;     // bvwn5 / blyp / b3vwn5 / b3lyp5
  const hfMix = isHF ? 1.0 : hfExchangeMixOf(method as FunctionalKind);
  const usedXCKernel = !isHF;

  // ── MO-basis ERI + chemist-notation accessor. ──────────────
  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);
  const eri = (p: number, q: number, r: number, s: number): number =>
    eri_MO[((p * n + q) * n + r) * n + s]!;

  // ── XC kernel piece (DFT only): build (ia|f_xc|jb). ────────
  // For LDA: K_xc[ia, jb] = ∫ w · ψ_ia · f_RR · ψ_jb dr.
  // For GGA / hybrid the kernel acquires four additional pieces
  // (Maxwell-symmetrized over (ia, jb) via the integrand structure):
  //   K_xc[ia, jb] = ∫ w · {
  //       ψ_ia · f_RR · ψ_jb
  //     + 2·ψ_ia · f_RG · α_jb
  //     + 2·α_ia · f_RG · ψ_jb
  //     + 4·α_ia · f_GG · α_jb
  //     + 2·v_γ · ∇ψ_ia·∇ψ_jb
  //   } dr
  // with ψ_ia(r) = φ_i^MO(r)·φ_a^MO(r) and α_ia(r) = ∇ρ·∇ψ_ia.
  let K_xc: Float64Array | null = null;
  if (usedXCKernel) {
    if (!opts.nucleiSymbols) {
      throw new Error("buildTDABlocks: DFT method requires opts.nucleiSymbols for grid rebuild.");
    }
    const grid = molecularGrid(integrals.nuclei, opts.nucleiSymbols, opts.grid ?? {});
    const basisCart = evalBasisOnGrid(integrals.shells, grid);
    const nGrid = grid.x.length;
    // Spherical-d transform on the grid: φ_sph[g, p] = Σ_μ T[p, μ]·φ_cart[g, μ].
    // T is row-major (n_sph × n_cart); when null, basis already matches `n`.
    const T = integrals.sphericalT;
    const nCart = integrals.shells.length;
    const basis = T
      ? { phi: applyTransformToGridValues(basisCart.phi, T, nCart, n, nGrid), n, nGrid }
      : basisCart;
    const phi = basis.phi;

    // MO orbital values on grid: φ_p^MO(g) = Σ_μ C_μp · φ_μ^AO(g).
    const phiMO = new Float64Array(nGrid * n);
    for (let g = 0; g < nGrid; g++) {
      const off = g * n;
      for (let p = 0; p < n; p++) {
        let s = 0;
        for (let mu = 0; mu < n; mu++) s += phi[off + mu]! * hf.C_MO[mu * n + p]!;
        phiMO[off + p] = s;
      }
    }

    // ψ_ia(g) = φ_i^MO(g) · φ_a^MO(g).
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

    if (isLDA) {
      // ── LDA: single-piece kernel. ───────────────────────────
      const rho = evalDensityOnGrid(hf.D, basis);
      const fxc = evalXCKernelLDA(rho);
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
    } else if (isGGA) {
      // ── GGA / hybrid: 5-piece kernel. ───────────────────────
      const basisGradCart = evalBasisGradOnGrid(integrals.shells, grid);
      const basisGrad = T
        ? {
            phix: applyTransformToGridValues(basisGradCart.phix, T, nCart, n, nGrid),
            phiy: applyTransformToGridValues(basisGradCart.phiy, T, nCart, n, nGrid),
            phiz: applyTransformToGridValues(basisGradCart.phiz, T, nCart, n, nGrid),
            n, nGrid,
          }
        : basisGradCart;
      const dg = evalDensityAndGradient(hf.D, basis, basisGrad);
      const rho = dg.rho;
      const gamma = dg.gamma;
      const gradX = dg.gradX, gradY = dg.gradY, gradZ = dg.gradZ;

      // v_xc first derivatives (we need v_γ for the last kernel term).
      const epsTmp = new Float64Array(nGrid);
      const vRho   = new Float64Array(nGrid);
      const vGamma = new Float64Array(nGrid);
      evalXC(method as FunctionalKind, rho, gamma, epsTmp, vRho, vGamma);

      // v_xc second derivatives.
      const ker = evalXCKernel(method as FunctionalKind, rho, gamma);

      // MO orbital gradients on the grid: ∇φ_p^MO(g) = Σ_μ C_μp · ∇φ_μ^AO(g).
      const phix = basisGrad.phix, phiy = basisGrad.phiy, phiz = basisGrad.phiz;
      const phiMOx = new Float64Array(nGrid * n);
      const phiMOy = new Float64Array(nGrid * n);
      const phiMOz = new Float64Array(nGrid * n);
      for (let g = 0; g < nGrid; g++) {
        const off = g * n;
        for (let p = 0; p < n; p++) {
          let sx = 0, sy = 0, sz = 0;
          for (let mu = 0; mu < n; mu++) {
            const c = hf.C_MO[mu * n + p]!;
            sx += phix[off + mu]! * c;
            sy += phiy[off + mu]! * c;
            sz += phiz[off + mu]! * c;
          }
          phiMOx[off + p] = sx;
          phiMOy[off + p] = sy;
          phiMOz[off + p] = sz;
        }
      }

      // ∇ψ_ia(g) = ∇φ_i^MO · φ_a^MO + φ_i^MO · ∇φ_a^MO  (3 components).
      // α_ia(g) = ∇ρ(g) · ∇ψ_ia(g).
      const dPsiX = new Float64Array(dim * nGrid);
      const dPsiY = new Float64Array(dim * nGrid);
      const dPsiZ = new Float64Array(dim * nGrid);
      const alpha = new Float64Array(dim * nGrid);
      for (let i = 0; i < nOcc; i++) {
        for (let a = 0; a < nVirt; a++) {
          const aMO = nOcc + a;
          const ia = i * nVirt + a;
          for (let g = 0; g < nGrid; g++) {
            const off = g * n;
            const phi_i = phiMO[off + i]!, phi_a = phiMO[off + aMO]!;
            const dx = phiMOx[off + i]! * phi_a + phi_i * phiMOx[off + aMO]!;
            const dy = phiMOy[off + i]! * phi_a + phi_i * phiMOy[off + aMO]!;
            const dz = phiMOz[off + i]! * phi_a + phi_i * phiMOz[off + aMO]!;
            dPsiX[ia * nGrid + g] = dx;
            dPsiY[ia * nGrid + g] = dy;
            dPsiZ[ia * nGrid + g] = dz;
            alpha[ia * nGrid + g] = gradX[g]! * dx + gradY[g]! * dy + gradZ[g]! * dz;
          }
        }
      }

      // Build K_xc[ia, jb] by accumulating the 5 contributions.
      // Pre-multiply per-grid-point factors once, then dot.
      K_xc = new Float64Array(dim * dim);
      for (let ia = 0; ia < dim; ia++) {
        for (let jb = 0; jb < dim; jb++) {
          let s = 0;
          for (let g = 0; g < nGrid; g++) {
            const w = grid.w[g]!;
            const psi_ia = psi[ia * nGrid + g]!;
            const psi_jb = psi[jb * nGrid + g]!;
            const a_ia  = alpha[ia * nGrid + g]!;
            const a_jb  = alpha[jb * nGrid + g]!;
            const dotDpsi =
                dPsiX[ia * nGrid + g]! * dPsiX[jb * nGrid + g]!
              + dPsiY[ia * nGrid + g]! * dPsiY[jb * nGrid + g]!
              + dPsiZ[ia * nGrid + g]! * dPsiZ[jb * nGrid + g]!;
            s += w * (
                ker.fRR[g]! * psi_ia * psi_jb
              + 2 * ker.fRG[g]! * (psi_ia * a_jb + a_ia * psi_jb)
              + 4 * ker.fGG[g]! * a_ia * a_jb
              + 2 * vGamma[g]! * dotDpsi
            );
          }
          K_xc[ia * dim + jb] = s;
        }
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
  const oscillatorStrengths = computeOscillatorStrengths(integrals, hf, energies, amps, null);
  return {
    nOccupied: nOcc, nVirtual: nVirt,
    singletEnergies: energies, singletAmplitudes: amps,
    oscillatorStrengths,
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
  /** Dimensionless oscillator strengths f_n per root. */
  readonly oscillatorStrengths: Float64Array;
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
  const oscillatorStrengths = computeOscillatorStrengths(integrals, hf, energies, amps, S);
  return {
    nOccupied: nOcc, nVirtual: nVirt,
    singletEnergies: energies, singletAmplitudes: amps,
    oscillatorStrengths,
    hfMix, usedXCKernel,
  };
}

// ── Oscillator strengths ───────────────────────────────────────
//
// For closed-shell singlet excitations:
//   T_axis_n = √2 · Σ_ia c_ia · ⟨φ_i^MO | r_axis | φ_a^MO⟩
//   f_n = (2/3) · ω_n · Σ_axis |T_axis_n|²
//       = (4/3) · ω_n · Σ_axis (Σ_ia c_ia · μ_ia^MO_axis)²    (TDA)
//       = (4/3) ·       Σ_axis (Σ_ia (S·Z')_ia · μ_ia^MO_axis)²  (TDDFT)
//
// (For TDDFT the explicit ω cancels with the Casida normalization
// of (X+Y) = (1/√ω)·S·Z'. In the TDA limit (S=√A, Z'=X), this
// reduces to the TDA formula.)

/**
 * Build oscillator strengths from raw eigenvectors.
 * - For TDA: pass `S = null`; `amplitudes` are X.
 * - For TDDFT: pass S = (A − B)^(1/2); `amplitudes` are Z'.
 */
function computeOscillatorStrengths(
  integrals: MolecularIntegrals,
  hf: HFLike,
  energies: Float64Array,
  amplitudes: Float64Array,
  S: Float64Array | null,
): Float64Array {
  const n = integrals.n;
  const nOcc = hf.nOccupied;
  const nVirt = n - nOcc;
  const dim = nOcc * nVirt;
  const nRoots = energies.length;

  // ── Build dipole AO matrices (3 × n × n). ──────────────────
  // ⟨χ_μ | r_axis | χ_ν⟩ — symmetric, so build upper triangle.
  const dipAO: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(n * n), new Float64Array(n * n), new Float64Array(n * n),
  ];
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu; nu < n; nu++) {
      for (let axis = 0 as 0 | 1 | 2; axis < 3; axis++) {
        const v = dipole_cg(integrals.shells[mu]!, integrals.shells[nu]!, axis);
        dipAO[axis]![mu * n + nu] = v;
        dipAO[axis]![nu * n + mu] = v;
      }
    }
  }

  // ── MO transform of the (occ × virt) block per axis. ───────
  // dipMO[axis][i·nVirt + a] = ⟨φ_i^MO | r_axis | φ_a^MO⟩.
  const dipMO: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(dim), new Float64Array(dim), new Float64Array(dim),
  ];
  for (let axis = 0; axis < 3; axis++) {
    for (let i = 0; i < nOcc; i++) {
      for (let a = 0; a < nVirt; a++) {
        const aMO = nOcc + a;
        let s = 0;
        for (let mu = 0; mu < n; mu++) {
          const Cmu_i = hf.C_MO[mu * n + i]!;
          if (Cmu_i === 0) continue;
          for (let nu = 0; nu < n; nu++) {
            s += Cmu_i * hf.C_MO[nu * n + aMO]! * dipAO[axis]![mu * n + nu]!;
          }
        }
        dipMO[axis]![i * nVirt + a] = s;
      }
    }
  }

  // ── Per-root: build effective amplitude (X for TDA, S·Z' for TDDFT). ──
  const f = new Float64Array(nRoots);
  const eff = new Float64Array(dim);
  for (let r = 0; r < nRoots; r++) {
    if (S) {
      // eff = S · Z' (matrix-vector with the symmetric S).
      for (let ia = 0; ia < dim; ia++) {
        let s = 0;
        for (let jb = 0; jb < dim; jb++) {
          s += S[ia * dim + jb]! * amplitudes[r * dim + jb]!;
        }
        eff[ia] = s;
      }
    } else {
      // TDA: eff = X amplitudes directly.
      for (let ia = 0; ia < dim; ia++) eff[ia] = amplitudes[r * dim + ia]!;
    }

    let sumSq = 0;
    for (let axis = 0; axis < 3; axis++) {
      let T = 0;
      for (let ia = 0; ia < dim; ia++) T += eff[ia]! * dipMO[axis]![ia]!;
      sumSq += T * T;
    }
    // (4/3) · ω · |T|² for TDA; the ω cancels for TDDFT (Casida norm).
    f[r] = S ? (4 / 3) * sumSq : (4 / 3) * energies[r]! * sumSq;
  }
  return f;
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

/**
 * Apply the Cartesian → spherical-harmonic transform T (row-major
 * `n_sph × n_cart`) to a per-grid AO array. Used for φ, ∇φ, ∇²φ
 * on the grid when the integrals were built with `spherical: true`.
 *   φ_sph[g, p] = Σ_μ T[p, μ]·φ_cart[g, μ]
 */
function applyTransformToGridValues(
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
