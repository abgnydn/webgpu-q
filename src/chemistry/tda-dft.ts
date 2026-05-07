// ─────────────────────────────────────────────────────────────
// tda-dft.ts — Tamm-Dancoff Approximation TDA-DFT and TDA-HF
// excitation energies (singlet sector). Tier 2 stage 9.
//
// Generalizes runCIS to Kohn-Sham reference + HF mixing + the
// LDA XC kernel:
//
//   A^singlet_{ia,jb} = (ε_a − ε_i) δ_ij δ_ab
//                     + 2·(ia|jb)                      (Hartree/J)
//                     − hfMix·(ij|ab)                  (HF exchange × hfMix)
//                     + 2·(ia|f_xc|jb)                 (XC kernel; LDA only)
//
// `hfMix` follows from the SCF method:
//   • HF                  → hfMix = 1, no XC kernel  (= CIS exactly)
//   • LDA (lda-svwn)      → hfMix = 0,    XC kernel from evalXCKernelLDA
//   • B3LYP-style hybrids → hfMix = 0.20, XC kernel ... NOT YET (GGA)
//
// Scope shipped this commit:
//   ✓ HF singlet path reproduces runCIS to fp.
//   ✓ LDA singlet path uses the closed-shell LDA XC kernel.
//   ✗ GGA / hybrid kernels (BVWN5, BLYP, B3VWN5, B3LYP5) require
//     f_ρρ + f_ργ + f_γγ tensors and basis-Hessian-style
//     integrals. The runner throws a clear "needs GGA TDA"
//     message for those — opt back into FD comparison if you
//     need them today. Follow-up.
//   ✗ Triplet TDA-DFT — closed-shell triplet kernel f_xc^triplet
//     differs from f_xc^singlet (involves the spin-symmetric vs
//     spin-antisymmetric second derivative). Deferred; runCIS
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
  const method: TDAMethod = opts.method ?? "hf";
  const n = integrals.n;
  const nOcc = hf.nOccupied;
  const nVirt = n - nOcc;
  const dim = nOcc * nVirt;
  if (nVirt <= 0) throw new Error(`runTDA: no virtual orbitals (n=${n}, nOcc=${nOcc})`);
  const nRoots = Math.min(opts.nRoots ?? dim, dim);

  const isHF = method === "hf";
  const isLDA = method === "lda-svwn";
  if (!isHF && !isLDA) {
    throw new Error(
      `runTDA: method '${method}' requires the GGA / hybrid XC kernel ` +
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
  // K_xc[ia, jb] = ∫ φ_i^MO(r) φ_a^MO(r) · f_xc(ρ(r)) · φ_j^MO(r) φ_b^MO(r) dr
  let K_xc: Float64Array | null = null;
  if (usedXCKernel) {
    if (!opts.nucleiSymbols) {
      throw new Error("runTDA: DFT method requires opts.nucleiSymbols for grid rebuild.");
    }
    const grid = molecularGrid(integrals.nuclei, opts.nucleiSymbols, opts.grid ?? {});
    const basis = evalBasisOnGrid(integrals.shells, grid);
    const rho = evalDensityOnGrid(hf.D, basis);
    const fxc = evalXCKernelLDA(rho);
    const nGrid = grid.x.length;
    const phi = basis.phi;       // φ_μ^AO at every grid point, [nGrid · n].
    // MO orbital values on the grid: φ_p^MO(r_p_grid) = Σ_μ C_μp · φ_μ^AO(r_p_grid).
    // Store only the (n_occ + n_virt) MOs we need = first n columns; reuse the
    // full transform — n is small enough.
    const phiMO = new Float64Array(nGrid * n);
    for (let g = 0; g < nGrid; g++) {
      const off = g * n;
      for (let p = 0; p < n; p++) {
        let s = 0;
        for (let mu = 0; mu < n; mu++) s += phi[off + mu]! * hf.C_MO[mu * n + p]!;
        phiMO[off + p] = s;
      }
    }
    // Pre-build ψ_ia(g) = φ_i^MO(g) · φ_a^MO(g) for every (i ∈ occ, a ∈ virt, g).
    // Layout: psi_ia[(i·nVirt + a)·nGrid + g].
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
    // Contract: K_xc[ia, jb] = Σ_g (w_g · f_xc_g · ψ_ia_g) · ψ_jb_g.
    // Pre-multiply ψ_ia · w · f_xc once, then dot against ψ_jb.
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

  // ── Build the A matrix. ────────────────────────────────────
  const A = new Float64Array(dim * dim);
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
          const coul = 2 * eri(i, ai, j, bj);
          const exch = hfMix !== 0 ? hfMix * eri(i, j, ai, bj) : 0;
          const kxc  = K_xc ? 2 * K_xc[row * dim + col]! : 0;
          A[row * dim + col] = diag + coul - exch + kxc;
        }
      }
    }
  }

  // ── Diagonalize, extract lowest nRoots. ────────────────────
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
    nOccupied: nOcc,
    nVirtual: nVirt,
    singletEnergies: energies,
    singletAmplitudes: amps,
    hfMix,
    usedXCKernel,
  };
}
