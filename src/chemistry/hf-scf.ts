// ─────────────────────────────────────────────────────────────
// hf-scf.ts — closed-shell Hartree-Fock self-consistent field
// (RHF). Phase D: the bridge from minimal-basis FCI to bigger
// basis sets and post-HF methods (MP2, CCSD coming in Phase E).
//
// What HF gives you:
//   • E_HF: the lowest-energy single Slater determinant of the
//     molecule. Includes electron-electron Coulomb + exchange
//     mean-field, but NO correlation beyond Pauli exclusion.
//   • C: MO coefficient matrix (AO → MO transform). The columns
//     are the molecular orbitals, ordered by energy.
//   • ε: orbital energies.
//   • D: AO-basis density matrix (Σ_occupied 2 C_i C_i^T).
//
// Algorithm (Roothaan-Hall + Pulay's DIIS extrapolation):
//   1. Compute S, h, ERI in AO basis (provided).
//   2. X = S^{-1/2} (orthogonalizing transform).
//   3. Initial guess: diagonalize core h → initial C, D.
//   4. Loop:
//      a. F = h + G(D)  (Fock matrix)
//      b. E = ½ tr(D · (h + F)) + Vnn
//      c. e = X^T (FDS − SDF) X  (orthogonal-basis error vector)
//      d. push (F, e) into DIIS history; solve for c that
//         minimizes ‖Σ c_i e_i‖ subject to Σ c_i = 1
//      e. F_extrap = Σ c_i F_i
//      f. F' = X^T F_extrap X, diagonalize → C, ε
//      g. D_new = 2 Σ_i^occ C_i C_i^T
//      h. Convergence: ‖e‖_∞ < tol AND |E_new − E_old| < tol
//
// DIIS gets H₂O cc-pVDZ from ~100 iterations down to ~10. For the
// first 2 iterations the DIIS subspace is too small to extrapolate
// usefully, so we just take the new Fock directly. From iter ≥ 3
// onwards DIIS picks the linear combination of past Fs that
// makes the error commutator [F, DS] vanish in the subspace.
// ─────────────────────────────────────────────────────────────

import { type MolecularIntegrals } from "./cg-molecular.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { choleskyDecomposeERI, buildJK_DF, type DFResult } from "./df.js";
import { buildGParallel } from "../parallel/parallel-buildG.js";
import { buildGWasmParallel } from "../parallel/parallel-buildG-wasm.js";
import { buildGGpu } from "./jk-gpu.js";
import { buildJK_DF_Parallel } from "../parallel/parallel-jk-df.js";

export interface HFResult {
  /** Total HF energy (Hartree) including nuclear repulsion. */
  readonly energy: number;
  /** Electronic energy = energy − Vnn. */
  readonly electronicEnergy: number;
  /** AO → MO coefficient matrix, n × n row-major. Column i = MO i. */
  readonly C_MO: Float64Array;
  /** Orbital energies, length n, ascending. */
  readonly orbitalEnergies: Float64Array;
  /** AO-basis density matrix, n × n row-major. */
  readonly D: Float64Array;
  /** Number of doubly occupied orbitals = nElectrons / 2. */
  readonly nOccupied: number;
  /** Number of SCF iterations performed. */
  readonly iter: number;
  /** True if both energy and density tolerances were met. */
  readonly converged: boolean;
  /** Per-iteration energy trace (for debugging / plotting). */
  readonly history: readonly number[];
}

export interface HFOpts {
  /** Hard cap on SCF iterations. Default 100. */
  readonly maxIter?: number;
  /**
   * Parallelize the JK build across a Web Worker pool. 0 = single-threaded
   * (default). Positive N spawns N workers and partitions the outer μ
   * loop. Requires the page to be cross-origin-isolated (COOP/COEP
   * headers, which webgpu-q ships for WebGPU). Most useful for n ≥ 20
   * basis functions; below that the worker overhead dominates.
   */
  readonly parallel?: number;
  /** Energy tolerance for convergence. Default 1e-8 Ha. */
  readonly energyTol?: number;
  /**
   * Density-matrix Frobenius-norm tolerance OR (DIIS path) error
   * vector ‖e‖_∞ tolerance. Default 1e-6.
   */
  readonly densityTol?: number;
  /**
   * If true (default), use Pulay DIIS extrapolation. If false,
   * use plain Roothaan-Hall with α-damping (legacy path).
   */
  readonly useDIIS?: boolean;
  /**
   * DIIS subspace size. Default 8. Only used when useDIIS is true.
   */
  readonly diisHistory?: number;
  /**
   * Damping factor α ∈ (0, 1] for the legacy non-DIIS path.
   * D_new ← α · D_new + (1 − α) · D_old. Default 0.5.
   */
  readonly damping?: number;
  /**
   * Virtual-orbital level shift (Hartree). When > 0, adds
   * `levelShift · P_v` to the orthogonal-basis Fock matrix each
   * iteration, where P_v = Σ_{p ≥ nOcc} c'_p · c'_p^T is the
   * virtual projector built from the PREVIOUS iteration's
   * orthogonal-basis eigenvectors. This artificially raises
   * virtual orbital energies, widening the HOMO-LUMO gap and
   * stabilizing convergence for stretched bonds and other
   * near-degenerate cases that the bare Roothaan-Hall / DIIS
   * loop fails on.
   *
   * Typical values: 0.1-1.0 Ha. The shift is removed from the
   * returned virtual orbital energies post-convergence, so the
   * reported `orbitalEnergies` are the physical Koopmans values.
   *
   * Composes with DIIS and damping. Default 0 (no shift).
   */
  readonly levelShift?: number;
  /**
   * Use density-fitting (Cholesky-decomposed ERI) for the Fock J + K
   * build. `true` → use default Cholesky threshold 1e-10 (machine
   * precision); number → use as threshold; DFResult → reuse a
   * pre-computed B-tensor. Default false (direct 4-index path).
   *
   * For closed-shell HF the DF Fock contribution is G_DF = J − ½K,
   * where (J, K) come from `buildJK_DF`. At threshold τ ≤ 1e-10 the
   * DF-HF energy matches the direct path to better than 1 µHa on
   * STO-3G systems and machine precision on cc-pVDZ.
   */
  readonly useDF?: boolean | number | DFResult;
  /**
   * Use the WASM-native JK build kernel inside the parallel path.
   * Each worker calls `fock_one_mu_row` (native Rust) for its μ rows
   * instead of running the inner accumulator in TypeScript. Only takes
   * effect when `parallel > 0` and SAB is available. Default true.
   */
  readonly useWasmJK?: boolean;
  /**
   * Use the WebGPU JK kernel via WGSL. Pass a ready `GPUDevice` from
   * `initGPU()`. f32 precision — for benzene-class cc-pVDZ the max
   * relative error in G is ~2e-4, which may prevent SCF from
   * converging tighter than ~1e-6 Ha. Trades precision for ~2.4×
   * speedup on the JK build at n=120 vs the WASM SIMD path.
   *
   * Takes precedence over `useWasmJK` when both are set.
   * No effect when `parallel === 0`.
   */
  readonly useWgpuJK?: GPUDevice;
  /**
   * If set, log per-iter wall-time breakdown (JK build, F assemble,
   * DIIS, F' transform, eigendecomp, density update) via the supplied
   * callback. Useful for finding the SCF bottleneck on a given system
   * — at n=190 most of the iter time is one of {JK_DF, eigsymmetric}
   * and which one wins depends on n and basis. Off by default.
   */
  readonly profileCallback?: (iter: number, ms: Record<string, number>) => void;
}

/**
 * Run closed-shell Hartree-Fock SCF on the given molecular
 * integrals with `nElectrons` electrons (must be even).
 */
export function runRHFSCF(
  integrals: MolecularIntegrals,
  nElectrons: number,
  opts: HFOpts = {},
): HFResult {
  if (nElectrons % 2 !== 0) {
    throw new Error(`runRHFSCF: open-shell systems require UHF (got nElectrons=${nElectrons}, odd)`);
  }
  const nOcc = nElectrons / 2;
  const n = integrals.n;
  if (nOcc > n) {
    throw new Error(`runRHFSCF: ${nElectrons} electrons in ${n} spatial orbitals — too many`);
  }
  const maxIter = opts.maxIter ?? 100;
  const eTol = opts.energyTol ?? 1e-8;
  const dTol = opts.densityTol ?? 1e-6;
  const useDIIS = opts.useDIIS ?? true;
  const diisMaxHistory = opts.diisHistory ?? 8;
  const damping = opts.damping ?? 0.5;
  const levelShift = opts.levelShift ?? 0;

  const { S_AO, h_AO, eri_AO, X, Vnn } = integrals;

  // ── Optional: pre-compute the density-fitting B-tensor. ──
  // useDF=true uses τ=1e-10; useDF=<number> uses that threshold;
  // useDF=<DFResult> reuses a caller-provided B-tensor.
  let dfTensor: DFResult | null = null;
  if (opts.useDF !== undefined && opts.useDF !== false) {
    if (typeof opts.useDF === "object") {
      dfTensor = opts.useDF;
    } else {
      const tau = typeof opts.useDF === "number" ? opts.useDF : 1e-10;
      dfTensor = choleskyDecomposeERI(eri_AO, n, tau);
    }
  }

  // ── Initial guess: diagonalize core h to get starting C ──
  const hPrime = transformSymmetric(h_AO, X, n);
  const initial = solveFock(hPrime, X, n);
  let C_MO = initial.C_MO;
  let eps = initial.eps;
  let cPrimePrev: Float64Array = initial.cPrime;
  let D = densityFromC(C_MO, nOcc, n);
  let E_old = Infinity;
  const history: number[] = [];
  let converged = false;
  let iter = 0;

  // DIIS subspace: parallel arrays of past Fock and error vectors.
  const diisF: Float64Array[] = [];
  const diisE: Float64Array[] = [];

  for (iter = 1; iter <= maxIter; iter++) {
    // ── Build Fock F = h + G(D) ─────────────────────────────
    const G = dfTensor !== null ? buildG_DF(D, dfTensor, n) : buildG(D, eri_AO, n);
    const F = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) F[i] = h_AO[i]! + G[i]!;

    // ── HF energy: E = ½ Σ D · (h + F) + Vnn ───────────────
    let Eelec = 0;
    for (let i = 0; i < n * n; i++) Eelec += 0.5 * D[i]! * (h_AO[i]! + F[i]!);
    const E = Eelec + Vnn;
    history.push(E);

    let F_use: Float64Array = F;
    let errMax = 0;

    if (useDIIS) {
      // ── DIIS error vector: e = X^T (FDS − SDF) X ──────────
      // Transformed to orthogonal basis so the Frobenius norm
      // is the right metric for the residual. Vanishes at SCF
      // convergence.
      const e = buildDIISError(F, D, S_AO, X, n);
      for (let i = 0; i < n * n; i++) {
        const a = Math.abs(e[i]!);
        if (a > errMax) errMax = a;
      }

      // ── Append to history (drop oldest if over capacity) ──
      diisF.push(F);
      diisE.push(e);
      if (diisF.length > diisMaxHistory) {
        diisF.shift();
        diisE.shift();
      }

      // ── DIIS extrapolation when we have enough history ────
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

    // ── F' = X^T F_use X, diagonalize → new C, eps ─────────
    const FPrime = transformSymmetric(F_use, X, n);

    // ── Optional virtual-orbital level shift ──────────────
    // F'_lifted = F' + levelShift · Σ_{p ≥ nOcc} c'_p ⊗ c'_p^T
    // Using the PREVIOUS iteration's c'_v as the virtual subspace.
    // Raises virtual orbital eigenvalues by `levelShift`, stabilizing
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

    // ── New density ────────────────────────────────────────
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
      // Legacy path: blend with previous density.
      for (let i = 0; i < n * n; i++) {
        D[i] = damping * D_new[i]! + (1 - damping) * D[i]!;
      }
    }

    // Convergence: energy AND (DIIS error OR density change)
    const residOk = useDIIS ? errMax < dTol : dNorm < dTol;
    if (Math.abs(E - E_old) < eTol && residOk) {
      converged = true;
      E_old = E;
      break;
    }
    E_old = E;
  }

  // ── Strip level-shift from reported virtual eigenvalues. ──
  // The lifted Fock matrix's virtual eigenvalues are eps_true + shift.
  // At convergence the eigenvectors are correct (the shift only changed
  // eigenvalues, not eigenvectors), so removing the shift restores the
  // physical Koopmans IPs / EAs. Energies don't depend on eps and stay
  // correct as-is (E_HF = ½·tr[D(h+F)] + Vnn).
  if (levelShift > 0) {
    const epsOut = new Float64Array(eps);
    for (let i = nOcc; i < n; i++) epsOut[i]! -= levelShift;
    eps = epsOut;
  }

  return {
    energy: E_old,
    electronicEnergy: E_old - Vnn,
    C_MO,
    orbitalEnergies: eps,
    D,
    nOccupied: nOcc,
    iter,
    converged,
    history,
  };
}

/**
 * Async variant of `runRHFSCF` that uses a Web Worker pool to parallelize
 * the JK build. For `opts.parallel === 0` (default), delegates directly
 * to `runRHFSCF` (zero overhead). For `opts.parallel > 0`, runs the
 * SCF loop with `await buildGParallel(...)`.
 *
 * MUST stay in sync with `runRHFSCF` above — the loop body is identical
 * except for the buildG call. If you change one, change the other.
 *
 * Cross-origin isolation (COOP/COEP headers, which webgpu-q ships) is
 * required for the parallel path. On hosts without isolation, falls
 * back to single-threaded.
 */
export async function runRHFSCFAsync(
  integrals: MolecularIntegrals,
  nElectrons: number,
  opts: HFOpts = {},
): Promise<HFResult> {
  const parallel = opts.parallel ?? 0;
  if (parallel <= 0) {
    return runRHFSCF(integrals, nElectrons, opts);
  }
  // Parallel path: SAB required.
  if (typeof SharedArrayBuffer === "undefined" ||
      (typeof crossOriginIsolated !== "undefined" && !crossOriginIsolated)) {
    return runRHFSCF(integrals, nElectrons, opts);
  }

  if (nElectrons % 2 !== 0) {
    throw new Error(`runRHFSCFAsync: open-shell systems require UHF (got nElectrons=${nElectrons}, odd)`);
  }
  const nOcc = nElectrons / 2;
  const n = integrals.n;
  if (nOcc > n) {
    throw new Error(`runRHFSCFAsync: ${nElectrons} electrons in ${n} spatial orbitals — too many`);
  }
  const maxIter = opts.maxIter ?? 100;
  const eTol = opts.energyTol ?? 1e-8;
  const dTol = opts.densityTol ?? 1e-6;
  const useDIIS = opts.useDIIS ?? true;
  const diisMaxHistory = opts.diisHistory ?? 8;
  const damping = opts.damping ?? 0.5;
  const levelShift = opts.levelShift ?? 0;

  const { S_AO, h_AO, eri_AO, X, Vnn } = integrals;

  let dfTensor: DFResult | null = null;
  if (opts.useDF !== undefined && opts.useDF !== false) {
    if (typeof opts.useDF === "object") dfTensor = opts.useDF;
    else {
      const tau = typeof opts.useDF === "number" ? opts.useDF : 1e-10;
      dfTensor = choleskyDecomposeERI(eri_AO, n, tau);
    }
  }

  const hPrime = transformSymmetric(h_AO, X, n);
  const initial = solveFock(hPrime, X, n);
  let C_MO = initial.C_MO;
  let eps = initial.eps;
  let cPrimePrev: Float64Array = initial.cPrime;
  let D = densityFromC(C_MO, nOcc, n);
  let E_old = Infinity;
  const history: number[] = [];
  let converged = false;
  let iter = 0;

  const diisF: Float64Array[] = [];
  const diisE: Float64Array[] = [];

  for (iter = 1; iter <= maxIter; iter++) {
    const prof = opts.profileCallback;
    const t0 = prof ? performance.now() : 0;
    // Async fork point — only difference from the sync runRHFSCF.
    const useWasmJK = opts.useWasmJK ?? true;
    let G: Float64Array;
    if (dfTensor !== null) {
      // DF path: prefer parallel JK_DF when SAB available + workers allow.
      const { J, K } = await buildJK_DF_Parallel(dfTensor, D, parallel);
      G = new Float64Array(n * n);
      for (let i = 0; i < n * n; i++) G[i] = J[i]! - 0.5 * K[i]!;
    } else if (opts.useWgpuJK) {
      G = await buildGGpu(opts.useWgpuJK, D, eri_AO, n);
    } else if (useWasmJK) {
      G = await buildGWasmParallel(D, eri_AO, n, parallel);
    } else {
      G = await buildGParallel(D, eri_AO, n, parallel);
    }
    const tAfterJK = prof ? performance.now() : 0;
    const F = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) F[i] = h_AO[i]! + G[i]!;

    let Eelec = 0;
    for (let i = 0; i < n * n; i++) Eelec += 0.5 * D[i]! * (h_AO[i]! + F[i]!);
    const E = Eelec + Vnn;
    history.push(E);

    let F_use: Float64Array = F;
    let errMax = 0;
    if (useDIIS) {
      const e = buildDIISError(F, D, S_AO, X, n);
      for (let i = 0; i < n * n; i++) {
        const a = Math.abs(e[i]!); if (a > errMax) errMax = a;
      }
      diisF.push(F); diisE.push(e);
      if (diisF.length > diisMaxHistory) { diisF.shift(); diisE.shift(); }
      if (diisF.length >= 2) {
        const c = solveDIISCoeffs(diisE, n);
        if (c !== null) {
          const F_ext = new Float64Array(n * n);
          for (let k = 0; k < diisF.length; k++) {
            const ck = c[k]!; const Fk = diisF[k]!;
            for (let i = 0; i < n * n; i++) F_ext[i]! += ck * Fk[i]!;
          }
          F_use = F_ext;
        }
      }
    }
    const tAfterDIIS = prof ? performance.now() : 0;

    const FPrime = transformSymmetric(F_use, X, n);
    if (levelShift > 0) {
      for (let p = nOcc; p < n; p++) {
        for (let i = 0; i < n; i++) {
          const cpi = cPrimePrev[p * n + i]!; if (cpi === 0) continue;
          const shifted_cpi = levelShift * cpi;
          for (let j = 0; j < n; j++) {
            FPrime[i * n + j]! += shifted_cpi * cPrimePrev[p * n + j]!;
          }
        }
      }
    }

    const tAfterTransform = prof ? performance.now() : 0;
    const sol = solveFock(FPrime, X, n);
    C_MO = sol.C_MO; eps = sol.eps; cPrimePrev = sol.cPrime;
    const tAfterEig = prof ? performance.now() : 0;

    const D_new = densityFromC(C_MO, nOcc, n);
    let dNorm = 0;
    for (let i = 0; i < n * n; i++) {
      const d = D_new[i]! - D[i]!; dNorm += d * d;
    }
    dNorm = Math.sqrt(dNorm);
    if (useDIIS) D = D_new;
    else for (let i = 0; i < n * n; i++) D[i] = damping * D_new[i]! + (1 - damping) * D[i]!;

    const residOk = useDIIS ? errMax < dTol : dNorm < dTol;
    if (prof) {
      const tEnd = performance.now();
      prof(iter, {
        jk: tAfterJK - t0,
        f_assemble: tAfterDIIS - tAfterJK,
        transform: tAfterTransform - tAfterDIIS,
        eig: tAfterEig - tAfterTransform,
        density: tEnd - tAfterEig,
        total: tEnd - t0,
      });
    }
    if (Math.abs(E - E_old) < eTol && residOk) {
      converged = true; E_old = E; break;
    }
    E_old = E;
  }

  if (levelShift > 0) {
    const epsOut = new Float64Array(eps);
    for (let i = nOcc; i < n; i++) epsOut[i]! -= levelShift;
    eps = epsOut;
  }

  return {
    energy: E_old, electronicEnergy: E_old - Vnn,
    C_MO, orbitalEnergies: eps, D, nOccupied: nOcc,
    iter, converged, history,
  };
}

// ── Helpers ─────────────────────────────────────────────────

/** M' = X^T M X for a symmetric M, n × n. Both X and M row-major. */
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

/**
 * Solve F' C' = ε C' (eigh in orthogonal basis), then back-
 * transform C = X C'. Returns the AO-basis MO coefficients
 * sorted by energy ascending.
 */
function solveFock(FPrime: Float64Array, X: Float64Array, n: number): {
  C_MO: Float64Array;
  eps: Float64Array;
  /** Orthogonal-basis eigenvectors, column-major: cPrime[c * n + k]
   *  is the k-th component of the c-th eigenvector. C_MO = X · cPrime. */
  cPrime: Float64Array;
} {
  const eig = eigsymmetric(FPrime, n);
  const C_MO = new Float64Array(n * n);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += X[r * n + k]! * eig.vectors[c * n + k]!;
      }
      C_MO[r * n + c] = s;
    }
  }
  return { C_MO, eps: eig.values, cPrime: eig.vectors };
}

/** D[μ, ν] = 2 Σ_{i=0}^{nOcc-1} C[μ, i] C[ν, i]. */
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

/**
 * Build the two-electron G matrix:
 *   G[μ, ν] = Σ_{λ, σ} D[λ, σ] · ( (μν|λσ) − ½ (μλ|νσ) )
 */
function buildG(D: Float64Array, eri_AO: Float64Array, n: number): Float64Array {
  const G = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++) {
        for (let si = 0; si < n; si++) {
          const Dls = D[la * n + si]!;
          if (Dls === 0) continue;
          const J = eri_AO[((mu * n + nu) * n + la) * n + si]!;
          const K = eri_AO[((mu * n + la) * n + nu) * n + si]!;
          s += Dls * (J - 0.5 * K);
        }
      }
      G[mu * n + nu] = s;
    }
  }
  return G;
}

/** Density-fitting variant: G = J − ½ K from B-tensor contractions. */
function buildG_DF(D: Float64Array, df: DFResult, n: number): Float64Array {
  const { J, K } = buildJK_DF(df, D);
  const G = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) G[i] = J[i]! - 0.5 * K[i]!;
  return G;
}

/**
 * DIIS error vector in the orthogonal basis:
 *   e = X^T · (FDS − SDF) · X
 *
 * In a converged closed-shell SCF, [F, DS] = 0 (the density
 * commutes with the Fock operator), so this vanishes. The
 * orthogonal-basis transform makes the Frobenius norm a proper
 * metric for "distance from convergence."
 */
function buildDIISError(
  F: Float64Array,
  D: Float64Array,
  S: Float64Array,
  X: Float64Array,
  n: number,
): Float64Array {
  // FD = F · D
  const FD = matmul(F, D, n);
  // FDS = FD · S
  const FDS = matmul(FD, S, n);
  // SD = S · D
  const SD = matmul(S, D, n);
  // SDF = SD · F
  const SDF = matmul(SD, F, n);
  // e_AO = FDS − SDF
  const eAO = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) eAO[i] = FDS[i]! - SDF[i]!;
  // e_OAO = X^T · eAO · X — same shape, nicer metric
  return transformGeneral(eAO, X, n);
}

/** General-matrix M' = X^T M X (M not necessarily symmetric). */
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

/** Plain n × n row-major matmul. */
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

/**
 * Solve the DIIS Lagrange system for extrapolation coefficients.
 *   minimize ‖Σ c_i e_i‖² subject to Σ c_i = 1
 * Augmented system:
 *   | B[i,j]  -1 | |c|     | 0 |
 *   |  -1      0 | |λ|  =  |-1 |
 * where B[i,j] = ⟨e_i | e_j⟩ Frobenius.
 *
 * Returns null if the system is singular (drop oldest entry and
 * try again next iter). Solver: Gaussian elimination with
 * partial pivoting on the (m+1) × (m+1) augmented matrix.
 */
function solveDIISCoeffs(diisE: Float64Array[], n: number): Float64Array | null {
  const m = diisE.length;
  const dim = m + 1;
  // Build B[i,j] = ⟨e_i | e_j⟩
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

  // Gaussian elimination with partial pivoting.
  for (let col = 0; col < dim; col++) {
    // pivot
    let piv = col;
    let pivAbs = Math.abs(A[col * dim + col]!);
    for (let r = col + 1; r < dim; r++) {
      const a = Math.abs(A[r * dim + col]!);
      if (a > pivAbs) { pivAbs = a; piv = r; }
    }
    if (pivAbs < 1e-14) return null;  // singular — caller drops oldest entry
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
