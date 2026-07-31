// ─────────────────────────────────────────────────────────────
// ccsd.ts — closed-shell CCSD (Coupled Cluster Singles Doubles)
// via the spin-orbital formulation. Phase E stage 3: the gold-
// standard single-reference correlation method, captures ≥ 95%
// of correlation for closed-shell molecules near equilibrium.
//
// Reference: Stanton, Gauss, Watts, Bartlett (1991) J. Chem.
// Phys. 94, 4334. Equations as restated in
//   Crawford & Schaefer, Reviews in Comp. Chem. vol 14 (2000),
//   "An Introduction to Coupled Cluster Theory", Table 1.
//
// Spin-orbital convention: spin-orbital P encodes spatial p =
// P >> 1, spin σ_P = P & 1. For closed-shell RHF, alpha (P=2p)
// and beta (P=2p+1) share spatial MOs and orbital energies, and
// the Fock matrix is diagonal: f_PQ = δ_PQ ε_p.
//
// Algorithm:
//   1. Convert MO chemist-notation (pq|rs) to spin-orbital
//      antisymmetric ⟨PQ||RS⟩ tensor (NSO⁴, ~40 MB at NSO=48).
//   2. Initialize T1 = 0, T2 = MP2 amplitudes
//      T2[i,j,a,b] = ⟨ij||ab⟩ / (ε_i + ε_j − ε_a − ε_b).
//   3. Iterate Stanton-Bartlett residual equations until energy
//      converges. Each iteration:
//        a. Build effective amplitudes
//             τ̃_ij^ab = t_ij^ab + ½(t_i^a t_j^b − t_i^b t_j^a)
//             τ_ij^ab  = t_ij^ab +   (t_i^a t_j^b − t_i^b t_j^a)
//        b. Build 1-body intermediates F_ae, F_mi, F_me.
//        c. Build 2-body intermediates W_mnij, W_abef, W_mbej.
//        d. Build T1, T2 residual RHS.
//        e. Update amplitudes by dividing by orbital-energy
//           denominators D_ia = ε_i − ε_a, D_ijab = ε_i + ε_j
//           − ε_a − ε_b.
//   4. E_CCSD_corr = ¼ Σ_ijab ⟨ij||ab⟩ τ_ij^ab
//      (canonical RHF: f_ia = 0, so Σ f_ia t_i^a vanishes).
//
// Cost: dominated by W_abef and the (½) Σ_ef τ_ij^ef W_abef
// contraction in the T2 residual, both O(N_o N_v⁴ + N_o² N_v⁴).
// For STO-3G H₂O (N_o=10, N_v=4): ~25k ops/iter, milliseconds.
// For cc-pVDZ H₂O (N_o=10, N_v=38): ~200M ops/iter, ~5 s/iter
// in plain TS, ~30 iters → minutes.
//
// For 2-electron systems (H₂), CCSD is exact (singles + doubles
// span the full Hilbert space). |E_CCSD − E_FCI| < 1e-7 is the
// strongest correctness check.
// ─────────────────────────────────────────────────────────────

import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import { transformERIToMO } from "./mp2.js";

export interface CCSDResult {
  /** CCSD correlation energy (Hartree). Negative for closed-shell near eq. */
  readonly correlationEnergy: number;
  /** Total CCSD energy = E_HF + E_CCSD_corr. */
  readonly totalEnergy: number;
  /** Singles amplitudes [i*nVirt + a], spin-orbital. */
  readonly T1: Float64Array;
  /** Doubles amplitudes [((i*nOcc + j)*nVirt + a)*nVirt + b]. */
  readonly T2: Float64Array;
  /** Per-iteration correlation energy trace. */
  readonly history: readonly number[];
  /** Number of iterations performed. */
  readonly iter: number;
  /** True if BOTH the energy change and the amplitude update ‖ΔT‖ fell below
   *  tol. A dE-only criterion can report convergence on an energy plateau
   *  while the amplitudes are still moving; requiring both closes that hole. */
  readonly converged: boolean;
  /** RMS amplitude update ‖ΔT‖ at the final iteration (residual proxy). */
  readonly residualNorm: number;
  /** T1 diagnostic (Lee, Taylor 1989): ‖T1‖_F / √(N_corr), where N_corr is the
   *  number of CORRELATED occupied spin-orbitals (NOCC − frozen). This docstring
   *  previously said √(2·n_electrons), which disagreed with both the
   *  implementation and its own docstring 80 lines below by a factor of √2.
   *  Heuristic flag for multi-reference character — CCSD becomes
   *  unreliable when t1Diagnostic > 0.02 (closed-shell) or > 0.05
   *  (open-shell). Values < 0.01 indicate confidently single-reference. */
  readonly t1Diagnostic: number;
  /** D1 diagnostic (Janssen, Nielsen 1998): largest singular value
   *  of T1 reshaped as an (n_occ × n_virt) matrix. More sensitive
   *  than the Lee T1 diagnostic to isolated bad orbitals — picks
   *  up cases where total ‖T1‖_F is OK but a single (i, a) amplitude
   *  is large. Multi-reference threshold ≈ 0.05; consult Janssen &
   *  Nielsen 1998 for sector-specific bars. */
  readonly d1Diagnostic: number;
}

export interface CCSDOpts {
  readonly maxIter?: number;
  readonly tol?: number;
  /** Damping for amplitude update. 1.0 = none (default). 0.5 = half-step. */
  readonly damping?: number;
  /** Pulay DIIS acceleration on the amplitude vector (default true). Plain
   *  energy-denominator iteration can enter an amplitude limit cycle — the
   *  energy plateaus (ΔE→0) while ‖ΔT‖ stalls well above tol and never
   *  settles (observed on frozen-core CCSD). DIIS extrapolates that soft mode
   *  out and drives the true residual to tol. Set false for the raw iteration. */
  readonly useDIIS?: boolean;
  /** Max DIIS subspace size (default 6). */
  readonly diisDim?: number;
  /**
   * Number of frozen-core spatial orbitals. The lowest `nFrozenCore`
   * occupied MOs are excluded from the correlation treatment by
   * zeroing T1[i,*] and T2[i,*,*,*] in the core block (and the
   * corresponding j blocks) at every iteration. Default 0.
   */
  readonly nFrozenCore?: number;
}

export function runCCSD(
  hf: HFResult,
  integrals: MolecularIntegrals,
  opts: CCSDOpts = {},
): CCSDResult {
  const n = integrals.n;
  const NSO = 2 * n;
  const NOCC = 2 * hf.nOccupied;
  const NVIRT = NSO - NOCC;
  if (NOCC === 0 || NVIRT === 0) {
    throw new Error(`runCCSD: trivial system (NOCC=${NOCC}, NVIRT=${NVIRT})`);
  }
  const nFrozenSO = 2 * (opts.nFrozenCore ?? 0);
  if (nFrozenSO < 0 || nFrozenSO >= NOCC) {
    throw new Error(
      `runCCSD: nFrozenCore=${opts.nFrozenCore} leaves no active occupied orbitals ` +
      `(NOCC=${NOCC} spin-orbitals)`,
    );
  }

  const eri_MO = transformERIToMO(integrals.eri_AO, hf.C_MO, n);
  const eri = buildSpinOrbitalERI(eri_MO, n);

  // Spin-orbital orbital energies (each spatial → 2 spin orbitals).
  const eps = new Float64Array(NSO);
  for (let P = 0; P < NSO; P++) eps[P] = hf.orbitalEnergies[P >> 1]!;

  const core = ccsdIterate(eri, eps, NOCC, NVIRT, NSO, nFrozenSO, opts);
  const diag = ccsdDiagnostics(core.T1, NOCC, NVIRT, nFrozenSO);
  return {
    correlationEnergy: core.correlationEnergy,
    totalEnergy: hf.energy + core.correlationEnergy,
    T1: core.T1,
    T2: core.T2,
    history: core.history,
    iter: core.iter,
    converged: core.converged,
    residualNorm: core.residualNorm,
    t1Diagnostic: diag.t1,
    d1Diagnostic: diag.d1,
  };
}

/** Compute T1-based wavefunction-quality diagnostics for a CCSD
 *  amplitude tensor. T1 here is the spin-orbital amplitudes
 *  T1[i * NVIRT + a] with i ∈ [0, NOCC), a ∈ [0, NVIRT).
 *
 *  - t1 (Lee, Taylor 1989): ‖T1‖_F / √(N_corr), with N_corr the number of
 *    CORRELATED occupied spin-orbitals. Under a frozen core that is
 *    NOCC − nFrozenSO, not NOCC: frozen amplitudes are held at zero, so
 *    dividing by the full NOCC dilutes the diagnostic and under-reports
 *    multi-reference character exactly when a core is frozen.
 *  - d1 (Janssen, Nielsen 1998): largest singular value of T1
 *    reshaped as the (n_occ × n_virt) matrix.
 *
 *  Both are heuristic diagnostics — flags, not error bars. The
 *  textbook bars (Lee-Taylor): t1 < 0.02 ≈ confidently single-ref;
 *  0.02 < t1 < 0.04 ≈ borderline; t1 > 0.05 ≈ trust at your own risk.
 *  D1 is roughly 2-3× more sensitive than the Lee diagnostic. */
export function ccsdDiagnostics(
  T1: Float64Array, NOCC: number, NVIRT: number, nFrozenSO = 0,
): { t1: number; d1: number } {
  // T1 Frobenius norm.
  let norm2 = 0;
  for (let k = 0; k < T1.length; k++) norm2 += T1[k]! * T1[k]!;
  const nCorrelated = Math.max(1, NOCC - nFrozenSO);
  const t1 = Math.sqrt(norm2) / Math.sqrt(nCorrelated);

  // D1: largest singular value of T1 reshaped as (NOCC × NVIRT).
  // For our small dimensions, computing σ_max via power iteration
  // on T1^T · T1 (NVIRT × NVIRT) is cheap and avoids a full SVD.
  // σ_max(T1) = √(largest eigenvalue of T1^T · T1).
  const TtT = new Float64Array(NVIRT * NVIRT);
  for (let a = 0; a < NVIRT; a++) {
    for (let b = 0; b < NVIRT; b++) {
      let s = 0;
      for (let i = 0; i < NOCC; i++) s += T1[i * NVIRT + a]! * T1[i * NVIRT + b]!;
      TtT[a * NVIRT + b] = s;
    }
  }
  // Power iteration: 50 iters is plenty for a small matrix.
  let v = new Float64Array(NVIRT);
  for (let k = 0; k < NVIRT; k++) v[k] = 1 / Math.sqrt(NVIRT);
  let lambda = 0;
  for (let iter = 0; iter < 50; iter++) {
    const Av = new Float64Array(NVIRT);
    for (let a = 0; a < NVIRT; a++) {
      let s = 0;
      for (let b = 0; b < NVIRT; b++) s += TtT[a * NVIRT + b]! * v[b]!;
      Av[a] = s;
    }
    let n2 = 0;
    for (let k = 0; k < NVIRT; k++) n2 += Av[k]! * Av[k]!;
    const n = Math.sqrt(n2);
    if (n < 1e-30) { lambda = 0; break; }
    for (let k = 0; k < NVIRT; k++) Av[k]! /= n;
    // Rayleigh quotient on the normalized vector.
    let newLam = 0;
    for (let a = 0; a < NVIRT; a++) {
      let s = 0;
      for (let b = 0; b < NVIRT; b++) s += TtT[a * NVIRT + b]! * Av[b]!;
      newLam += Av[a]! * s;
    }
    if (Math.abs(newLam - lambda) < 1e-12) { lambda = newLam; break; }
    lambda = newLam;
    v = Av;
  }
  const d1 = Math.sqrt(Math.max(0, lambda));
  return { t1, d1 };
}

/** Solve the Pulay DIIS coefficient system for a set of error vectors.
 *  Minimises ‖Σ cᵢ eᵢ‖ subject to Σ cᵢ = 1 via the bordered normal
 *  equations [[B, −1], [−1ᵀ, 0]] [c; λ] = [0; −1], B_ij = ⟨eᵢ, eⱼ⟩.
 *  Returns the n coefficients, or null if the system is singular (caller
 *  then skips extrapolation for this step). */
function solveDIIS(errs: readonly Float64Array[]): number[] | null {
  const n = errs.length;
  const m = n + 1;
  const A: number[][] = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  for (let i = 0; i < n; i++) {
    const ei = errs[i]!;
    for (let j = i; j < n; j++) {
      const ej = errs[j]!;
      let s = 0;
      for (let k = 0; k < ei.length; k++) s += ei[k]! * ej[k]!;
      A[i]![j] = s;
      A[j]![i] = s;
    }
  }
  for (let i = 0; i < n; i++) { A[i]![n] = -1; A[n]![i] = -1; }
  A[n]![n] = 0;
  const rhs = new Array<number>(m).fill(0);
  rhs[n] = -1;
  const sol = gaussSolve(A, rhs);
  if (!sol) return null;
  return sol.slice(0, n);
}

/** Dense Gaussian elimination with partial pivoting for a small square
 *  system A x = b. Returns null if A is singular. A and b are consumed. */
function gaussSolve(A: number[][], b: number[]): number[] | null {
  const m = b.length;
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) if (Math.abs(A[r]![col]!) > Math.abs(A[piv]![col]!)) piv = r;
    if (Math.abs(A[piv]![col]!) < 1e-14) return null;
    if (piv !== col) { [A[piv], A[col]] = [A[col]!, A[piv]!]; [b[piv], b[col]] = [b[col]!, b[piv]!]; }
    const pivRow = A[col]!;
    const pivVal = pivRow[col]!;
    for (let r = 0; r < m; r++) {
      if (r === col) continue;
      const factor = A[r]![col]! / pivVal;
      if (factor === 0) continue;
      const row = A[r]!;
      for (let c = col; c < m; c++) row[c] = row[c]! - factor * pivRow[c]!;
      b[r] = b[r]! - factor * b[col]!;
    }
  }
  const x = new Array<number>(m);
  for (let i = 0; i < m; i++) x[i] = b[i]! / A[i]![i]!;
  return x;
}

/**
 * Shared CCSD iteration core. Takes a precomputed spin-orbital
 * antisymmetric ERI tensor, spin-resolved orbital energies, and
 * occupied/virtual counts. Runs the Stanton-Bartlett residual loop.
 *
 * Used by both closed-shell `runCCSD` (RHF input) and open-shell
 * `runUCCSD` (UHF input). For canonical RHF and canonical UHF the
 * spin-orbital Fock matrix is diagonal (f_PQ = δ_PQ ε_P^σ), so the
 * existing residual machinery applies in both cases — UHF just
 * supplies different ε for α and β.
 */
export function ccsdIterate(
  eri: Float64Array,
  eps: Float64Array,
  NOCC: number,
  NVIRT: number,
  NSO: number,
  /** Frozen occupied SO indices. Accepts either:
   *   - `number`: legacy contiguous-frozen — freezes SOs [0, nFrozen)
   *     in the interleaved RHF SO ordering (P = 2p + σ).
   *   - `ReadonlySet<number>`: explicit set — required for UCCSD,
   *     where the SO ordering is "all-α-occ first, then all-β-occ"
   *     and freezing α+β of the lowest k spatials is NOT a
   *     contiguous SO range. Pass `new Set([α-SOs of frozen
   *     spatials, β-SOs of frozen spatials])`.
   */
  frozenOccSO: number | ReadonlySet<number>,
  opts: CCSDOpts,
): Omit<CCSDResult, "totalEnergy" | "t1Diagnostic" | "d1Diagnostic"> {
  // Normalize to a Set for downstream uniform handling.
  const frozenSet: ReadonlySet<number> = typeof frozenOccSO === "number"
    ? new Set(Array.from({ length: frozenOccSO }, (_, i) => i))
    : frozenOccSO;
  // Energy denominators.
  const D_ia = new Float64Array(NOCC * NVIRT);
  for (let i = 0; i < NOCC; i++) {
    for (let a = 0; a < NVIRT; a++) {
      D_ia[i * NVIRT + a] = eps[i]! - eps[a + NOCC]!;
    }
  }
  const D_ijab = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          D_ijab[((i * NOCC + j) * NVIRT + a) * NVIRT + b] =
            eps[i]! + eps[j]! - eps[a + NOCC]! - eps[b + NOCC]!;
        }
      }
    }
  }

  // Initial T1 = 0, T2 = MP2 amplitudes.
  const T1 = new Float64Array(NOCC * NVIRT);
  const T2 = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          const idx = ((i * NOCC + j) * NVIRT + a) * NVIRT + b;
          const v = oovv(eri, NSO, NOCC, i, j, a, b);
          T2[idx] = v / D_ijab[idx]!;
        }
      }
    }
  }
  // Frozen-core: zero amplitudes whose occupied index falls in the core
  // block. The CC residual machinery preserves zeros when the input
  // amplitudes are zero (every term either pairs with or excites out
  // of the core), so re-zeroing each iter keeps the core inactive.
  zeroCoreAmplitudes(T1, T2, frozenSet, NOCC, NVIRT);

  const maxIter = opts.maxIter ?? 100;
  const tol = opts.tol ?? 1e-9;
  const damping = opts.damping ?? 1.0;
  const useDIIS = opts.useDIIS ?? true;
  const diisDim = opts.diisDim ?? 6;
  // DIIS subspace: parameter vectors (post-Jacobi amplitudes) and their error
  // vectors (the Jacobi update ΔT). Concatenation order is [T1 | T2].
  const nAmpTotal = NOCC * NVIRT + NOCC * NOCC * NVIRT * NVIRT;
  const diisP: Float64Array[] = [];
  const diisE: Float64Array[] = [];

  let E_corr = computeECorr(T1, T2, eri, NOCC, NVIRT, NSO);
  const history: number[] = [E_corr];
  let converged = false;
  let iter = 0;
  let E_old = E_corr;
  // RMS of the amplitude update ‖ΔT‖ — a residual proxy. A dE-only stop can
  // report `converged` on an energy plateau while the amplitudes are still
  // moving (the energy is stationary to first order in the amplitude error, so
  // |ΔE| can undershoot ‖ΔT‖ near the fixed point). We require BOTH to fall
  // below tol before declaring convergence, and surface the final value.
  let residualNorm = Infinity;

  for (iter = 1; iter <= maxIter; iter++) {
    const tau_t = makeTau(T1, T2, NOCC, NVIRT, 0.5);
    const tau   = makeTau(T1, T2, NOCC, NVIRT, 1.0);

    const F_ae = makeF_ae(T1, eps, eri, tau_t, NOCC, NVIRT, NSO);
    const F_mi = makeF_mi(T1, eps, eri, tau_t, NOCC, NVIRT, NSO);
    const F_me = makeF_me(T1, eri, NOCC, NVIRT, NSO);

    const W_mnij = makeW_mnij(T1, tau, eri, NOCC, NVIRT, NSO);
    const W_abef = makeW_abef(T1, tau, eri, NOCC, NVIRT, NSO);
    const W_mbej = makeW_mbej(T1, T2, eri, NOCC, NVIRT, NSO);

    const r1 = makeRHS_T1(T1, T2, F_ae, F_mi, F_me, eri, NOCC, NVIRT, NSO);
    const r2 = makeRHS_T2(T1, T2, tau, F_ae, F_mi, F_me,
                          W_mnij, W_abef, W_mbej, eri, NOCC, NVIRT, NSO);

    // Update amplitudes via energy denominators, capturing the Jacobi update
    // ΔT as the DIIS error vector [T1 | T2] and accumulating ‖ΔT‖² so we can
    // measure how far the amplitudes still moved this step (the true residual).
    const off2 = NOCC * NVIRT;
    const errVec = new Float64Array(nAmpTotal);
    let sumSqDelta = 0;
    let nActive = 0;
    // Frozen (core) amplitudes are constrained to zero by zeroCoreAmplitudes,
    // so the Jacobi step still produces a nonzero r/D push for them that we
    // immediately discard. Counting that push as "residual" measures progress
    // along a projected-out direction that can never converge — it pins the
    // frozen-core residual at ~1e-4 forever. Exclude those constrained
    // directions from BOTH the residual norm and the DIIS error vector; only
    // the free amplitudes define convergence.
    for (let i = 0; i < NOCC; i++) {
      const frozenI = frozenSet.has(i);
      for (let a = 0; a < NVIRT; a++) {
        const idx = i * NVIRT + a;
        const tnew = r1[idx]! / D_ia[idx]!;
        const updated = damping * tnew + (1 - damping) * T1[idx]!;
        const delta = updated - T1[idx]!;
        T1[idx] = updated;
        if (frozenI) continue;
        errVec[idx] = delta;
        sumSqDelta += delta * delta;
        nActive++;
      }
    }
    for (let i = 0; i < NOCC; i++) {
      const frozenI = frozenSet.has(i);
      for (let j = 0; j < NOCC; j++) {
        const frozenIJ = frozenI || frozenSet.has(j);
        for (let a = 0; a < NVIRT; a++) {
          for (let b = 0; b < NVIRT; b++) {
            const idx = ((i * NOCC + j) * NVIRT + a) * NVIRT + b;
            const tnew = r2[idx]! / D_ijab[idx]!;
            const updated = damping * tnew + (1 - damping) * T2[idx]!;
            const delta = updated - T2[idx]!;
            T2[idx] = updated;
            if (frozenIJ) continue;
            errVec[off2 + idx] = delta;
            sumSqDelta += delta * delta;
            nActive++;
          }
        }
      }
    }
    zeroCoreAmplitudes(T1, T2, frozenSet, NOCC, NVIRT);
    // ‖ΔT‖ of the raw Jacobi step (pre-extrapolation) over the FREE amplitudes
    // — the honest measure of how far we still are from the fixed point.
    residualNorm = nActive > 0 ? Math.sqrt(sumSqDelta / nActive) : 0;

    // Pulay DIIS: extrapolate the amplitudes from the stored subspace so the
    // soft/limit-cycle mode that stalls the raw iteration is projected out.
    if (useDIIS) {
      const parVec = new Float64Array(nAmpTotal);
      parVec.set(T1, 0);
      parVec.set(T2, off2);
      diisP.push(parVec);
      diisE.push(errVec);
      if (diisP.length > diisDim) { diisP.shift(); diisE.shift(); }
      if (diisP.length >= 2) {
        const c = solveDIIS(diisE);
        if (c) {
          const ext = new Float64Array(nAmpTotal);
          for (let k = 0; k < c.length; k++) {
            const ck = c[k]!;
            const pk = diisP[k]!;
            for (let t = 0; t < nAmpTotal; t++) ext[t]! += ck * pk[t]!;
          }
          T1.set(ext.subarray(0, off2));
          T2.set(ext.subarray(off2));
          zeroCoreAmplitudes(T1, T2, frozenSet, NOCC, NVIRT);
        }
      }
    }

    E_corr = computeECorr(T1, T2, eri, NOCC, NVIRT, NSO);
    history.push(E_corr);

    // Converge only when the energy AND the amplitudes have both settled — a
    // dE-only test can stop early on a plateau while ‖ΔT‖ is still sizeable.
    if (Math.abs(E_corr - E_old) < tol && residualNorm < tol) {
      converged = true;
      break;
    }
    E_old = E_corr;
  }

  return {
    correlationEnergy: E_corr,
    T1, T2, history, iter, converged, residualNorm,
  };
}

// ─────────────────────────────────────────────────────────────
// Spin-orbital antisymmetric ERI tensor builder.
//
// Convention: chemist's notation in the spatial MO ERI is
// (pq|rs) = ∫∫ φ_p(1) φ_q(1) (1/r12) φ_r(2) φ_s(2).
// Physicist's: ⟨pq|rs⟩ = (pr|qs).  Antisymm: ⟨pq||rs⟩ = ⟨pq|rs⟩
// − ⟨pq|sr⟩ = (pr|qs) − (ps|qr).
//
// Spin block: ⟨PQ|RS⟩_phys = ⟨pq|rs⟩_phys δ(σ_P,σ_R) δ(σ_Q,σ_S).
// ─────────────────────────────────────────────────────────────
export function buildSpinOrbitalERI(eri_MO_chem: Float64Array, n: number): Float64Array {
  const NSO = 2 * n;
  const out = new Float64Array(NSO * NSO * NSO * NSO);
  for (let P = 0; P < NSO; P++) {
    const p = P >> 1, sp = P & 1;
    for (let Q = 0; Q < NSO; Q++) {
      const q = Q >> 1, sq = Q & 1;
      for (let R = 0; R < NSO; R++) {
        const r = R >> 1, sr = R & 1;
        for (let S = 0; S < NSO; S++) {
          const s = S >> 1, ss = S & 1;
          const direct = (sp === sr && sq === ss)
            ? eri_MO_chem[((p * n + r) * n + q) * n + s]! : 0;
          const exch = (sp === ss && sq === sr)
            ? eri_MO_chem[((p * n + s) * n + q) * n + r]! : 0;
          out[((P * NSO + Q) * NSO + R) * NSO + S] = direct - exch;
        }
      }
    }
  }
  return out;
}

// Convenience accessor for ⟨ij||ab⟩ where i,j are occ-block indices,
// a,b are virt-block indices. Spin-orb idx = i for occ, a + NOCC for virt.
function oovv(eri: Float64Array, NSO: number, NOCC: number,
              i: number, j: number, a: number, b: number): number {
  return eri[((i * NSO + j) * NSO + (a + NOCC)) * NSO + (b + NOCC)]!;
}

/**
 * Zero out amplitudes whose occupied spin-orbital index falls in the
 * frozen-core block [0, nFrozenSO). The CC residual equations are
 * homogeneous in T1, T2 across each occupied row/column — if every
 * core-row amplitude is zero on entry, every core-row residual is
 * also zero on exit. So re-zeroing each iter is equivalent to
 * solving the CC equations with the constraint T_core = 0.
 */
function zeroCoreAmplitudes(
  T1: Float64Array, T2: Float64Array,
  frozenSet: ReadonlySet<number>, NOCC: number, NVIRT: number,
): void {
  if (frozenSet.size === 0) return;
  for (const i of frozenSet) {
    for (let a = 0; a < NVIRT; a++) T1[i * NVIRT + a] = 0;
  }
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      if (!frozenSet.has(i) && !frozenSet.has(j)) continue;
      const base = ((i * NOCC + j) * NVIRT) * NVIRT;
      for (let k = 0; k < NVIRT * NVIRT; k++) T2[base + k] = 0;
    }
  }
}

// ─────────────────────────────────────────────────────────────
// Effective amplitudes:
//   τ̃ (alpha=½): t_ij^ab + ½ (t_i^a t_j^b − t_i^b t_j^a)   used in F intermediates
//   τ  (alpha=1): t_ij^ab + (t_i^a t_j^b − t_i^b t_j^a)     used in W intermediates + energy
// ─────────────────────────────────────────────────────────────
export function makeTau(T1: Float64Array, T2: Float64Array,
                 NOCC: number, NVIRT: number, alpha: number): Float64Array {
  const out = new Float64Array(T2.length);
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          const idx = ((i * NOCC + j) * NVIRT + a) * NVIRT + b;
          const tia = T1[i * NVIRT + a]!;
          const tjb = T1[j * NVIRT + b]!;
          const tib = T1[i * NVIRT + b]!;
          const tja = T1[j * NVIRT + a]!;
          out[idx] = T2[idx]! + alpha * (tia * tjb - tib * tja);
        }
      }
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────
// 1-body intermediates (Stanton-Bartlett, canonical-RHF form).
// In canonical RHF, f_PQ is diagonal so (1−δ) f_off = 0 and the
// f_me, f_ia blocks vanish identically. We keep the f_me terms
// explicit (would matter for non-canonical bases) but evaluate
// them using the diagonal-f shortcut: f_PQ = δ_PQ ε_p.
//
// F_ae = − ½ Σ_m f_me t_m^a + Σ_mf t_m^f ⟨ma||fe⟩
//        − ½ Σ_mnf τ̃_mn^af ⟨mn||ef⟩
// F_mi = + ½ Σ_e t_i^e f_me + Σ_en t_n^e ⟨mn||ie⟩
//        + ½ Σ_nef τ̃_in^ef ⟨mn||ef⟩
// F_me = f_me + Σ_nf t_n^f ⟨mn||ef⟩
//
// For canonical RHF, f_me = δ_me ε_e but m is occupied and e
// virtual (different blocks) → f_me ≡ 0. We just drop those
// terms here.
// ─────────────────────────────────────────────────────────────

export function makeF_ae(T1: Float64Array, eps: Float64Array, eri: Float64Array,
                  tau_t: Float64Array, NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const F = new Float64Array(NVIRT * NVIRT);
  for (let a = 0; a < NVIRT; a++) {
    const A = a + NOCC;
    for (let e = 0; e < NVIRT; e++) {
      const E = e + NOCC;
      let s = 0;
      // Σ_mf t_m^f ⟨ma||fe⟩
      for (let m = 0; m < NOCC; m++) {
        for (let f = 0; f < NVIRT; f++) {
          const F_v = f + NOCC;
          s += T1[m * NVIRT + f]! * eri[((m * NSO + A) * NSO + F_v) * NSO + E]!;
        }
      }
      // − ½ Σ_mnf τ̃_mn^af ⟨mn||ef⟩
      for (let m = 0; m < NOCC; m++) {
        for (let nIdx = 0; nIdx < NOCC; nIdx++) {
          for (let f = 0; f < NVIRT; f++) {
            const F_v = f + NOCC;
            s -= 0.5 * tau_t[((m * NOCC + nIdx) * NVIRT + a) * NVIRT + f]! *
                       eri[((m * NSO + nIdx) * NSO + E) * NSO + F_v]!;
          }
        }
      }
      // Note: − ½ Σ_m f_me t_m^a — vanishes in canonical RHF (f_me = 0).
      F[a * NVIRT + e] = s;
    }
  }
  void eps;
  return F;
}

export function makeF_mi(T1: Float64Array, eps: Float64Array, eri: Float64Array,
                  tau_t: Float64Array, NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const F = new Float64Array(NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let i = 0; i < NOCC; i++) {
      let s = 0;
      // Σ_en t_n^e ⟨mn||ie⟩
      for (let e = 0; e < NVIRT; e++) {
        const E = e + NOCC;
        for (let nIdx = 0; nIdx < NOCC; nIdx++) {
          s += T1[nIdx * NVIRT + e]! * eri[((m * NSO + nIdx) * NSO + i) * NSO + E]!;
        }
      }
      // + ½ Σ_nef τ̃_in^ef ⟨mn||ef⟩
      for (let nIdx = 0; nIdx < NOCC; nIdx++) {
        for (let e = 0; e < NVIRT; e++) {
          const E = e + NOCC;
          for (let f = 0; f < NVIRT; f++) {
            const F_v = f + NOCC;
            s += 0.5 * tau_t[((i * NOCC + nIdx) * NVIRT + e) * NVIRT + f]! *
                       eri[((m * NSO + nIdx) * NSO + E) * NSO + F_v]!;
          }
        }
      }
      // Note: + ½ Σ_e t_i^e f_me — vanishes in canonical RHF.
      F[m * NOCC + i] = s;
    }
  }
  void eps;
  return F;
}

export function makeF_me(T1: Float64Array, eri: Float64Array,
                  NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const F = new Float64Array(NOCC * NVIRT);
  // Canonical RHF: f_me = 0. Just Σ_nf t_n^f ⟨mn||ef⟩.
  for (let m = 0; m < NOCC; m++) {
    for (let e = 0; e < NVIRT; e++) {
      const E = e + NOCC;
      let s = 0;
      for (let nIdx = 0; nIdx < NOCC; nIdx++) {
        for (let f = 0; f < NVIRT; f++) {
          const F_v = f + NOCC;
          s += T1[nIdx * NVIRT + f]! * eri[((m * NSO + nIdx) * NSO + E) * NSO + F_v]!;
        }
      }
      F[m * NVIRT + e] = s;
    }
  }
  return F;
}

// ─────────────────────────────────────────────────────────────
// 2-body intermediates.
//
// W_mnij = ⟨mn||ij⟩ + P(ij) Σ_e t_j^e ⟨mn||ie⟩ + ¼ Σ_ef τ_ij^ef ⟨mn||ef⟩
// W_abef = ⟨ab||ef⟩ − P(ab) Σ_m t_m^b ⟨am||ef⟩ + ¼ Σ_mn τ_mn^ab ⟨mn||ef⟩
// W_mbej = ⟨mb||ej⟩ + Σ_f t_j^f ⟨mb||ef⟩ − Σ_n t_n^b ⟨mn||ej⟩
//          − Σ_nf (½ t_jn^fb + t_j^f t_n^b) ⟨mn||ef⟩
//
// P(ij) X[i,j] = X[i,j] − X[j,i].
// ─────────────────────────────────────────────────────────────

export function makeW_mnij(T1: Float64Array, tau: Float64Array, eri: Float64Array,
                    NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const W = new Float64Array(NOCC * NOCC * NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let nIdx = 0; nIdx < NOCC; nIdx++) {
      for (let i = 0; i < NOCC; i++) {
        for (let j = 0; j < NOCC; j++) {
          let s = eri[((m * NSO + nIdx) * NSO + i) * NSO + j]!;  // ⟨mn||ij⟩
          // P(ij) Σ_e t_j^e ⟨mn||ie⟩  =  Σ_e t_j^e ⟨mn||ie⟩ − Σ_e t_i^e ⟨mn||je⟩
          for (let e = 0; e < NVIRT; e++) {
            const E = e + NOCC;
            s += T1[j * NVIRT + e]! * eri[((m * NSO + nIdx) * NSO + i) * NSO + E]!;
            s -= T1[i * NVIRT + e]! * eri[((m * NSO + nIdx) * NSO + j) * NSO + E]!;
          }
          // ¼ Σ_ef τ_ij^ef ⟨mn||ef⟩
          let q = 0;
          for (let e = 0; e < NVIRT; e++) {
            const E = e + NOCC;
            for (let f = 0; f < NVIRT; f++) {
              const F_v = f + NOCC;
              q += tau[((i * NOCC + j) * NVIRT + e) * NVIRT + f]! *
                   eri[((m * NSO + nIdx) * NSO + E) * NSO + F_v]!;
            }
          }
          s += 0.25 * q;
          W[((m * NOCC + nIdx) * NOCC + i) * NOCC + j] = s;
        }
      }
    }
  }
  return W;
}

export function makeW_abef(T1: Float64Array, tau: Float64Array, eri: Float64Array,
                    NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const W = new Float64Array(NVIRT * NVIRT * NVIRT * NVIRT);
  for (let a = 0; a < NVIRT; a++) {
    const A = a + NOCC;
    for (let b = 0; b < NVIRT; b++) {
      const B = b + NOCC;
      for (let e = 0; e < NVIRT; e++) {
        const E = e + NOCC;
        for (let f = 0; f < NVIRT; f++) {
          const F_v = f + NOCC;
          let s = eri[((A * NSO + B) * NSO + E) * NSO + F_v]!;  // ⟨ab||ef⟩
          // − P(ab) Σ_m t_m^b ⟨am||ef⟩
          for (let m = 0; m < NOCC; m++) {
            s -= T1[m * NVIRT + b]! * eri[((A * NSO + m) * NSO + E) * NSO + F_v]!;
            s += T1[m * NVIRT + a]! * eri[((B * NSO + m) * NSO + E) * NSO + F_v]!;
          }
          // ¼ Σ_mn τ_mn^ab ⟨mn||ef⟩
          let q = 0;
          for (let m = 0; m < NOCC; m++) {
            for (let nIdx = 0; nIdx < NOCC; nIdx++) {
              q += tau[((m * NOCC + nIdx) * NVIRT + a) * NVIRT + b]! *
                   eri[((m * NSO + nIdx) * NSO + E) * NSO + F_v]!;
            }
          }
          s += 0.25 * q;
          W[((a * NVIRT + b) * NVIRT + e) * NVIRT + f] = s;
        }
      }
    }
  }
  return W;
}

export function makeW_mbej(T1: Float64Array, T2: Float64Array, eri: Float64Array,
                    NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const W = new Float64Array(NOCC * NVIRT * NVIRT * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let b = 0; b < NVIRT; b++) {
      const B = b + NOCC;
      for (let e = 0; e < NVIRT; e++) {
        const E = e + NOCC;
        for (let j = 0; j < NOCC; j++) {
          let s = eri[((m * NSO + B) * NSO + E) * NSO + j]!;  // ⟨mb||ej⟩
          // + Σ_f t_j^f ⟨mb||ef⟩
          for (let f = 0; f < NVIRT; f++) {
            const F_v = f + NOCC;
            s += T1[j * NVIRT + f]! * eri[((m * NSO + B) * NSO + E) * NSO + F_v]!;
          }
          // − Σ_n t_n^b ⟨mn||ej⟩
          for (let nIdx = 0; nIdx < NOCC; nIdx++) {
            s -= T1[nIdx * NVIRT + b]! * eri[((m * NSO + nIdx) * NSO + E) * NSO + j]!;
          }
          // − Σ_nf (½ t_jn^fb + t_j^f t_n^b) ⟨mn||ef⟩
          for (let nIdx = 0; nIdx < NOCC; nIdx++) {
            for (let f = 0; f < NVIRT; f++) {
              const F_v = f + NOCC;
              const t2val = T2[((j * NOCC + nIdx) * NVIRT + f) * NVIRT + b]!;
              const t1prod = T1[j * NVIRT + f]! * T1[nIdx * NVIRT + b]!;
              const eriVal = eri[((m * NSO + nIdx) * NSO + E) * NSO + F_v]!;
              s -= (0.5 * t2val + t1prod) * eriVal;
            }
          }
          W[((m * NVIRT + b) * NVIRT + e) * NOCC + j] = s;
        }
      }
    }
  }
  return W;
}

// ─────────────────────────────────────────────────────────────
// T1 residual (RHS of  D_ia · t_i^a = RHS):
//   RHS = f_ia + Σ_e t_i^e F_ae − Σ_m t_m^a F_mi + Σ_me t_im^ae F_me
//        − Σ_nf t_n^f ⟨na||if⟩
//        − ½ Σ_mef t_im^ef ⟨ma||ef⟩
//        − ½ Σ_mne t_mn^ae ⟨nm||ei⟩
// f_ia = 0 in canonical RHF.
// ─────────────────────────────────────────────────────────────
function makeRHS_T1(T1: Float64Array, T2: Float64Array,
                    F_ae: Float64Array, F_mi: Float64Array, F_me: Float64Array,
                    eri: Float64Array, NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const r = new Float64Array(NOCC * NVIRT);
  for (let i = 0; i < NOCC; i++) {
    for (let a = 0; a < NVIRT; a++) {
      const A = a + NOCC;
      let s = 0;
      // Σ_e t_i^e F_ae
      for (let e = 0; e < NVIRT; e++) {
        s += T1[i * NVIRT + e]! * F_ae[a * NVIRT + e]!;
      }
      // − Σ_m t_m^a F_mi
      for (let m = 0; m < NOCC; m++) {
        s -= T1[m * NVIRT + a]! * F_mi[m * NOCC + i]!;
      }
      // + Σ_me t_im^ae F_me
      for (let m = 0; m < NOCC; m++) {
        for (let e = 0; e < NVIRT; e++) {
          s += T2[((i * NOCC + m) * NVIRT + a) * NVIRT + e]! * F_me[m * NVIRT + e]!;
        }
      }
      // − Σ_nf t_n^f ⟨na||if⟩
      for (let nIdx = 0; nIdx < NOCC; nIdx++) {
        for (let f = 0; f < NVIRT; f++) {
          const F_v = f + NOCC;
          s -= T1[nIdx * NVIRT + f]! * eri[((nIdx * NSO + A) * NSO + i) * NSO + F_v]!;
        }
      }
      // − ½ Σ_mef t_im^ef ⟨ma||ef⟩
      for (let m = 0; m < NOCC; m++) {
        for (let e = 0; e < NVIRT; e++) {
          const E = e + NOCC;
          for (let f = 0; f < NVIRT; f++) {
            const F_v = f + NOCC;
            s -= 0.5 * T2[((i * NOCC + m) * NVIRT + e) * NVIRT + f]! *
                       eri[((m * NSO + A) * NSO + E) * NSO + F_v]!;
          }
        }
      }
      // − ½ Σ_mne t_mn^ae ⟨nm||ei⟩
      for (let m = 0; m < NOCC; m++) {
        for (let nIdx = 0; nIdx < NOCC; nIdx++) {
          for (let e = 0; e < NVIRT; e++) {
            const E = e + NOCC;
            s -= 0.5 * T2[((m * NOCC + nIdx) * NVIRT + a) * NVIRT + e]! *
                       eri[((nIdx * NSO + m) * NSO + E) * NSO + i]!;
          }
        }
      }
      r[i * NVIRT + a] = s;
    }
  }
  return r;
}

// ─────────────────────────────────────────────────────────────
// T2 residual (RHS of  D_ijab · t_ij^ab = RHS):
//   RHS = ⟨ij||ab⟩
//       + P(ab) Σ_e t_ij^ae · (F_be − ½ Σ_m t_m^b F_me)
//       − P(ij) Σ_m t_im^ab · (F_mj + ½ Σ_e t_j^e F_me)
//       + ½ Σ_mn τ_mn^ab W_mnij
//       + ½ Σ_ef τ_ij^ef W_abef
//       + P(ij) P(ab) Σ_me [ t_im^ae W_mbej − t_i^e t_m^a ⟨mb||ej⟩ ]
//       + P(ij) Σ_e t_i^e ⟨ab||ej⟩
//       − P(ab) Σ_m t_m^a ⟨mb||ij⟩
// ─────────────────────────────────────────────────────────────
function makeRHS_T2(T1: Float64Array, T2: Float64Array, tau: Float64Array,
                    F_ae: Float64Array, _F_mi: Float64Array, F_me: Float64Array,
                    W_mnij: Float64Array, W_abef: Float64Array, W_mbej: Float64Array,
                    eri: Float64Array, NOCC: number, NVIRT: number, NSO: number): Float64Array {
  const F_mi = _F_mi;
  const r = new Float64Array(NOCC * NOCC * NVIRT * NVIRT);

  // Helper: an effective F_be_eff[b,e] = F_be − ½ Σ_m t_m^b F_me  (for the t_ij^ae · F_be_eff term)
  const F_be_eff = new Float64Array(NVIRT * NVIRT);
  for (let bIdx = 0; bIdx < NVIRT; bIdx++) {
    for (let eIdx = 0; eIdx < NVIRT; eIdx++) {
      let s = F_ae[bIdx * NVIRT + eIdx]!;
      for (let m = 0; m < NOCC; m++) {
        s -= 0.5 * T1[m * NVIRT + bIdx]! * F_me[m * NVIRT + eIdx]!;
      }
      F_be_eff[bIdx * NVIRT + eIdx] = s;
    }
  }
  // Effective F_mj_eff[m,j] = F_mj + ½ Σ_e t_j^e F_me
  const F_mj_eff = new Float64Array(NOCC * NOCC);
  for (let m = 0; m < NOCC; m++) {
    for (let j = 0; j < NOCC; j++) {
      let s = F_mi[m * NOCC + j]!;
      for (let e = 0; e < NVIRT; e++) {
        s += 0.5 * T1[j * NVIRT + e]! * F_me[m * NVIRT + e]!;
      }
      F_mj_eff[m * NOCC + j] = s;
    }
  }

  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        const A = a + NOCC;
        for (let b = 0; b < NVIRT; b++) {
          const B = b + NOCC;
          // ⟨ij||ab⟩
          let s = oovv(eri, NSO, NOCC, i, j, a, b);

          // + P(ab) Σ_e t_ij^ae F_be_eff
          for (let e = 0; e < NVIRT; e++) {
            s += T2[((i * NOCC + j) * NVIRT + a) * NVIRT + e]! * F_be_eff[b * NVIRT + e]!;
            s -= T2[((i * NOCC + j) * NVIRT + b) * NVIRT + e]! * F_be_eff[a * NVIRT + e]!;
          }
          // − P(ij) Σ_m t_im^ab F_mj_eff
          for (let m = 0; m < NOCC; m++) {
            s -= T2[((i * NOCC + m) * NVIRT + a) * NVIRT + b]! * F_mj_eff[m * NOCC + j]!;
            s += T2[((j * NOCC + m) * NVIRT + a) * NVIRT + b]! * F_mj_eff[m * NOCC + i]!;
          }
          // + ½ Σ_mn τ_mn^ab W_mnij
          let q1 = 0;
          for (let m = 0; m < NOCC; m++) {
            for (let nIdx = 0; nIdx < NOCC; nIdx++) {
              q1 += tau[((m * NOCC + nIdx) * NVIRT + a) * NVIRT + b]! *
                    W_mnij[((m * NOCC + nIdx) * NOCC + i) * NOCC + j]!;
            }
          }
          s += 0.5 * q1;
          // + ½ Σ_ef τ_ij^ef W_abef
          let q2 = 0;
          for (let e = 0; e < NVIRT; e++) {
            for (let f = 0; f < NVIRT; f++) {
              q2 += tau[((i * NOCC + j) * NVIRT + e) * NVIRT + f]! *
                    W_abef[((a * NVIRT + b) * NVIRT + e) * NVIRT + f]!;
            }
          }
          s += 0.5 * q2;
          // + P(ij) P(ab) Σ_me [ t_im^ae W_mbej − t_i^e t_m^a ⟨mb||ej⟩ ]
          // Compute base T(i,j,a,b) = Σ_me [ ... ], then add T − Tji − Tab + Tjiab.
          for (let m = 0; m < NOCC; m++) {
            for (let e = 0; e < NVIRT; e++) {
              const E = e + NOCC;
              const w_iajb = W_mbej[((m * NVIRT + b) * NVIRT + e) * NOCC + j]!;
              const w_jaib = W_mbej[((m * NVIRT + b) * NVIRT + e) * NOCC + i]!;
              const w_ibja = W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + j]!;
              const w_jbia = W_mbej[((m * NVIRT + a) * NVIRT + e) * NOCC + i]!;
              const eri_jb = eri[((m * NSO + B) * NSO + E) * NSO + j]!;
              const eri_ib = eri[((m * NSO + B) * NSO + E) * NSO + i]!;
              const eri_ja = eri[((m * NSO + A) * NSO + E) * NSO + j]!;
              const eri_ia = eri[((m * NSO + A) * NSO + E) * NSO + i]!;
              const t_imae = T2[((i * NOCC + m) * NVIRT + a) * NVIRT + e]!;
              const t_jmae = T2[((j * NOCC + m) * NVIRT + a) * NVIRT + e]!;
              const t_imbe = T2[((i * NOCC + m) * NVIRT + b) * NVIRT + e]!;
              const t_jmbe = T2[((j * NOCC + m) * NVIRT + b) * NVIRT + e]!;
              const t_ie = T1[i * NVIRT + e]!;
              const t_je = T1[j * NVIRT + e]!;
              const t_ma = T1[m * NVIRT + a]!;
              const t_mb = T1[m * NVIRT + b]!;
              // T(i,j,a,b) = t_im^ae W_mbej − t_i^e t_m^a ⟨mb||ej⟩
              const T_ijab = t_imae * w_iajb - t_ie * t_ma * eri_jb;
              const T_jiab = t_jmae * w_jaib - t_je * t_ma * eri_ib;
              const T_ijba = t_imbe * w_ibja - t_ie * t_mb * eri_ja;
              const T_jiba = t_jmbe * w_jbia - t_je * t_mb * eri_ia;
              s += T_ijab - T_jiab - T_ijba + T_jiba;
            }
          }
          // + P(ij) Σ_e t_i^e ⟨ab||ej⟩  =  Σ_e t_i^e ⟨ab||ej⟩ − Σ_e t_j^e ⟨ab||ei⟩
          for (let e = 0; e < NVIRT; e++) {
            const E = e + NOCC;
            s += T1[i * NVIRT + e]! * eri[((A * NSO + B) * NSO + E) * NSO + j]!;
            s -= T1[j * NVIRT + e]! * eri[((A * NSO + B) * NSO + E) * NSO + i]!;
          }
          // − P(ab) Σ_m t_m^a ⟨mb||ij⟩  =  −Σ_m t_m^a ⟨mb||ij⟩ + Σ_m t_m^b ⟨ma||ij⟩
          for (let m = 0; m < NOCC; m++) {
            s -= T1[m * NVIRT + a]! * eri[((m * NSO + B) * NSO + i) * NSO + j]!;
            s += T1[m * NVIRT + b]! * eri[((m * NSO + A) * NSO + i) * NSO + j]!;
          }

          r[((i * NOCC + j) * NVIRT + a) * NVIRT + b] = s;
        }
      }
    }
  }
  return r;
}

// ─────────────────────────────────────────────────────────────
// Energy:
//   E_corr = Σ_ia f_ia t_i^a + ¼ Σ_ijab ⟨ij||ab⟩ τ_ij^ab
// f_ia = 0 in canonical RHF, so just the τ contraction. Equivalent:
//   E_corr = ¼ Σ ⟨ij||ab⟩ t_ij^ab + ½ Σ ⟨ij||ab⟩ t_i^a t_j^b
// ─────────────────────────────────────────────────────────────
function computeECorr(T1: Float64Array, T2: Float64Array, eri: Float64Array,
                      NOCC: number, NVIRT: number, NSO: number): number {
  let e = 0;
  for (let i = 0; i < NOCC; i++) {
    for (let j = 0; j < NOCC; j++) {
      for (let a = 0; a < NVIRT; a++) {
        for (let b = 0; b < NVIRT; b++) {
          const v = oovv(eri, NSO, NOCC, i, j, a, b);
          const t2 = T2[((i * NOCC + j) * NVIRT + a) * NVIRT + b]!;
          const t1prod = T1[i * NVIRT + a]! * T1[j * NVIRT + b]!
                       - T1[i * NVIRT + b]! * T1[j * NVIRT + a]!;
          e += 0.25 * v * (t2 + t1prod);
        }
      }
    }
  }
  return e;
}
