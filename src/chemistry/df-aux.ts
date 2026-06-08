// df-aux.ts — Aux-basis density fitting (proper RI).
//
// Replaces the rank-4 ERI tensor (μν|λσ) with a rank-3 B-tensor:
//   (μν|λσ) ≈ Σ_P B[μν, P] · B[λσ, P]
//
// Built from:
//   V[μν, P] = (μν|P)   — 3-index ERI between orbital pair and aux
//   M[P, Q]  = (P|Q)    — 2-index ERI metric in aux space
//   B = V · M^(-1/2)    — contracted along the aux dimension
//
// Different from the shipped CD-DF (`df.ts`) which builds the FULL
// n⁴ ERI first then Cholesky-decomposes. This path NEVER builds the
// 4-index tensor — V is n²·n_aux (n_aux ≪ n² typically), M is
// n_aux². Memory: 1.65 GB on benzene cc-pVDZ becomes ~30 MB.
//
// Phase 1 (this commit): the algorithm is wired but uses the ORBITAL
// basis as the auxiliary basis (saves ~10× memory). Phase 2 would
// load a proper cc-pVDZ-jkfit aux basis (~3× smaller → ~30× total
// memory savings vs 4-index).

import type { CGShell } from "./integrals-cg.js";
import type { DFResult } from "./df.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { sabAvailable } from "../parallel/worker-pool.js";
import { getSharedWorkerPool } from "../parallel/worker-pool-shared.js";
import { formBFromCholeskyParallel } from "../parallel/parallel-form-b-cholesky.js";

/** Pivoted incomplete Cholesky of a PSD n × n matrix M, returning
 *  L such that M ≈ L · L^T with rank ≤ n. L has shape n × r where
 *  r is the discovered rank. Pivot indices are returned to enable
 *  back-substitution against L^T.
 *
 *  Drops dimensions where the remaining diagonal residual falls
 *  below `threshold`. For a well-conditioned PSD matrix this is
 *  full rank; for the rank-deficient case (e.g., auto-aux M with
 *  cross-atom redundancy at benzene scale) it naturally truncates
 *  to the effective rank without producing spurious modes the way
 *  eigendecomp + regularization does.
 *
 *  Cost: O(n²·r). Memory: O(n·r). */
function pivotedCholesky(
  M: Float64Array,
  n: number,
  threshold: number,
): { L: Float64Array; pivots: number[] } {
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = M[i * n + i]!;
  const L_cols: Float64Array[] = [];
  const pivots: number[] = [];

  for (let step = 0; step < n; step++) {
    let piv = -1;
    let pmax = 0;
    for (let i = 0; i < n; i++) {
      if (d[i]! > pmax) { pmax = d[i]!; piv = i; }
    }
    if (piv < 0 || pmax < threshold) break;
    const sqrtPiv = Math.sqrt(pmax);
    const newCol = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let v = M[i * n + piv]!;
      for (let j = 0; j < L_cols.length; j++) {
        v -= L_cols[j]![i]! * L_cols[j]![piv]!;
      }
      newCol[i] = v / sqrtPiv;
    }
    L_cols.push(newCol);
    pivots.push(piv);
    for (let i = 0; i < n; i++) {
      const v = newCol[i]!;
      d[i] = Math.max(0, d[i]! - v * v);
    }
    d[piv] = 0;
  }

  // Pack columns into row-major L[i, k] = L_cols[k][i].
  const r = L_cols.length;
  const L = new Float64Array(n * r);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < r; k++) {
      L[i * r + k] = L_cols[k]![i]!;
    }
  }
  return { L, pivots };
}

/** TS form_b_from_cholesky — kept as the active path.
 *
 *  An honest negative documented 2026-05-28: porting this to Rust+WASM
 *  was SLOWER than TS (91 s vs 40 s on naphthalene cc-pVDZ at n=190,
 *  n_aux=1080). Two reasons:
 *   1. wasm-bindgen `&[f64]` copies the V tensor (~312 MB on
 *      naphthalene) into WASM linear memory every call.
 *   2. The piv_k-indirect access into V and L is cache-unfriendly;
 *      WASM lost the same way TS did, plus paid the copy cost.
 *
 *  Output B is contiguous (μν, k), so V8 JIT can prefetch the writes
 *  better than expected. Inner k loop has serial dependency on
 *  B[μν, j<k], so SIMD doesn't help the back-substitution.
 *
 *  Cost: O(n_orb² · r²). Memory: O(n_orb² · r). */
function formBFromCholesky(
  V: Float64Array,
  L: Float64Array,
  pivots: number[],
  nOrb: number,
  nAux: number,
): { B: Float64Array; r: number } {
  const r = pivots.length;
  const B = new Float64Array(nOrb * nOrb * r);
  for (let mu = 0; mu < nOrb; mu++) {
    for (let nu = 0; nu < nOrb; nu++) {
      const vBase = (mu * nOrb + nu) * nAux;
      const bBase = (mu * nOrb + nu) * r;
      for (let k = 0; k < r; k++) {
        const pivK = pivots[k]!;
        let s = V[vBase + pivK]!;
        for (let j = 0; j < k; j++) {
          s -= L[pivK * r + j]! * B[bBase + j]!;
        }
        const diag = L[pivK * r + k]!;
        B[bBase + k] = s / diag;
      }
    }
  }
  return { B, r };
}

/**
 * Generate an "auto-aux" basis from the orbital basis: decontract each
 * primitive into a separate single-primitive aux shell and (optionally)
 * extend the angular momentum range to L_orb_max + extraL to span
 * orbital pair products.
 *
 * This is a stop-gap between "orbital-as-aux" (insufficient) and
 * proper Weigend cc-pVDZ-jkfit data (which we don't have tables for).
 * Auto-aux gives 2-3× richer aux for arbitrary orbital basis at zero
 * data cost, but exponents aren't optimal — expect mHa-class DF
 * errors rather than the sub-µHa of purpose-built jkfit.
 *
 *   extraL = 0:  decontract only, same angular range as orbital
 *   extraL = 1:  add L_orb + 1 aux at each orbital exponent
 *   extraL = 2:  also add L_orb + 2 (e.g., g-functions for d-orbital products)
 *
 * Recommended: extraL = 2 to cover (d|d) → g products from cc-pVDZ.
 */
export function generateAutoAux(
  orbitalShells: readonly CGShell[],
  extraL = 2,
): CGShell[] {
  const auxShells: CGShell[] = [];
  // Track which atom-centered primitive-exponent + angular combos we've
  // already emitted so we don't duplicate (different orbital shells on
  // the same atom often share primitives with different contraction
  // coefs — for aux we just want the exponent once per angular slot).
  const seen = new Set<string>();
  const key = (cx: number, cy: number, cz: number, α: number, ax: number, ay: number, az: number): string =>
    `${cx.toFixed(8)},${cy.toFixed(8)},${cz.toFixed(8)},${α.toFixed(10)},${ax},${ay},${az}`;

  // Emit Cartesian-component shells for a given (center, α, total-L).
  const emitL = (center: readonly [number, number, number], α: number, totalL: number, label: string): void => {
    for (let ax = totalL; ax >= 0; ax--) {
      for (let ay = totalL - ax; ay >= 0; ay--) {
        const az = totalL - ax - ay;
        const k = key(center[0], center[1], center[2], α, ax, ay, az);
        if (seen.has(k)) continue;
        seen.add(k);
        auxShells.push({
          center,
          alpha: [α],
          c: [1.0],
          angular: [ax, ay, az],
          label: `${label}:α=${α.toExponential(2)}`,
        });
      }
    }
  };

  // Empirically extraL=1 is the production sweet spot on cc-pVDZ:
  //   - decontract each primitive, emit at its own Lorb
  //   - PLUS emit L+1 aux at every primitive exponent (rich f-aux
  //     coverage from p AND s primitive exponents, not just one
  //     value from the d-orbital)
  //
  // extraL=2 produces severe linear dependence (313 aux for 25-orb
  // H₂O) because each primitive contributes redundant g-aux at
  // similar exponents. The eigendecomp orthogonalization fails
  // catastrophically, giving meaningless B and a runaway SCF.
  // A jkfit-style curated aux basis is the proper fix; for now,
  // stick to extraL ≤ 1 in production code.
  for (const sh of orbitalShells) {
    const Lorb = sh.angular[0] + sh.angular[1] + sh.angular[2];
    for (const α of sh.alpha) {
      emitL(sh.center, α, Lorb, `aux-L${Lorb}`);
      for (let dl = 1; dl <= extraL; dl++) {
        emitL(sh.center, α, Lorb + dl, `aux-L${Lorb + dl}`);
      }
    }
  }
  return auxShells;
}

interface WasmEriModule {
  default(): Promise<unknown>;
  eri_3idx_build(
    nOrb: number, nAux: number,
    nPrimsOrb: Uint32Array, primOffOrb: Uint32Array,
    alphaOrb: Float64Array, cOrb: Float64Array,
    centerOrb: Float64Array, angularOrb: Int32Array,
    nPrimsAux: Uint32Array, primOffAux: Uint32Array,
    alphaAux: Float64Array, cAux: Float64Array,
    centerAux: Float64Array, angularAux: Int32Array,
  ): Float64Array;
  eri_2idx_build(
    nAux: number,
    nPrimsAux: Uint32Array, primOffAux: Uint32Array,
    alphaAux: Float64Array, cAux: Float64Array,
    centerAux: Float64Array, angularAux: Int32Array,
  ): Float64Array;
  /** Builds (μν|P) for μ ∈ `mus` only, all ν ≥ μ, all P. Returns a packed
   *  [μ, ν, P, value, …] array of length 4·K — never the full n²·n_aux tensor.
   *  Used by the streaming per-slice build so no tab materializes all of V. */
  eri_3idx_build_slice(
    mus: Uint32Array,
    nOrbital: number, nAux: number,
    nPrimsOrb: Uint32Array, primOffOrb: Uint32Array,
    alphaOrb: Float64Array, cOrb: Float64Array,
    centerOrb: Float64Array, angularOrb: Int32Array,
    nPrimsAux: Uint32Array, primOffAux: Uint32Array,
    alphaAux: Float64Array, cAux: Float64Array,
    centerAux: Float64Array, angularAux: Int32Array,
  ): Float64Array;
  form_b_tensor(
    n: number,
    nAux: number,
    v: Float64Array,
    u: Float64Array,
    invSqrtLam: Float64Array,
  ): Float64Array;
  /** Mode-basis DF projection for one μ-block (streaming swarm). Returns
   *  [rows·n·m_local], B[μν,m]=Σ_Q V[μν,Q]·w_cm[m·n_aux+Q] for ν ≥ μ. */
  df_project_block_modes(
    vblk: Float64Array,
    wCm: Float64Array,
    rows: number, n: number, nAux: number, mLocal: number, mu0: number,
  ): Float64Array;
  form_b_from_cholesky(
    n: number, nAux: number, r: number,
    v: Float64Array,
    l: Float64Array,
    pivots: Uint32Array,
  ): Float64Array;
}

let wasmModule: WasmEriModule | null = null;
async function loadWasm(): Promise<WasmEriModule> {
  if (wasmModule) return wasmModule;
  const mod = await import(
    /* @vite-ignore */
    "../../wasm-eri/pkg/wasm_eri.js" as string,
  ) as WasmEriModule;
  // The wasm-bindgen `--target web` init resolves the .wasm via fetch(), which
  // cannot read file:// URLs under Node (vitest). In Node only, read the bytes
  // and hand them to default(); the browser/worker path is byte-for-byte
  // unchanged. This makes the whole DF layer unit-testable without a browser.
  const isNode = typeof process !== "undefined" && !!process.versions?.node;
  if (isNode) {
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const wasmPath = fileURLToPath(
      new URL("../../wasm-eri/pkg/wasm_eri_bg.wasm", import.meta.url),
    );
    const bytes = await readFile(wasmPath);
    await (mod as unknown as {
      default(o: { module_or_path: BufferSource }): Promise<unknown>;
    }).default({ module_or_path: bytes });
  } else {
    await mod.default();
  }
  wasmModule = mod;
  return mod;
}

function packShells(shells: readonly CGShell[]): {
  nPrims: Uint32Array; primOff: Uint32Array;
  alpha: Float64Array; c: Float64Array;
  center: Float64Array; angular: Int32Array;
} {
  const sz = shells.length;
  const nPrims = new Uint32Array(sz);
  const primOff = new Uint32Array(sz);
  let total = 0;
  for (let i = 0; i < sz; i++) {
    nPrims[i] = shells[i]!.alpha.length;
    primOff[i] = total;
    total += shells[i]!.alpha.length;
  }
  const alpha = new Float64Array(total);
  const c = new Float64Array(total);
  for (let i = 0; i < sz; i++) {
    const off = primOff[i]!;
    for (let p = 0; p < shells[i]!.alpha.length; p++) {
      alpha[off + p] = shells[i]!.alpha[p]!;
      c[off + p] = shells[i]!.c[p]!;
    }
  }
  const center = new Float64Array(sz * 3);
  const angular = new Int32Array(sz * 3);
  for (let i = 0; i < sz; i++) {
    center[i * 3] = shells[i]!.center[0];
    center[i * 3 + 1] = shells[i]!.center[1];
    center[i * 3 + 2] = shells[i]!.center[2];
    angular[i * 3] = shells[i]!.angular[0];
    angular[i * 3 + 1] = shells[i]!.angular[1];
    angular[i * 3 + 2] = shells[i]!.angular[2];
  }
  return { nPrims, primOff, alpha, c, center, angular };
}

/** CPU (f64 WASM) 2-index DF metric M[P,Q] = (P|Q), row-major n_aux × n_aux.
 *  The reference the GPU f32 metric kernel (df-gpu.ts) is validated against. */
export async function buildMetric2idxCPU(auxShells: readonly CGShell[]): Promise<Float64Array> {
  const mod = await loadWasm();
  const aux = packShells(auxShells);
  return mod.eri_2idx_build(
    auxShells.length, aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
}

/** Build the mode-basis DF tensor from a PRECOMPUTED 3-index V[μν,Q] (e.g. one
 *  the GPU produced). Returns the same mode-basis B as buildAuxBasisDFStreaming
 *  (nKept columns), so it drops straight into buildJK_DF / runRHFSCF useDF. Used
 *  to run HF end-to-end on GPU-computed integrals. */
export async function buildBFromV(
  orbShells: readonly CGShell[], auxShells: readonly CGShell[],
  V: Float64Array, metricRegularization = 1e-10,
): Promise<DFResult> {
  const mod = await loadWasm();
  const aux = packShells(auxShells);
  const n = orbShells.length;
  const nAux = auxShells.length;
  const M = mod.eri_2idx_build(
    nAux, aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
  const eig = eigsymmetric(M, nAux);
  const kept: number[] = [];
  for (let i = 0; i < nAux; i++) if (eig.values[i]! > metricRegularization) kept.push(i);
  const nKept = kept.length;
  // W mode-major: Wcm[m·n_aux + Q] = U[Q, kept_m]·λ^(−1/2). Project in WASM SIMD
  // (df_project_block_modes) over μ-blocks, not a TS triple loop — the projection
  // is the build's dominant ~n⁴ term.
  const Wcm = new Float64Array(nKept * nAux);
  for (let m = 0; m < nKept; m++) {
    const i = kept[m]!;
    const inv = 1.0 / Math.sqrt(eig.values[i]!);
    const col = i * nAux;
    const wBase = m * nAux;
    for (let Q = 0; Q < nAux; Q++) Wcm[wBase + Q] = eig.vectors[col + Q]! * inv;
  }
  const B = new Float64Array(n * n * nKept);
  const muBlock = 8;
  for (let mu0 = 0; mu0 < n; mu0 += muBlock) {
    const mu1 = Math.min(mu0 + muBlock, n);
    const rows = mu1 - mu0;
    // V rows [mu0,mu1) are a contiguous slice; the kernel reads ν ≥ μ.
    const vblk = V.subarray(mu0 * n * nAux, mu1 * n * nAux);
    const Bblk = mod.df_project_block_modes(vblk, Wcm, rows, n, nAux, nKept, mu0);
    B.set(Bblk, mu0 * n * nKept);
  }
  // Symmetrize: B[νμ] = B[μν] for μ < ν.
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu + 1; nu < n; nu++) {
      const up = (mu * n + nu) * nKept;
      const lo = (nu * n + mu) * nKept;
      for (let m = 0; m < nKept; m++) B[lo + m] = B[up + m]!;
    }
  }
  return { B, nAux: nKept, threshold: metricRegularization, n };
}

/** CPU (f64 WASM) 3-index tensor V[μν,P] = (μν|P), layout (μ·n+ν)·n_aux + P.
 *  The reference the GPU f32 3-index kernel (df-gpu.ts) is validated against. */
export async function buildV3idxCPU(
  orbShells: readonly CGShell[], auxShells: readonly CGShell[],
): Promise<Float64Array> {
  const mod = await loadWasm();
  const orb = packShells(orbShells);
  const aux = packShells(auxShells);
  return mod.eri_3idx_build(
    orbShells.length, auxShells.length,
    orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
    aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
}

/**
 * Build the density-fitting B-tensor from explicit 3-index and
 * 2-index integrals. Aux basis defaults to the orbital basis (Phase 1
 * PoC); for production a proper cc-pVDZ-jkfit basis would be supplied.
 *
 * Algorithm:
 *   V[μν, P] = (μν|P)                    via eri_3idx_build
 *   M[P, Q]  = (P|Q)                     via eri_2idx_build
 *   M = U · Λ · U^T                      eigendecomp (Jacobi)
 *   M^(-1/2) = U · Λ^(-1/2) · U^T        ignoring λ_i < ε (regularize)
 *   B[μν, P] = Σ_Q V[μν, Q] · M^(-1/2)[Q, P]
 *
 * Returns the standard DFResult shape used by `buildJK_DF`.
 */
export async function buildAuxBasisDF(
  orbitalShells: readonly CGShell[],
  auxShells?: readonly CGShell[],
  metricRegularization = 1e-10,
): Promise<DFResult> {
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  // 3-index V[μν, P] and 2-index M[P, Q].
  const V = mod.eri_3idx_build(
    n, nAux,
    orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
    aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
  const M = mod.eri_2idx_build(
    nAux,
    aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );

  // Eigendecomp of M. (Jacobi returns ascending eigenvalues + column-major V_eig.)
  const eig = eigsymmetric(M, nAux);
  // M^(-1/2) = U · diag(1/√λ) · U^T  (where U columns are eigenvectors).
  // Regularize: zero out eigenvalues below `metricRegularization`.
  const invSqrtLam = new Float64Array(nAux);
  let nKept = 0;
  for (let i = 0; i < nAux; i++) {
    const lam = eig.values[i]!;
    if (lam > metricRegularization) {
      invSqrtLam[i] = 1.0 / Math.sqrt(lam);
      nKept++;
    } else {
      invSqrtLam[i] = 0.0;
    }
  }

  // Form M^(-1/2) (we don't need it explicitly — fold into B directly).
  // B[μν, P] = Σ_Q V[μν, Q] · (U · Λ^(-1/2) · U^T)[Q, P]
  //          = Σ_Q Σ_i V[μν, Q] · U[Q, i] · λ_i^(-1/2) · U[P, i]
  // Pre-compute T[μν, i] = Σ_Q V[μν, Q] · U[Q, i] · λ_i^(-1/2)
  // Then B[μν, P] = Σ_i T[μν, i] · U[P, i]
  //
  // Layout: eig.vectors is column-major, so U[Q, i] = vectors[i*nAux + Q].
  // The full B = V · U · diag(λ⁻¹⸍²) · Uᵀ matmul runs in Rust+WASM
  // with f64x2 SIMD on the inner P loop — ~5× over the TS version.
  const B = mod.form_b_tensor(n, nAux, V, eig.vectors, invSqrtLam);

  return {
    B,
    nAux: nKept,
    threshold: metricRegularization,
    n,
  };
}

/** Result of {@link buildAuxBasisDFStreaming}: a DF tensor plus the peak number
 *  of f64 V-tensor elements held at once during the build. For the streaming
 *  path that peak is `muBlock·n·n_aux`, NOT `n²·n_aux` — the evidence that the
 *  full 3-index tensor is never materialized on any single tab. */
export interface StreamingDFResult extends DFResult {
  readonly peakVFloats: number;
  /** Total kept modes across ALL tabs (this slice holds `nAux` of them). Lets a
   *  tab know the full B-tensor size (n²·nKeptTotal·8) without ever building it. */
  readonly nKeptTotal: number;
}

/**
 * Streaming, integral-direct, mode-partitioned auxiliary-basis DF build.
 *
 * Fits the same Coulomb metric as {@link buildAuxBasisDF}, but produced so the
 * full V[μν,Q] 3-index tensor is NEVER resident: V is built one μ-block at a
 * time and immediately projected onto the regularized inverse-metric modes
 *
 *     B[μν,i] = Σ_Q V[μν,Q] · U[Q,i] · λ_i^(−1/2)      (i = kept eigen-modes),
 *
 * which is itself a valid DF tensor — Σ_i B[μν,i]B[λσ,i] = (μν|λσ)_fit, because
 * Σ_i U[Q,i]·λ_i^(−1)·U[R,i] = (M⁻¹)[Q,R]. The mode index i is exactly the axis
 * the JK build contracts over, so it is the natural axis to PARTITION across a
 * swarm: pass `modeStart`/`modeEnd` and a tab builds only B[:, :, modeStart:modeEnd]
 * — still without ever holding all of V. Partial JK over disjoint mode ranges
 * sums to the full JK. With the default full mode range this is a single-tab
 * drop-in whose peak V footprint is `muBlock·n·n_aux`.
 *
 * The eri_3idx_build_slice kernel returns only ν ≥ μ pairs, so each block fills
 * the upper triangle of its B rows; the B slice is symmetrized at the end (the
 * B slice is the kept output — full V is not).
 */
export async function buildAuxBasisDFStreaming(
  orbitalShells: readonly CGShell[],
  auxShells?: readonly CGShell[],
  metricRegularization = 1e-10,
  opts: {
    modeStart?: number;
    modeEnd?: number;
    muBlock?: number;
    /** Swarm convenience: build this tab's contiguous share of the kept modes.
     *  Each tab computes the (deterministic) eigendecomposition independently,
     *  so they agree on nKept and tile it without coordination. Overrides
     *  modeStart/modeEnd. */
    partition?: { tab: number; of: number };
  } = {},
): Promise<StreamingDFResult> {
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  // 2-index metric → eigendecomposition (ascending λ, column-major U).
  const M = mod.eri_2idx_build(
    nAux, aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
  const eig = eigsymmetric(M, nAux);
  // Kept modes: λ_i > reg. U[Q,i] = eig.vectors[i*nAux + Q].
  const keptModes: number[] = [];
  for (let i = 0; i < nAux; i++) if (eig.values[i]! > metricRegularization) keptModes.push(i);
  const nKept = keptModes.length;

  let modeStart = opts.modeStart ?? 0;
  let modeEnd = opts.modeEnd ?? nKept;
  if (opts.partition) {
    const { tab, of } = opts.partition;
    modeStart = Math.floor((tab * nKept) / of);
    modeEnd = Math.floor(((tab + 1) * nKept) / of);
  }
  const mLocal = modeEnd - modeStart;

  // W mode-major: Wcm[m·n_aux + Q] = U[Q, i_m]·λ_{i_m}^(−1/2). Mode-major so
  // each mode's coefficient vector is contiguous in Q for the SIMD dot product.
  const Wcm = new Float64Array(mLocal * nAux);
  for (let m = 0; m < mLocal; m++) {
    const i = keptModes[modeStart + m]!;
    const inv = 1.0 / Math.sqrt(eig.values[i]!);
    const col = i * nAux;
    const wBase = m * nAux;
    for (let Q = 0; Q < nAux; Q++) Wcm[wBase + Q] = eig.vectors[col + Q]! * inv;
  }

  // Output: local mode slice B[(μ·n+ν)·mLocal + m]. Never the full tensor.
  const B = new Float64Array(n * n * mLocal);
  const muBlock = Math.max(1, opts.muBlock ?? 8);
  const Vblk = new Float64Array(muBlock * n * nAux); // transient, reused per block
  const peakVFloats = Vblk.length;

  for (let mu0 = 0; mu0 < n; mu0 += muBlock) {
    const mu1 = Math.min(mu0 + muBlock, n);
    const rows = mu1 - mu0;
    const mus = new Uint32Array(rows);
    for (let r = 0; r < rows; r++) mus[r] = mu0 + r;

    // Build V[μ∈block, ν≥μ, Q] as packed [μ,ν,P,val]; unpack into Vblk.
    Vblk.fill(0, 0, rows * n * nAux);
    const packed = mod.eri_3idx_build_slice(
      mus, n, nAux,
      orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
      aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
    );
    const K = packed.length / 4;
    for (let k = 0; k < K; k++) {
      const base = k * 4;
      const mu = packed[base]! | 0;
      const nu = packed[base + 1]! | 0;
      const P = packed[base + 2]! | 0;
      Vblk[((mu - mu0) * n + nu) * nAux + P] = packed[base + 3]!;
    }

    // Project this block onto the mode slice in WASM (SIMD f64x2 dot products);
    // upper-triangle (ν ≥ μ) only — symmetrized below.
    const Bblk = mod.df_project_block_modes(
      Vblk.subarray(0, rows * n * nAux), Wcm, rows, n, nAux, mLocal, mu0,
    );
    B.set(Bblk, mu0 * n * mLocal);
  }

  // Symmetrize: B[νμ] = B[μν] for μ < ν (B slice is the kept output).
  for (let mu = 0; mu < n; mu++) {
    for (let nu = mu + 1; nu < n; nu++) {
      const up = (mu * n + nu) * mLocal;
      const lo = (nu * n + mu) * mLocal;
      for (let m = 0; m < mLocal; m++) B[lo + m] = B[up + m]!;
    }
  }

  return { B, nAux: mLocal, threshold: metricRegularization, n, peakVFloats, nKeptTotal: nKept };
}

/** Result of {@link buildAuxBasisDFStreamingCooperative}: one DF mode-slice per
 *  tab, plus build statistics that substantiate the no-redundancy / never-full-V
 *  claims. */
export interface CooperativeDFResult {
  /** One mode-partitioned DF tensor per tab. In a real swarm each lives on its
   *  own tab; summed, the partial (J,K) reproduce the single-tab build. */
  readonly slices: StreamingDFResult[];
  /** Effective DF rank (sum of slice mode counts). */
  readonly nKept: number;
  /** Peak f64 V elements held at once = one μ-block (`muBlock·n·n_aux`). */
  readonly peakVFloats: number;
  /** Number of eri_3idx_build_slice invocations. Cooperative = ceil(n/muBlock)
   *  (each integral built ONCE and fanned out to all tabs), versus
   *  `nTabs · ceil(n/muBlock)` if every tab built independently. */
  readonly kernelCalls: number;
}

/**
 * Cooperative streaming aux-DF build: the single-machine (CI) engine.
 *
 * Same mode-partitioned fit as {@link buildAuxBasisDFStreaming}, but each V
 * μ-block is built EXACTLY ONCE and immediately projected onto every tab's mode
 * slice, instead of every tab rebuilding all integrals. This removes the N×
 * redundant integral cost of fully-independent per-tab builds — the right
 * trade when the tabs share a machine (a CI runner, multiple same-origin tabs).
 * Only one V μ-block is resident at a time; each tab keeps only its B slice.
 *
 * Returns one DF slice per tab. (In this single-process validator all slices
 * are resident at once — their sum is the full B; the memory DISTRIBUTION is a
 * property of the real per-tab browser deployment, where each tab holds just
 * `slices[t]`.)
 */
export async function buildAuxBasisDFStreamingCooperative(
  orbitalShells: readonly CGShell[],
  auxShells: readonly CGShell[] | undefined,
  nTabs: number,
  metricRegularization = 1e-10,
  muBlock = 8,
): Promise<CooperativeDFResult> {
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  const M = mod.eri_2idx_build(
    nAux, aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
  const eig = eigsymmetric(M, nAux);
  const keptModes: number[] = [];
  for (let i = 0; i < nAux; i++) if (eig.values[i]! > metricRegularization) keptModes.push(i);
  const nKept = keptModes.length;

  // Contiguous mode partition across tabs; per-tab W[Q,m] = U[Q,mode]·λ^(−1/2).
  const ranges: Array<{ start: number; end: number }> = [];
  for (let t = 0; t < nTabs; t++) {
    ranges.push({
      start: Math.floor((t * nKept) / nTabs),
      end: Math.floor(((t + 1) * nKept) / nTabs),
    });
  }
  const Wt = ranges.map(({ start, end }) => {
    const mLocal = end - start;
    const W = new Float64Array(mLocal * nAux); // mode-major for the SIMD kernel
    for (let m = 0; m < mLocal; m++) {
      const i = keptModes[start + m]!;
      const inv = 1.0 / Math.sqrt(eig.values[i]!);
      const col = i * nAux;
      const wBase = m * nAux;
      for (let Q = 0; Q < nAux; Q++) W[wBase + Q] = eig.vectors[col + Q]! * inv;
    }
    return W;
  });
  const Bt = ranges.map(({ start, end }) => new Float64Array(n * n * (end - start)));

  const muBlk = Math.max(1, muBlock);
  const Vblk = new Float64Array(muBlk * n * nAux);
  const peakVFloats = Vblk.length;
  let kernelCalls = 0;

  for (let mu0 = 0; mu0 < n; mu0 += muBlk) {
    const mu1 = Math.min(mu0 + muBlk, n);
    const rows = mu1 - mu0;
    const mus = new Uint32Array(rows);
    for (let r = 0; r < rows; r++) mus[r] = mu0 + r;

    // Build this V μ-block ONCE (the cooperative win).
    Vblk.fill(0, 0, rows * n * nAux);
    const packed = mod.eri_3idx_build_slice(
      mus, n, nAux,
      orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
      aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
    );
    kernelCalls++;
    const Kp = packed.length / 4;
    for (let k = 0; k < Kp; k++) {
      const base = k * 4;
      const mu = packed[base]! | 0;
      const nu = packed[base + 1]! | 0;
      const P = packed[base + 2]! | 0;
      Vblk[((mu - mu0) * n + nu) * nAux + P] = packed[base + 3]!;
    }

    // Fan out: project this block into every tab's slice in WASM (one V build,
    // reused across all tabs — the cooperative win).
    const vsub = Vblk.subarray(0, rows * n * nAux);
    for (let t = 0; t < ranges.length; t++) {
      const { start, end } = ranges[t]!;
      const mLocal = end - start;
      if (mLocal === 0) continue;
      const Bblk = mod.df_project_block_modes(vsub, Wt[t]!, rows, n, nAux, mLocal, mu0);
      Bt[t]!.set(Bblk, mu0 * n * mLocal);
    }
  }

  // Symmetrize each slice (B[νμ] = B[μν]).
  for (let t = 0; t < ranges.length; t++) {
    const mLocal = ranges[t]!.end - ranges[t]!.start;
    if (mLocal === 0) continue;
    const B = Bt[t]!;
    for (let mu = 0; mu < n; mu++) {
      for (let nu = mu + 1; nu < n; nu++) {
        const up = (mu * n + nu) * mLocal;
        const lo = (nu * n + mu) * mLocal;
        for (let m = 0; m < mLocal; m++) B[lo + m] = B[up + m]!;
      }
    }
  }

  const slices: StreamingDFResult[] = ranges.map((rg, t) => ({
    B: Bt[t]!, nAux: rg.end - rg.start, threshold: metricRegularization, n, peakVFloats,
    nKeptTotal: nKept,
  }));
  return { slices, nKept, peakVFloats, kernelCalls };
}

/** Pivoted-Cholesky variant of `buildAuxBasisDF`. Handles
 *  rank-deficient auxiliary metrics (e.g., auto-aux with cross-atom
 *  redundancy on multi-heavy-atom systems) without the spurious-mode
 *  problem that eigendecomp + regularization produces.
 *
 *  Same algorithm as `buildAuxBasisDF` except the M-factorization
 *  step: pivoted Cholesky of M gives L such that M ≈ L · Lᵀ, then
 *  B is formed by forward-substitution against Lᵀ. The discovered
 *  rank r is the auto-detected effective basis size.
 *
 *  Returns DFResult with nAux=r (the discovered rank).
 */
export async function buildAuxBasisDFCholesky(
  orbitalShells: readonly CGShell[],
  auxShells?: readonly CGShell[],
  choleskyThreshold = 1e-8,
  poolSize = 0,
): Promise<DFResult> {
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  // ── 3-index V tensor: parallel when SAB available ──
  const useSAB = sabAvailable();
  const N = poolSize > 0 ? poolSize : Math.max(1, (navigator?.hardwareConcurrency ?? 4) - 1);
  let V: Float64Array;
  if (useSAB) {
    const vSAB = new SharedArrayBuffer(n * n * nAux * 8);
    const muAssignments: number[][] = Array.from({ length: N }, () => []);
    for (let mu = 0; mu < n; mu++) muAssignments[mu % N]!.push(mu);
    const workers = getSharedWorkerPool("wasm", N);
    await Promise.all(workers.map((w, i) => new Promise<void>((resolve, reject) => {
      const onMessage = (ev: MessageEvent): void => {
        w.removeEventListener("message", onMessage);
        if (ev.data?.ok) resolve();
        else reject(new Error(ev.data?.error ?? "worker failed"));
      };
      w.addEventListener("message", onMessage);
      w.postMessage({
        kind: "eri-3idx-wasm-slice",
        mus: muAssignments[i]!,
        muStart: 0, muEnd: n,
        nOrbital: n, nAux,
        nPrimsOrb: orb.nPrims, primOffsetsOrb: orb.primOff,
        alphaOrb: orb.alpha, cOrb: orb.c,
        centerOrb: orb.center, angularOrb: orb.angular,
        nPrimsAux: aux.nPrims, primOffsetsAux: aux.primOff,
        alphaAux: aux.alpha, cAux: aux.c,
        centerAux: aux.center, angularAux: aux.angular,
        v: vSAB,
      });
    })));
    // V points into the SAB so the parallel back-sub can share zero-copy.
    V = new Float64Array(vSAB);
  } else {
    V = mod.eri_3idx_build(
      n, nAux,
      orb.nPrims, orb.primOff, orb.alpha, orb.c, orb.center, orb.angular,
      aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
    );
  }

  const M = mod.eri_2idx_build(
    nAux,
    aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );

  // Pivoted Cholesky of M → effective rank r ≤ n_aux.
  const { L, pivots } = pivotedCholesky(M, nAux, choleskyThreshold);

  // B-tensor back-substitution. Parallel TS workers when SAB available;
  // 4-6× win on n=190 because the outer (μ,ν) loop is fully parallel.
  // WASM port turned out SLOWER (91 s vs 40 s single-thread on
  // naphthalene) — see formBFromCholesky JSDoc above for full diagnosis.
  let B: Float64Array;
  if (useSAB) {
    B = await formBFromCholeskyParallel(V, L, pivots, n, nAux, N);
  } else {
    B = formBFromCholesky(V, L, pivots, n, nAux).B;
  }

  // Free the V tensor SAB before returning. On large molecules
  // (anthracene n=274: V is 900 MB), keeping V alive through SCF
  // pushes peak resident SAB past Chrome's ~2 GB tab ceiling and
  // crashes the renderer. B is fully formed at this point — V is
  // dead weight. Reassigning to a zero-length view drops the
  // SharedArrayBuffer reference; the next GC cycle reclaims it.
  V = new Float64Array(0);
  return { B, nAux: pivots.length, threshold: choleskyThreshold, n };
}

/** Parallel variant of `buildAuxBasisDF` — the 3-index V tensor build
 *  is partitioned across N workers via the shared worker pool. The
 *  2-index M build runs on the main thread (cheap, ~50 ms at n_aux=400).
 *
 *  Requires SharedArrayBuffer (COOP/COEP isolation). Falls back to the
 *  single-thread `buildAuxBasisDF` if SAB unavailable. */
export async function buildAuxBasisDFParallel(
  orbitalShells: readonly CGShell[],
  auxShells?: readonly CGShell[],
  poolSize = 0,
  metricRegularization = 1e-10,
): Promise<DFResult> {
  if (!sabAvailable()) {
    return buildAuxBasisDF(orbitalShells, auxShells, metricRegularization);
  }
  const N = poolSize > 0 ? poolSize : (navigator.hardwareConcurrency ?? 4) - 1;
  const mod = await loadWasm();
  const orb = packShells(orbitalShells);
  const aux = auxShells ? packShells(auxShells) : orb;
  const n = orbitalShells.length;
  const nAux = auxShells ? auxShells.length : n;

  // ── 3-index V tensor via worker pool ──
  const vSAB = new SharedArrayBuffer(n * n * nAux * 8);
  const muAssignments: number[][] = Array.from({ length: N }, () => []);
  for (let mu = 0; mu < n; mu++) muAssignments[mu % N]!.push(mu);

  const workers = getSharedWorkerPool("wasm", N);
  await Promise.all(workers.map((w, i) => new Promise<void>((resolve, reject) => {
    const onMessage = (ev: MessageEvent): void => {
      w.removeEventListener("message", onMessage);
      if (ev.data?.ok) resolve();
      else reject(new Error(ev.data?.error ?? "worker failed"));
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      kind: "eri-3idx-wasm-slice",
      mus: muAssignments[i]!,
      muStart: 0, muEnd: n,
      nOrbital: n, nAux,
      nPrimsOrb: orb.nPrims, primOffsetsOrb: orb.primOff,
      alphaOrb: orb.alpha, cOrb: orb.c,
      centerOrb: orb.center, angularOrb: orb.angular,
      nPrimsAux: aux.nPrims, primOffsetsAux: aux.primOff,
      alphaAux: aux.alpha, cAux: aux.c,
      centerAux: aux.center, angularAux: aux.angular,
      v: vSAB,
    });
  })));
  const V = new Float64Array(n * n * nAux);
  V.set(new Float64Array(vSAB));

  // ── 2-index M (single-thread, cheap) ──
  const M = mod.eri_2idx_build(
    nAux,
    aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );

  // ── Eigendecomp + B formation (same as single-thread path) ──
  const eig = eigsymmetric(M, nAux);
  const invSqrtLam = new Float64Array(nAux);
  let nKept = 0;
  for (let i = 0; i < nAux; i++) {
    const lam = eig.values[i]!;
    if (lam > metricRegularization) {
      invSqrtLam[i] = 1.0 / Math.sqrt(lam);
      nKept++;
    } else {
      invSqrtLam[i] = 0.0;
    }
  }
  // Full B = V · U · diag(λ⁻¹⸍²) · Uᵀ matmul via Rust+WASM with
  // f64x2 SIMD on the inner P-loop. ~5× over the TS version at
  // benzene n_aux=400.
  const B = mod.form_b_tensor(n, nAux, V, eig.vectors, invSqrtLam);
  return { B, nAux: nKept, threshold: metricRegularization, n };
}
