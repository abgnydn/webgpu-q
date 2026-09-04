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
import { sabAvailable } from "../parallel/worker-pool.js";
import { getSharedWorkerPool } from "../parallel/worker-pool-shared.js";
import { formBFromCholeskyParallel } from "../parallel/parallel-form-b-cholesky.js";
import { pivotedCholesky, formBFromCholesky } from "./df-aux-cholesky.js";

export {
  buildAuxBasisDF,
  pivotedCholesky,
  formBFromCholesky,
} from "./df-aux-cholesky.js";
export { buildAuxBasisDFParallel } from "./df-aux-parallel.js";
export {
  buildAuxBasisDFStreaming,
  buildAuxBasisDFStreamingCooperative,
  type StreamingDFResult,
  type CooperativeDFResult,
} from "./df-aux-streaming.js";
export {
  buildBFromV,
  buildBFromVBlocks,
  buildMetric2idxCPU,
  buildV3idxCPU,
} from "./df-aux-hybrid.js";

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

export interface WasmEriModule {
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
export async function loadWasm(): Promise<WasmEriModule> {
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

export function packShells(shells: readonly CGShell[]): {
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
