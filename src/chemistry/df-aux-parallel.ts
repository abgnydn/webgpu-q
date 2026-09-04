import type { CGShell } from "./integrals-cg.js";
import type { DFResult } from "./df.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { sabAvailable } from "../parallel/worker-pool.js";
import { getSharedWorkerPool } from "../parallel/worker-pool-shared.js";
import { buildAuxBasisDF } from "./df-aux-cholesky.js";
import { loadWasm, packShells } from "./df-aux.js";

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
  // Full nAux stride, not nKept — same contract as buildAuxBasisDF above.
  const B = mod.form_b_tensor(n, nAux, V, eig.vectors, invSqrtLam);
  return { B, nAux, nKeptModes: nKept, threshold: metricRegularization, n };
}
