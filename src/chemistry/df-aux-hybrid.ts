import type { CGShell } from "./integrals-cg.js";
import type { DFResult } from "./df.js";
import { eigsymmetric } from "../manybody/dense-eig.js";
import { DF_CHOLESKY_TOL } from "./numerical-tolerances.js";
import { loadWasm, packShells } from "./df-aux.js";

/** CPU (f64 WASM) 2-index DF metric M[P,Q] = (P|Q), row-major n_aux × n_aux.
 *  The reference the GPU f32 metric kernel (df-gpu.ts) is validated against. */
export async function buildMetric2idxCPU(auxShells: readonly CGShell[]): Promise<Float64Array> {
  const mod = await loadWasm();
  const aux = packShells(auxShells);
  return mod.eri_2idx_build(
    auxShells.length, aux.nPrims, aux.primOff, aux.alpha, aux.c, aux.center, aux.angular,
  );
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

/** Build the mode-basis DF tensor from a PRECOMPUTED 3-index V[μν,Q] (e.g. one
 *  the GPU produced). Returns the same mode-basis B as buildAuxBasisDFStreaming
 *  (nKept columns), so it drops straight into buildJK_DF / runRHFSCF useDF. Used
 *  to run HF end-to-end on GPU-computed integrals. */
export async function buildBFromV(
  orbShells: readonly CGShell[], auxShells: readonly CGShell[],
  V: Float64Array, metricRegularization = DF_CHOLESKY_TOL,
): Promise<DFResult> {
  const n = orbShells.length;
  const nAux = auxShells.length;
  // The full V is already in hand → each block is a zero-copy subarray.
  return buildBFromVBlocks(
    orbShells, auxShells,
    (mu0, mu1) => V.subarray(mu0 * n * nAux, mu1 * n * nAux),
    metricRegularization,
  );
}

/** Like buildBFromV, but pulls each μ-block of the 3-index V on demand from a
 *  callback instead of holding the full tensor. This is what lets a large-molecule
 *  GPU/WASM hybrid build avoid materializing the full f64 V (312 MB at n=190,
 *  which thrashes a tab): the caller assembles one block at a time. getBlock(mu0,
 *  mu1) must return the f64 V rows [mu0,mu1) in layout ((μ−mu0)·n + ν)·nAux + P,
 *  the same contiguous shape buildBFromV's subarray had. */
export async function buildBFromVBlocks(
  orbShells: readonly CGShell[], auxShells: readonly CGShell[],
  getBlock: (mu0: number, mu1: number) => Float64Array,
  metricRegularization = DF_CHOLESKY_TOL,
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
    // V rows [mu0,mu1): provided by the caller. The kernel reads ν ≥ μ.
    const vblk = getBlock(mu0, mu1);
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
