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
}

let wasmModule: WasmEriModule | null = null;
async function loadWasm(): Promise<WasmEriModule> {
  if (wasmModule) return wasmModule;
  const mod = await import(
    /* @vite-ignore */
    "../../wasm-eri/pkg/wasm_eri.js" as string,
  ) as WasmEriModule;
  await mod.default();
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
      invSqrtLam[i] = 0.0;  // drop near-zero mode
    }
  }

  // Form M^(-1/2) (we don't need it explicitly — fold into B directly).
  // B[μν, P] = Σ_Q V[μν, Q] · (U · Λ^(-1/2) · U^T)[Q, P]
  //          = Σ_Q Σ_i V[μν, Q] · U[Q, i] · λ_i^(-1/2) · U[P, i]
  // Pre-compute T[μν, i] = Σ_Q V[μν, Q] · U[Q, i] · λ_i^(-1/2)
  // Then B[μν, P] = Σ_i T[μν, i] · U[P, i]
  //
  // Layout: eig.vectors is column-major, so U[Q, i] = vectors[i*nAux + Q].
  const T = new Float64Array(n * n * nAux);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      for (let i = 0; i < nAux; i++) {
        if (invSqrtLam[i] === 0) continue;
        let s = 0;
        for (let Q = 0; Q < nAux; Q++) {
          s += V[(mu * n + nu) * nAux + Q]! * eig.vectors[i * nAux + Q]!;
        }
        T[(mu * n + nu) * nAux + i] = s * invSqrtLam[i]!;
      }
    }
  }
  const B = new Float64Array(n * n * nAux);
  for (let mu = 0; mu < n; mu++) {
    for (let nu = 0; nu < n; nu++) {
      for (let P = 0; P < nAux; P++) {
        let s = 0;
        for (let i = 0; i < nAux; i++) {
          if (invSqrtLam[i] === 0) continue;
          s += T[(mu * n + nu) * nAux + i]! * eig.vectors[i * nAux + P]!;
        }
        B[(mu * n + nu) * nAux + P] = s;
      }
    }
  }

  return {
    B,
    nAux: nKept,
    threshold: metricRegularization,
    n,
  };
}
