// runRHFAuto — the "do the right thing" closed-shell RHF entry point.
//
// What a computational chemist actually wants from one call: exact when it's
// affordable, density-fitting when the exact 4-index ERI won't fit, and full
// transparency about which path ran. This wires together the pieces that already
// exist (exact ERI SCF, WASM aux-basis DF, the WGSL GPU DF build + GPU DF-JK)
// behind a single size-gated decision, and REPORTS the method/engine/precision so
// the number is never unattributed.
//
// Decision (researcher defaults):
//   • n ≤ exactMaxN            → EXACT 4-index ERI (gold standard, f64).
//   • n >  exactMaxN           → DENSITY FITTING (the exact ERI is O(n⁴) and
//                                won't fit in a tab — DF is the enabling path).
//        - default engine      = WASM, f64  (standard RI-JK, well-characterized).
//        - opts.fast + d-regime= GPU, f32   (the WGSL integral build + GPU DF-JK;
//                                ~6e-4 element-precision floor, disclosed).
//
// exactMaxN default 80: the ERI is n⁴·8 bytes, ~256 MB at n=75 — past that a
// browser tab can't hold it (naphthalene cc-pVDZ n≈180 → 10 GB).

import { type CGShell } from "./integrals-cg.js";
import { computeMolecularIntegrals, type Nucleus, type MolecularIntegrals } from "./cg-molecular.js";
import { runRHFSCFAsync, type HFResult, type HFOpts } from "./hf-scf.js";
import { generateAutoAux, buildAuxBasisDFStreaming } from "./df-aux.js";
import { buildDFAuto } from "./df-gpu.js";
import { makeGpuDFJK } from "./jk-df-gpu.js";

export interface RHFAutoOpts {
  /** Largest n that still uses the exact 4-index ERI. Above it → DF. Default 80. */
  readonly exactMaxN?: number;
  /** In the DF regime, prefer the GPU f32 path (faster, ~6e-4 floor) over WASM f64. */
  readonly fast?: boolean;
  /** Force a path regardless of size: "exact" | "df". Default: auto by size. */
  readonly force?: "exact" | "df";
  /** SCF controls passed through (tolerances, maxIter, DIIS, parallel). */
  readonly hf?: HFOpts;
}

/** How the energy was actually produced — record this in any artifact. */
export interface RHFAutoProvenance {
  readonly method: "exact-eri" | "density-fitting";
  readonly engine: "wasm" | "gpu";
  readonly precision: "f64" | "f32";
  /** Number of auxiliary functions (DF only; 0 for exact). */
  readonly nAux: number;
  /** Expected method error vs the exact result, order of magnitude, human-readable. */
  readonly expectedError: string;
}

export interface RHFAutoResult {
  readonly hf: HFResult;
  readonly integrals: MolecularIntegrals;
  readonly provenance: RHFAutoProvenance;
}

function hasDFunctions(shells: readonly CGShell[]): boolean {
  for (const s of shells) if (s.angular[0] + s.angular[1] + s.angular[2] >= 2) return true;
  return false;
}

/** Closed-shell RHF with automatic exact/DF selection by system size, returning
 *  the converged result, the integrals, and the provenance of the number. */
export async function runRHFAuto(
  shells: readonly CGShell[],
  nuclei: readonly Nucleus[],
  nElectrons: number,
  opts: RHFAutoOpts = {},
): Promise<RHFAutoResult> {
  const n = shells.length;
  const exactMaxN = opts.exactMaxN ?? 80;
  const useDF = opts.force === "df" || (opts.force !== "exact" && n > exactMaxN);
  const hfOpts: HFOpts = {
    useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
    ...opts.hf,
  };

  if (!useDF) {
    // Gold standard: build the full ERI and run the exact SCF.
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = await runRHFSCFAsync(integrals, nElectrons, hfOpts);
    return {
      hf, integrals,
      provenance: { method: "exact-eri", engine: "wasm", precision: "f64", nAux: 0,
        expectedError: "exact (no integral approximation)" },
    };
  }

  // DF regime: skip the O(n⁴) ERI entirely — DF never needs it.
  const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true });

  const wantGPU = !!opts.fast && hasDFunctions(shells) &&
    !!(navigator as unknown as { gpu?: unknown }).gpu;

  if (wantGPU) {
    // GPU f32: WGSL integral build for B + GPU DF-JK every SCF iteration. The
    // f32 JK floor caps convergence near ~6e-4, so loosen tolerances to match.
    // NOTE the accuracy ceiling: the WGSL 3-index kernel only handles up to d
    // (maxL ≤ 2), so the GPU path is limited to the extraL=0 aux basis (no
    // f-aux). On cc-pVDZ that level-0 aux is itself ~30 mHa from exact — so the
    // GPU fast path trades chemistry-grade accuracy for speed/feasibility. Use
    // it for large-system screening, not for chemical-accuracy numbers.
    const auxGpu = generateAutoAux(shells, 0);
    try {
      const { df, path } = await buildDFAuto(shells, auxGpu, { gpu: true });
      if (path === "gpu") {
        const gpuJK = await makeGpuDFJK(df);
        const customJKBuilder = async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
          const { J, K } = await gpuJK.jk(D);
          const Jf = new Float64Array(J.length), Kf = new Float64Array(K.length);
          for (let i = 0; i < J.length; i++) { Jf[i] = J[i]!; Kf[i] = K[i]!; }
          return { J: Jf, K: Kf };
        };
        try {
          const hf = await runRHFSCFAsync(integrals, nElectrons, {
            ...hfOpts, parallel: 1, customJKBuilder,
            energyTol: Math.max(hfOpts.energyTol ?? 0, 1e-5),
            densityTol: Math.max(hfOpts.densityTol ?? 0, 1e-4),
          });
          return {
            hf, integrals,
            provenance: { method: "density-fitting", engine: "gpu", precision: "f32", nAux: df.nAux,
              expectedError: "level-0 aux (d-only kernel, ~30 mHa vs exact) + f32 JK floor (~6e-4 Ha) — screening, NOT chemical accuracy" },
          };
        } finally { gpuJK.dispose(); }
      }
    } catch {
      /* GPU path unavailable / failed — fall through to f64 WASM DF */
    }
  }

  // Default DF: f64 WASM aux-basis with extraL=1 (the cc-pVDZ sweet spot — adds
  // f-aux for chemistry-grade accuracy; extraL=2 over-completes and breaks the
  // metric orthogonalization). Standard RI-JK; B reused across the SCF.
  const aux = generateAutoAux(shells, 1);
  const df = await buildAuxBasisDFStreaming(shells, aux);
  const hf = await runRHFSCFAsync(integrals, nElectrons, { ...hfOpts, useDF: df });
  return {
    hf, integrals,
    provenance: { method: "density-fitting", engine: "wasm", precision: "f64", nAux: df.nAux,
      expectedError: "DF extraL=1 aux-basis vs exact (few mHa); f64 throughout" },
  };
}
