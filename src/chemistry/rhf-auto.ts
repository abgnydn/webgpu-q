// runRHFAuto — the "do the right thing" closed-shell RHF entry point.
//
// What a computational chemist actually wants from one call: exact when it's
// affordable, density-fitting when the exact 4-index ERI won't fit, and full
// transparency about which path ran. This wires together the pieces that already
// exist (exact ERI SCF, WASM aux-basis DF, the hybrid GPU/WASM integral build)
// behind a single size-gated decision, and REPORTS the method/engine/precision so
// the number is never unattributed.
//
// Decision (researcher defaults):
//   • n ≤ exactMaxN            → EXACT 4-index ERI (gold standard, f64).
//   • n >  exactMaxN           → DENSITY FITTING (the exact ERI is O(n⁴) and
//                                won't fit in a tab — DF is the enabling path).
//        - default engine      = WASM, f64 (standard RI-JK, well-characterized).
//        - opts.fast + d-regime= HYBRID: GPU f32 builds the s/p/d-aux columns,
//                                WASM f64 the f-aux, f64 JK — GPU-accelerated AND
//                                chemistry-grade (~0.2 mHa, the f32 block costs
//                                ~8 µHa). Replaces the old d-only level-0 GPU path
//                                that was ~30 mHa screening-only.
//
// exactMaxN default 80: the ERI is n⁴·8 bytes, ~256 MB at n=75 — past that a
// browser tab can't hold it (naphthalene cc-pVDZ n≈180 → 10 GB).

import { type CGShell } from "./integrals-cg.js";
import { computeMolecularIntegrals, type Nucleus, type MolecularIntegrals } from "./cg-molecular.js";
import { runRHFSCFAsync, type HFResult, type HFOpts } from "./hf-scf.js";
import { generateAutoAux, buildAuxBasisDFStreaming, buildBFromV } from "./df-aux.js";
import { buildV3idxHybrid } from "./df-gpu.js";
import { type DFResult } from "./df.js";
import { runRKSDFT, type RKSResult, type RKSOpts } from "./dft/rks-scf.js";
import { type FunctionalKind } from "./dft/functional.js";
import { type AtomSymbol } from "./atoms.js";

export interface RHFAutoOpts {
  /** Largest n that still uses the exact 4-index ERI. Above it → DF. Default 80. */
  readonly exactMaxN?: number;
  /** In the DF regime, use the hybrid GPU/WASM integral build (GPU-accelerated,
   *  still chemistry-grade) instead of pure WASM. Default false (pure f64 WASM). */
  readonly fast?: boolean;
  /** Force a path regardless of size: "exact" | "df". Default: auto by size. */
  readonly force?: "exact" | "df";
  /** SCF controls passed through (tolerances, maxIter, DIIS, parallel). */
  readonly hf?: HFOpts;
}

/** How the energy was actually produced — record this in any artifact. */
export interface RHFAutoProvenance {
  readonly method: "exact-eri" | "density-fitting";
  readonly engine: "wasm" | "gpu" | "gpu+wasm";
  readonly precision: "f64" | "f32" | "mixed";
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
  const { df, provenance } = await buildDFForRegime(shells, !!opts.fast);
  const hf = await runRHFSCFAsync(integrals, nElectrons, { ...hfOpts, useDF: df });
  return { hf, integrals, provenance };
}

/** Build the DF tensor for the large-system regime, shared by HF and DFT. fast →
 *  the hybrid GPU/WASM build (GPU-accelerated, chemistry-grade); else pure f64
 *  WASM RI-JK. Returns the tensor and the provenance of how it was built. */
async function buildDFForRegime(
  shells: readonly CGShell[], fast: boolean,
): Promise<{ df: DFResult; provenance: RHFAutoProvenance }> {
  // extraL=1 aux (the cc-pVDZ sweet spot — adds f-aux for chemistry-grade
  // accuracy; extraL=2 over-completes and breaks the metric orthogonalization).
  const aux = generateAutoAux(shells, 1);
  const n = shells.length;

  // The hybrid path projects via buildBFromV, which MATERIALIZES the full f64 V
  // tensor (n²·nAux·8 B) on top of the GPU f32 V and the B tensor. That's fine
  // for medium molecules but blows memory at PAH scale (naphthalene n=190:
  // V alone is 312 MB → on a tab with <1 GB free it thrashes and runs ~2×
  // SLOWER than streaming). The streaming WASM path never materializes full V,
  // so above this size we use it even when fast was requested. 200 MB full-V cap.
  const fullVbytes = n * n * aux.length * 8;
  const hybridFits = fullVbytes < 200 * 1024 * 1024;
  const wantGPU = fast && hybridFits && hasDFunctions(shells) &&
    !!(navigator as unknown as { gpu?: unknown }).gpu;

  if (wantGPU) {
    // GPU-accelerated AND chemically accurate: the HYBRID 3-index build does the
    // s/p/d-aux columns on the GPU (f32) and the f-aux columns on WASM (f64),
    // then projects in f64. Measured: the f32 low-aux block costs only ~8 µHa, so
    // the result stays chemistry-grade (H2O 0.19 mHa vs exact) — unlike the
    // d-only level-0 GPU path (~30 mHa) it replaces.
    try {
      const { V, auxOrdered } = await buildV3idxHybrid(shells, aux);
      const df = await buildBFromV(shells, auxOrdered, V);
      return {
        df,
        provenance: { method: "density-fitting", engine: "gpu+wasm", precision: "mixed", nAux: df.nAux,
          expectedError: "hybrid extraL=1: GPU f32 s/p/d-aux + f64 f-aux + f64 J/K — chemistry-grade (~0.2 mHa vs exact)" },
      };
    } catch {
      /* GPU/hybrid path unavailable — fall through to pure f64 WASM DF */
    }
  }

  // Default DF: f64 WASM aux-basis (standard RI-JK); B reused across the SCF.
  const df = await buildAuxBasisDFStreaming(shells, aux);
  return {
    df,
    provenance: { method: "density-fitting", engine: "wasm", precision: "f64", nAux: df.nAux,
      expectedError: "DF extraL=1 aux-basis vs exact (few mHa); f64 throughout" },
  };
}

export interface RKSAutoOpts {
  /** Largest n that still uses the exact 4-index ERI. Above it → DF. Default 80. */
  readonly exactMaxN?: number;
  /** Use the hybrid GPU/WASM DF build (GPU-accelerated, chemistry-grade) in the
   *  DF regime. Pure functionals need only the cheap DF J. Default false. */
  readonly fast?: boolean;
  /** Force a path regardless of size: "exact" | "df". Default: auto by size. */
  readonly force?: "exact" | "df";
  /** XC functional. Default "lda-svwn". */
  readonly functional?: FunctionalKind;
  /** RKS SCF controls passed through. */
  readonly rks?: RKSOpts;
}

export interface RKSAutoResult {
  readonly rks: RKSResult;
  readonly integrals: MolecularIntegrals;
  readonly provenance: RHFAutoProvenance;
}

/** Closed-shell Kohn-Sham DFT with automatic exact/DF selection by system size.
 *  The DF path makes large-molecule DFT feasible in a tab (the 4-index ERI is
 *  O(n⁴)); pure functionals ride the cheap DF J, hybrids also use the DF K. The
 *  XC term is the numerical grid either way. Returns the result, integrals, and
 *  provenance of the number. */
export async function runRKSAuto(
  shells: readonly CGShell[],
  nuclei: readonly Nucleus[],
  nElectrons: number,
  symbols: readonly AtomSymbol[],
  opts: RKSAutoOpts = {},
): Promise<RKSAutoResult> {
  const n = shells.length;
  const exactMaxN = opts.exactMaxN ?? 80;
  const useDF = opts.force === "df" || (opts.force !== "exact" && n > exactMaxN);
  const rksOpts: RKSOpts = { functional: opts.functional, ...opts.rks };

  if (!useDF) {
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const rks = runRKSDFT(integrals, nElectrons, symbols, rksOpts);
    return {
      rks, integrals,
      provenance: { method: "exact-eri", engine: "wasm", precision: "f64", nAux: 0,
        expectedError: "exact ERI J/K (XC on grid); no integral approximation" },
    };
  }

  const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true });
  const { df, provenance } = await buildDFForRegime(shells, !!opts.fast);
  const rks = runRKSDFT(integrals, nElectrons, symbols, { ...rksOpts, useDF: df });
  return { rks, integrals, provenance };
}
