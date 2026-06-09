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
import { runUHFSCF, type UHFResult, type UHFOpts } from "./uhf-scf.js";
import { generateAutoAux, buildAuxBasisDFStreaming } from "./df-aux.js";
import { buildHybridDFStreaming } from "./df-gpu.js";
import { type DFResult } from "./df.js";
import { runRKSDFT, type RKSResult, type RKSOpts } from "./dft/rks-scf.js";
import { runUKSDFT, type UKSResult, type UKSOpts } from "./dft/uks-scf.js";
import { runMP2, mp2EnergyDF } from "./mp2.js";
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

  // Gate the hybrid to the scale where it actually wins. Its limiter at large n
  // is NOT memory (streaming fixed that) but the per-block merge: assembling each
  // f64 V block from the GPU f32 low-aux + WASM f64 f-aux is a JS triple-loop,
  // O(n²·nAux), which the all-WASM streaming path does in WASM SIMD. MEASURED:
  // naphthalene (n²·nAux ≈ 39 M) ran the hybrid >2× SLOWER than WASM streaming;
  // benzene (≈ 9.7 M) is fine. So gate on the merge cost; above it the all-WASM
  // streaming DF wins even with fast=true. (A WASM merge kernel would lift this —
  // the real lever for large-molecule GPU DF.)
  const mergeOps = n * n * aux.length;
  const hybridWins = mergeOps < 12_000_000;
  const wantGPU = fast && hybridWins && hasDFunctions(shells) &&
    !!(navigator as unknown as { gpu?: unknown }).gpu;

  if (wantGPU) {
    // GPU-accelerated AND chemically accurate: GPU f32 builds the s/p/d-aux
    // columns, WASM f64 the f-aux, f64 projection + J/K. The f32 low-aux block
    // costs only ~8 µHa, so the result stays chemistry-grade (H2O 0.19 mHa vs
    // exact). Streaming keeps peak memory low enough to win at large n too.
    try {
      const { df } = await buildHybridDFStreaming(shells, aux);
      return {
        df,
        provenance: { method: "density-fitting", engine: "gpu+wasm", precision: "mixed", nAux: df.nAux,
          expectedError: "hybrid extraL=1 (streaming): GPU f32 s/p/d-aux + f64 f-aux + f64 J/K — chemistry-grade (~0.2 mHa vs exact)" },
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

export interface UHFAutoOpts {
  /** Largest n that still uses the exact 4-index ERI. Above it → DF. Default 80. */
  readonly exactMaxN?: number;
  /** Use the hybrid GPU/WASM DF build in the DF regime. Default false. */
  readonly fast?: boolean;
  /** Force a path regardless of size: "exact" | "df". Default: auto by size. */
  readonly force?: "exact" | "df";
  /** UHF SCF controls passed through. */
  readonly uhf?: UHFOpts;
}

export interface UHFAutoResult {
  readonly uhf: UHFResult;
  readonly integrals: MolecularIntegrals;
  readonly provenance: RHFAutoProvenance;
}

/** Open-shell UHF with automatic exact/DF selection by system size — radicals,
 *  doublets, O₂, etc. The DF path makes large open-shell systems feasible in a
 *  tab (J from the total density, K from each spin density, off the DF B-tensor).
 *  Closed-shell (nAlpha = nBeta) collapses to RHF. Returns result + provenance. */
export async function runUHFAuto(
  shells: readonly CGShell[],
  nuclei: readonly Nucleus[],
  nAlpha: number,
  nBeta: number,
  opts: UHFAutoOpts = {},
): Promise<UHFAutoResult> {
  const n = shells.length;
  const exactMaxN = opts.exactMaxN ?? 80;
  const useDF = opts.force === "df" || (opts.force !== "exact" && n > exactMaxN);
  const uhfOpts: UHFOpts = { useDIIS: true, energyTol: 1e-9, densityTol: 1e-7, maxIter: 200, ...opts.uhf };

  if (!useDF) {
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uhf = runUHFSCF(integrals, nAlpha, nBeta, uhfOpts);
    return {
      uhf, integrals,
      provenance: { method: "exact-eri", engine: "wasm", precision: "f64", nAux: 0,
        expectedError: "exact (no integral approximation)" },
    };
  }

  const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true });
  const { df, provenance } = await buildDFForRegime(shells, !!opts.fast);
  const uhf = runUHFSCF(integrals, nAlpha, nBeta, { ...uhfOpts, useDF: df });
  return { uhf, integrals, provenance };
}

export interface UKSAutoOpts {
  /** Largest n that still uses the exact 4-index ERI. Above it → DF. Default 80. */
  readonly exactMaxN?: number;
  /** Use the hybrid GPU/WASM DF build in the DF regime. Default false. */
  readonly fast?: boolean;
  /** Force a path regardless of size: "exact" | "df". Default: auto by size. */
  readonly force?: "exact" | "df";
  /** XC functional. Default "lda-svwn". */
  readonly functional?: FunctionalKind;
  /** UKS SCF controls passed through. */
  readonly uks?: UKSOpts;
}

export interface UKSAutoResult {
  readonly uks: UKSResult;
  readonly integrals: MolecularIntegrals;
  readonly provenance: RHFAutoProvenance;
}

/** Open-shell Kohn-Sham DFT with size-gated exact/DF — radical DFT (UKS LDA/GGA/
 *  hybrid). Pure functionals ride the cheap DF J; hybrids also use per-spin DF K.
 *  XC stays on the grid. Closed-shell collapses to RKS. Returns result + provenance. */
export async function runUKSAuto(
  shells: readonly CGShell[],
  nuclei: readonly Nucleus[],
  nAlpha: number,
  nBeta: number,
  symbols: readonly AtomSymbol[],
  opts: UKSAutoOpts = {},
): Promise<UKSAutoResult> {
  const n = shells.length;
  const exactMaxN = opts.exactMaxN ?? 80;
  const useDF = opts.force === "df" || (opts.force !== "exact" && n > exactMaxN);
  const uksOpts: UKSOpts = { functional: opts.functional, ...opts.uks };

  if (!useDF) {
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const uks = runUKSDFT(integrals, nAlpha, nBeta, symbols, uksOpts);
    return {
      uks, integrals,
      provenance: { method: "exact-eri", engine: "wasm", precision: "f64", nAux: 0,
        expectedError: "exact ERI J/K (XC on grid); no integral approximation" },
    };
  }

  const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true });
  const { df, provenance } = await buildDFForRegime(shells, !!opts.fast);
  const uks = runUKSDFT(integrals, nAlpha, nBeta, symbols, { ...uksOpts, useDF: df });
  return { uks, integrals, provenance };
}

export interface MP2AutoOpts {
  /** Largest n that still uses the exact 4-index ERI. Above it → DF. Default 80. */
  readonly exactMaxN?: number;
  /** Use the hybrid GPU/WASM DF build in the DF regime. Default false. */
  readonly fast?: boolean;
  /** Force a path regardless of size: "exact" | "df". Default: auto by size. */
  readonly force?: "exact" | "df";
  /** Frozen-core spatial orbitals excluded from the correlation sum. Default 0. */
  readonly nFrozenCore?: number;
  /** RHF SCF controls passed through. */
  readonly hf?: HFOpts;
}

export interface MP2AutoResult {
  readonly hfEnergy: number;
  readonly correlationEnergy: number;
  readonly totalEnergy: number;
  readonly nOccupied: number;
  readonly integrals: MolecularIntegrals;
  readonly provenance: RHFAutoProvenance;
}

/** Closed-shell RHF-MP2 with automatic exact/DF selection by system size. The DF
 *  path is the standard way to get MP2 on large molecules: it builds the HF
 *  reference AND the MP2 (ia|jb) integrals from the SAME DF B-tensor, so the
 *  O(n⁴) ERI is never materialized — correlation energy in a tab for systems
 *  where conventional MP2 can't allocate. Small systems use the exact 4-index
 *  path. Returns the HF + correlation + total energy with provenance. */
export async function runMP2Auto(
  shells: readonly CGShell[],
  nuclei: readonly Nucleus[],
  nElectrons: number,
  opts: MP2AutoOpts = {},
): Promise<MP2AutoResult> {
  const n = shells.length;
  const exactMaxN = opts.exactMaxN ?? 80;
  const useDF = opts.force === "df" || (opts.force !== "exact" && n > exactMaxN);
  const nFrozen = opts.nFrozenCore ?? 0;
  const hfOpts: HFOpts = { useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200, ...opts.hf };

  if (!useDF) {
    // Exact: full ERI HF reference, conventional 4-index MP2.
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = await runRHFSCFAsync(integrals, nElectrons, hfOpts);
    const mp2 = runMP2(hf, integrals, { nFrozenCore: nFrozen });
    return {
      hfEnergy: hf.energy, correlationEnergy: mp2.correlationEnergy, totalEnergy: mp2.totalEnergy,
      nOccupied: hf.nOccupied, integrals,
      provenance: { method: "exact-eri", engine: "wasm", precision: "f64", nAux: 0,
        expectedError: "exact MP2 (no integral approximation)" },
    };
  }

  // DF: one B-tensor serves BOTH the HF reference and the MP2 (ia|jb) — no ERI.
  const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true });
  const { df, provenance } = await buildDFForRegime(shells, !!opts.fast);
  const hf = await runRHFSCFAsync(integrals, nElectrons, { ...hfOpts, useDF: df });
  const eCorr = mp2EnergyDF(hf.C_MO, hf.orbitalEnergies, hf.nOccupied, integrals.n, nFrozen, df);
  return {
    hfEnergy: hf.energy, correlationEnergy: eCorr, totalEnergy: hf.energy + eCorr,
    nOccupied: hf.nOccupied, integrals,
    provenance: { ...provenance, expectedError: `DF-MP2: ${provenance.expectedError}` },
  };
}
