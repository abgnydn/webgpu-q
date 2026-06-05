// Streaming, mode-partitioned aux-DF build (capability: browser-tab swarm that
// never materializes the full 3-index tensor on any tab).
//
// Increment 1 — numerics only, single process. Proves:
//   1. The streaming mode-basis B reproduces the reference eigendecomposition
//      DF (buildAuxBasisDF) J and K on a real density to ≤ 1e-9 Ha.
//   2. Partitioning the mode axis across "tabs" and summing partial (J,K)
//      reproduces the single-tab build exactly (the swarm-partition identity,
//      now at BUILD time, not just JK time).
//   3. The streaming path's peak V footprint is muBlock·n·n_aux, NOT n²·n_aux —
//      the evidence the full tensor is never resident.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF, runRHFSCFAsync } from "../../src/chemistry/hf-scf.js";
import { buildJK_DF } from "../../src/chemistry/df.js";
import type { DFResult } from "../../src/chemistry/df.js";
import {
  buildAuxBasisDF,
  buildAuxBasisDFStreaming,
  buildAuxBasisDFStreamingCooperative,
  generateAutoAux,
} from "../../src/chemistry/df-aux.js";

const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

function maxAbsDiff(A: Float64Array, B: Float64Array): number {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    const d = Math.abs(A[i]! - B[i]!);
    if (d > m) m = d;
  }
  return m;
}

describe("streaming mode-partitioned aux-DF build", () => {
  test("H₂O STO-3G — streaming B reproduces reference DF J/K to ≤ 1e-9", async () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const n = shells.length;
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    const D = hf.D; // converged RHF density matrix, the real test load

    const aux = generateAutoAux(shells, 1);
    const ref = await buildAuxBasisDF(shells, aux);
    const stream = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { muBlock: 2 });

    const jkRef = buildJK_DF(ref, D);
    const jkStream = buildJK_DF(stream, D);

    const dJ = maxAbsDiff(jkRef.J, jkStream.J);
    const dK = maxAbsDiff(jkRef.K, jkStream.K);
    console.log(`[df-streaming] max|ΔJ| = ${dJ.toExponential(2)}, max|ΔK| = ${dK.toExponential(2)}`);
    console.log(`[df-streaming] ref nAux=${ref.nAux}, stream modes=${stream.nAux}, peakVFloats=${stream.peakVFloats} (full=${n * n * aux.length})`);

    expect(dJ).toBeLessThan(1e-9);
    expect(dK).toBeLessThan(1e-9);
  });

  test("mode-partition across 'tabs' sums to the single-tab build", async () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const n = shells.length;
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    const D = hf.D; // converged RHF density matrix, the real test load

    const aux = generateAutoAux(shells, 1);
    const full = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { muBlock: 4 });
    const nKept = full.nAux;

    // Split the mode axis across 3 "tabs"; each builds only its slice.
    const cuts = [0, Math.floor(nKept / 3), Math.floor((2 * nKept) / 3), nKept];
    const J = new Float64Array(n * n);
    const K = new Float64Array(n * n);
    let maxSliceVFloats = 0;
    for (let t = 0; t < 3; t++) {
      const slice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, {
        modeStart: cuts[t]!, modeEnd: cuts[t + 1]!, muBlock: 4,
      });
      maxSliceVFloats = Math.max(maxSliceVFloats, slice.peakVFloats);
      const part = buildJK_DF(slice, D);
      for (let i = 0; i < J.length; i++) { J[i] = J[i]! + part.J[i]!; K[i] = K[i]! + part.K[i]!; }
    }

    const ref = buildJK_DF(full, D);
    const dJ = maxAbsDiff(ref.J, J);
    const dK = maxAbsDiff(ref.K, K);
    console.log(`[df-streaming-swarm] 3-tab partition max|ΔJ| = ${dJ.toExponential(2)}, max|ΔK| = ${dK.toExponential(2)}`);
    console.log(`[df-streaming-swarm] per-tab peak V floats = ${maxSliceVFloats} vs full tensor ${n * n * aux.length}`);

    // Summed partial JK over disjoint mode ranges == single-tab JK, bit-close.
    expect(dJ).toBeLessThan(1e-12);
    expect(dK).toBeLessThan(1e-12);
    // The streaming footprint never reaches the full n²·n_aux tensor.
    expect(maxSliceVFloats).toBeLessThan(n * n * aux.length);
  });

  test("full DF-HF SCF from independently-built mode-slices == single-tab SCF", async () => {
    // Increment 2: a complete RHF SCF where the Fock build is driven entirely
    // by mode-slices each "tab" builds independently (no shared full tensor).
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const nElec = 10;

    const direct = runRHFSCF(integrals, nElec); // exact HF reference

    const aux = generateAutoAux(shells, 1);

    // A customJKBuilder that sums partial (J,K) from a set of DF slices —
    // exactly the swarm's per-iteration contract, here in one process.
    const jkFromSlices = (slices: DFResult[]) =>
      async (D: Float64Array): Promise<{ J: Float64Array; K: Float64Array }> => {
        const N = D.length;
        const J = new Float64Array(N);
        const K = new Float64Array(N);
        for (const s of slices) {
          const part = buildJK_DF(s, D);
          for (let i = 0; i < N; i++) { J[i] = J[i]! + part.J[i]!; K[i] = K[i]! + part.K[i]!; }
        }
        return { J, K };
      };

    // Single "tab": full mode range.
    const full = await buildAuxBasisDFStreaming(shells, aux, 1e-10);
    const single = await runRHFSCFAsync(integrals, nElec, { customJKBuilder: jkFromSlices([full]) });

    // Four "tabs": disjoint mode ranges, each built independently from scratch.
    const nKept = full.nAux;
    const T = 4;
    const slices: DFResult[] = [];
    for (let t = 0; t < T; t++) {
      const a = Math.floor((t * nKept) / T);
      const b = Math.floor(((t + 1) * nKept) / T);
      slices.push(await buildAuxBasisDFStreaming(shells, aux, 1e-10, { modeStart: a, modeEnd: b }));
    }
    const swarm = await runRHFSCFAsync(integrals, nElec, { customJKBuilder: jkFromSlices(slices) });

    const dPartition = Math.abs(swarm.energy - single.energy);
    const dfError = Math.abs(single.energy - direct.energy);
    console.log(`[df-streaming-scf] direct HF   = ${direct.energy.toFixed(10)} Ha`);
    console.log(`[df-streaming-scf] 1-tab DF-HF = ${single.energy.toFixed(10)} Ha (DF err ${dfError.toExponential(2)})`);
    console.log(`[df-streaming-scf] ${T}-tab DF-HF = ${swarm.energy.toFixed(10)} Ha (vs 1-tab Δ ${dPartition.toExponential(2)})`);
    expect(swarm.converged).toBe(true);
    // Capability claim: partitioning the build across tabs changes nothing.
    expect(dPartition).toBeLessThan(1e-9);
    // Sanity: DF-HF tracks exact HF to the fitting error (auto-aux level 1).
    expect(dfError).toBeLessThan(5e-2);
  });

  test("cooperative build: integrals built ONCE, fanned out, slices sum to reference", async () => {
    // Step 1 of the CI engine: build each V μ-block once and project to every
    // tab's mode slice — no N× redundant integral work.
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const n = shells.length;
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    const D = hf.D;

    const aux = generateAutoAux(shells, 1);
    const ref = await buildAuxBasisDF(shells, aux);
    const refJK = buildJK_DF(ref, D);

    const muBlock = 4;
    const nTabs = 4;
    const coop = await buildAuxBasisDFStreamingCooperative(shells, aux, nTabs, 1e-10, muBlock);

    // Sum partial (J,K) from the cooperatively-built tab slices.
    const J = new Float64Array(n * n);
    const K = new Float64Array(n * n);
    for (const s of coop.slices) {
      const part = buildJK_DF(s, D);
      for (let i = 0; i < J.length; i++) { J[i] = J[i]! + part.J[i]!; K[i] = K[i]! + part.K[i]!; }
    }
    const dJ = maxAbsDiff(refJK.J, J);
    const dK = maxAbsDiff(refJK.K, K);

    const expectedCalls = Math.ceil(n / muBlock);          // one pass over μ
    const independentCalls = nTabs * expectedCalls;        // what per-tab rebuild costs
    console.log(`[df-coop] ${nTabs} tabs, max|ΔJ|=${dJ.toExponential(2)} max|ΔK|=${dK.toExponential(2)}`);
    console.log(`[df-coop] kernelCalls=${coop.kernelCalls} (cooperative) vs ${independentCalls} (independent rebuild) — ${(independentCalls / coop.kernelCalls).toFixed(1)}× fewer`);
    console.log(`[df-coop] peak V floats=${coop.peakVFloats} vs full tensor ${n * n * aux.length}`);

    expect(dJ).toBeLessThan(1e-9);
    expect(dK).toBeLessThan(1e-9);
    // No redundancy: each integral built exactly once regardless of tab count.
    expect(coop.kernelCalls).toBe(expectedCalls);
    // Never materializes the full 3-index tensor.
    expect(coop.peakVFloats).toBeLessThan(n * n * aux.length);
  });

  test("partition:{tab,of} — independent tabs self-tile the mode axis", async () => {
    // The swarm one-liner: each tab calls with its index, no shared nKept.
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const n = shells.length;
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const D = runRHFSCF(integrals, 10).D;
    const aux = generateAutoAux(shells, 1);

    const full = await buildAuxBasisDFStreaming(shells, aux);
    const refJK = buildJK_DF(full, D);

    const OF = 5;
    const J = new Float64Array(n * n);
    const K = new Float64Array(n * n);
    let totalModes = 0;
    for (let tab = 0; tab < OF; tab++) {
      // Exactly what a tab runs — no knowledge of the others or of nKept.
      const slice = await buildAuxBasisDFStreaming(shells, aux, 1e-10, { partition: { tab, of: OF } });
      totalModes += slice.nAux;
      const part = buildJK_DF(slice, D);
      for (let i = 0; i < J.length; i++) { J[i] = J[i]! + part.J[i]!; K[i] = K[i]! + part.K[i]!; }
    }
    expect(totalModes).toBe(full.nAux); // tiles cover every mode exactly once
    expect(maxAbsDiff(refJK.J, J)).toBeLessThan(1e-12);
    expect(maxAbsDiff(refJK.K, K)).toBeLessThan(1e-12);
  });
});
