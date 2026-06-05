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
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { buildJK_DF } from "../../src/chemistry/df.js";
import {
  buildAuxBasisDF,
  buildAuxBasisDFStreaming,
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
});
