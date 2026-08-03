// Permanent verifier for the DFResult stride contract.
//
// Regression: buildAuxBasisDF regularizes by zeroing λ⁻¹⸍² but emits B at the
// FULL nAux stride, yet used to report `nAux: nKept`. Every consumer strides by
// the reported nAux (`df.ts` buildJK_DF / reconstructERI do `B[i·nAux + P]`), so
// any dropped mode misaligned every read — silently, with no throw and no NaN.
//
// Measured before the fix, H₂O/STO-3G, one single dropped mode at reg=1e-4
// (65 of 66 kept): max|J_DF − J_exact| = 5.0 Ha. At the shipped default
// reg=1e-10 nothing is dropped, so this was latent — which is exactly why it
// needs a test that deliberately forces rank deficiency.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { buildAuxBasisDF, buildAuxBasisDFStreaming, generateAutoAux } from "../../src/chemistry/df-aux.js";
import { buildJK_DF } from "../../src/chemistry/df.js";

const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

function exactJ(eri: Float64Array, D: Float64Array, n: number): Float64Array {
  const J = new Float64Array(n * n);
  for (let mu = 0; mu < n; mu++)
    for (let nu = 0; nu < n; nu++) {
      let s = 0;
      for (let la = 0; la < n; la++)
        for (let si = 0; si < n; si++)
          s += eri[((mu * n + nu) * n + la) * n + si]! * D[la * n + si]!;
      J[mu * n + nu] = s;
    }
  return J;
}

describe("DFResult stride contract", () => {
  // Regularizations chosen to bracket the rank-deficient regime: 1e-10 keeps all
  // 66 aux modes on this system, 1e-4 drops one, 1e-2 drops seven.
  test("reported nAux equals B's actual stride, and J stays accurate, at every regularization", async () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O as never, "sto-3g");
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = shells.length;
    const aux = generateAutoAux(shells, 1);

    const D = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) D[i * n + j] = i === j ? 1.0 : 0.15;
    const Jref = exactJ(integrals.eri_AO, D, n);

    for (const reg of [1e-10, 1e-6, 1e-4, 1e-2]) {
      const df = await buildAuxBasisDF(shells, aux, reg);
      const actualStride = df.B.length / (n * n);

      // The contract itself. This alone would have caught the bug.
      expect(df.nAux, `reg=${reg}: reported nAux must equal B's real stride`).toBe(actualStride);

      const { J } = buildJK_DF(df, D);
      let maxAbs = 0;
      for (let i = 0; i < n * n; i++) maxAbs = Math.max(maxAbs, Math.abs(J[i]! - Jref[i]!));

      // Bound is the DF *fitting* error for this aux basis (~3e-4 Ha), generously
      // rounded. The bug produced 5.0 Ha here — five thousand× outside this bar,
      // so the gate does not depend on a tight tolerance to bite.
      expect(maxAbs, `reg=${reg}: DF J vs exact J`).toBeLessThan(5e-3);
    }
  }, 300_000);

  test("streaming builder compacts B, so its nAux is also its true stride", async () => {
    const { shells } = moleculeToShellsNuclei(H2O as never, "sto-3g");
    const n = shells.length;
    const aux = generateAutoAux(shells, 1);
    // Deliberately rank-deficient so the compaction path is actually exercised.
    const df = await buildAuxBasisDFStreaming(shells, aux, 1e-4);
    expect(df.nAux).toBe(df.B.length / (n * n));
  }, 300_000);
});
