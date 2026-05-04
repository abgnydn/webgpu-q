// dmrg.ts validation: exact ground-state via direct diagonalization
// + statevector-to-MPS compression. The MPS form should:
//   1. Have energy matching the dense diagonalization to f64 precision
//      (no information loss when chiMax ≥ required bond dim).
//   2. Truncate gracefully at smaller chiMax (truncation fidelity drops
//      smoothly as chi shrinks).
//   3. Match the ITensor reference energies for the same model + N.

import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { heisenberg, tfim, xxz } from "../../src/manybody/hamiltonian.js";
import { exactGroundStateMPS } from "../../src/manybody/dmrg.js";
import { energyOf } from "../../src/manybody/ground-state.js";

interface ItensorEntry {
  model: "heisenberg" | "tfim" | "xxz";
  N?: number | string;
  energy?: number;
  args?: Record<string, number>;
  regime?: string;
  kind?: string;
}

const refPath = resolve(__dirname, "itensor-reference.json");
const ref = JSON.parse(readFileSync(refPath, "utf-8")) as { results: ItensorEntry[] };

describe("exactGroundStateMPS: dense-eigh + SVD compression", () => {
  test("Heisenberg N=6 J=1: MPS energy matches direct diagonalization", () => {
    const H = heisenberg(6, 1);
    const result = exactGroundStateMPS(H);
    const eMps = energyOf(result.mps, H);
    // Same Hamiltonian, same wavefunction (untruncated since chiMax ≥ 8 ≥ max bond).
    expect(Math.abs(eMps - result.energy)).toBeLessThan(1e-9);
    expect(result.truncationFidelity).toBeGreaterThan(1 - 1e-12);
  });

  test("TFIM N=6 J=1 h=1: MPS energy matches direct diagonalization at chiMax=8", () => {
    const H = tfim(6, 1, 1);
    const result = exactGroundStateMPS(H, { chiMax: 8 });
    const eMps = energyOf(result.mps, H);
    expect(Math.abs(eMps - result.energy)).toBeLessThan(1e-9);
    expect(result.truncationFidelity).toBeGreaterThan(1 - 1e-12);
  });

  test("XXZ N=6 J=1 Δ=1: MPS energy matches direct diagonalization", () => {
    const H = xxz(6, 1, 1);
    const result = exactGroundStateMPS(H);
    const eMps = energyOf(result.mps, H);
    expect(Math.abs(eMps - result.energy)).toBeLessThan(1e-9);
    expect(result.truncationFidelity).toBeGreaterThan(1 - 1e-12);
  });

  test("Aggressive truncation: chiMax=2 on Heisenberg N=6 keeps the trend", () => {
    const H = heisenberg(6, 1);
    const fullChi = exactGroundStateMPS(H);                 // chiMax = 2^3 = 8
    const lowChi = exactGroundStateMPS(H, { chiMax: 2 });
    // Truncation should drop fidelity but the truncated-MPS energy should
    // be ≥ the exact energy (variational bound) — it's an approximation
    // of a state in a smaller manifold.
    const eLow = energyOf(lowChi.mps, H);
    expect(eLow).toBeGreaterThan(fullChi.energy - 1e-9);
    // And truncation fidelity should drop noticeably from full.
    expect(lowChi.truncationFidelity).toBeLessThan(fullChi.truncationFidelity);
  });
});

describe("exactGroundStateMPS vs ITensor", () => {
  for (const entry of ref.results) {
    if (entry.kind || typeof entry.N !== "number") continue;
    if (entry.N > 8) continue; // dense Hamiltonian capped at N=8 for unit-test speed
    const N = entry.N;
    const args = entry.args ?? {};
    const expected = entry.energy!;
    const label = `${entry.model} N=${N} ${entry.regime ? `(${entry.regime})` : ""}`.trim();
    test(`${label}: dense-MPS energy matches ITensor DMRG`, () => {
      let H;
      if (entry.model === "heisenberg") H = heisenberg(N, args["1"]!);
      else if (entry.model === "tfim")   H = tfim(N, args["1"]!, args["2"]!);
      else if (entry.model === "xxz")    H = xxz(N, args["1"]!, args["2"]!);
      else throw new Error(`unknown model ${entry.model}`);

      const result = exactGroundStateMPS(H);
      // Direct-diag energy should match to machine precision; the MPS-form
      // energy should match too since chiMax = 2^(N/2) ≥ Schmidt rank.
      expect(Math.abs(result.energy - expected)).toBeLessThan(1e-7);
      const eMps = energyOf(result.mps, H);
      expect(Math.abs(eMps - expected)).toBeLessThan(1e-7);
    });
  }
});
