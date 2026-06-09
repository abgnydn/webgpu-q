// Unit tests for the swarm chemistry kernel — runs HF on individual
// molecules + a bond-length scan that should bracket the H₂ minimum.

import { describe, expect, test } from "vitest";
import {
  runChemEnergyTile, bondScanTiles, type ChemEnergyTile,
} from "../../src/swarm/chemistry-kernel.js";

describe("Swarm chem-energy kernel", () => {
  test("HF on H₂ STO-3G at r=0.7414 Å gives the expected energy", async () => {
    const tile: ChemEnergyTile = {
      label: "H2 at equilibrium",
      atoms: [
        { symbol: "H", pos: [0, 0, 0] },
        { symbol: "H", pos: [0, 0, 0.7414] },
      ],
      method: "hf",
      basis: "sto-3g",
    };
    const r = await runChemEnergyTile(tile);
    // E_HF(H2/STO-3G, r=0.7414) ≈ -1.117 Ha (Szabo & Ostlund Table 3.5
    // lists -1.1175 with a slightly different basis variant; our
    // implementation gives -1.11668 which is the standard STO-3G value).
    expect(r.energy).toBeCloseTo(-1.117, 2);
    expect(r.converged).toBe(true);
    expect(r.nElectrons).toBe(2);
    expect(r.nBasisFunctions).toBe(2);
    expect(r.dfMethod).toBe("exact-eri"); // tiny tile → exact, gold standard
  });

  test("Bond-scan on H₂ has its minimum near 0.73 Å", async () => {
    const tiles = bondScanTiles({
      atomA: "H", atomB: "H", basis: "sto-3g",
      rMin: 0.5, rMax: 1.4, nPoints: 19,
    });
    expect(tiles.length).toBe(19);
    const results = await Promise.all(tiles.map((t) => runChemEnergyTile(t)));
    // Find the minimum-energy index.
    let minI = 0;
    for (let i = 1; i < results.length; i++) {
      if (results[i]!.energy < results[minI]!.energy) minI = i;
    }
    const rEq = 0.5 + minI * (1.4 - 0.5) / 18;
    // HF/STO-3G predicts H₂ r_eq ≈ 0.73 Å (literature: 0.730 Å,
    // experiment 0.7414 Å — the basis biases r_eq slightly short).
    expect(rEq).toBeGreaterThanOrEqual(0.7);
    expect(rEq).toBeLessThanOrEqual(0.8);
  });

  test("Bond-scan energies are smooth (no sudden jumps near minimum)", async () => {
    const tiles = bondScanTiles({
      atomA: "H", atomB: "H", basis: "sto-3g",
      rMin: 0.6, rMax: 1.0, nPoints: 9,
    });
    const results = await Promise.all(tiles.map((t) => runChemEnergyTile(t)));
    const energies = results.map((r) => r.energy);
    // Difference between adjacent points should be < 0.05 Ha — smooth.
    for (let i = 1; i < energies.length; i++) {
      const d = Math.abs(energies[i]! - energies[i - 1]!);
      expect(d).toBeLessThan(0.05);
    }
  });
});
