// Tier 2 stage 19: ideal-gas thermochemistry.
//
// Pass bars:
//   1. H₂O at 298.15 K, 1 atm: ZPE in [10, 18] kcal/mol (HF/STO-3G
//      has freqs ~10% high vs experiment, so ZPE inflates).
//      Reference experimental ZPE = 13.4 kcal/mol; HF/STO-3G ~14.5.
//   2. Total entropy at 298 K: in [40, 60] cal/(mol·K). Experimental
//      H₂O is 45.1 cal/(mol·K).
//   3. G(T) − E_e is small and positive at 298 K (entropy mostly
//      cancels enthalpy + ZPE).
//   4. Linear molecule (H₂) flag works: rotational entropy formula
//      switches from asymmetric-top to linear; result is finite.
//   5. Symmetry number σ scales rotational entropy as −R·ln(σ):
//      doubling σ subtracts R·ln(2) ≈ 1.376 cal/(mol·K).
import { describe, expect, test } from "vitest";
import {
  thermochemistry,
  HA_TO_KCAL_PER_MOL, HA_PER_K_TO_CAL_PER_MOL_K,
} from "../../src/chemistry/thermochemistry.js";
import type { Atom } from "../../src/chemistry/atoms.js";

const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ] as Atom[];
})();

// HF/STO-3G frequencies for H₂O (from stage 16 reference output).
const H2O_FREQS_CM_INV = new Float64Array([2170, 4140, 4391]);

describe("Thermochemistry — H₂O at 298.15 K, 1 atm", () => {
  test("ZPE for H₂O HF/STO-3G is ~13–17 kcal/mol", () => {
    const r = thermochemistry(H2O, H2O_FREQS_CM_INV, { symmetryNumber: 2 });
    const zpe_kcal = r.zpe * HA_TO_KCAL_PER_MOL;
    // ZPE = (1/2)·Σ ℏω_k. With freqs 2170, 4140, 4391 cm⁻¹:
    //   ZPE = (1/2)·(2170+4140+4391) cm⁻¹ = 5350 cm⁻¹ = 15.3 kcal/mol.
    expect(zpe_kcal).toBeGreaterThan(13);
    expect(zpe_kcal).toBeLessThan(17);
  });

  test("Total entropy at 298 K is in the experimental range", () => {
    const r = thermochemistry(H2O, H2O_FREQS_CM_INV, { symmetryNumber: 2 });
    const S_cal = r.entropy * HA_PER_K_TO_CAL_PER_MOL_K;
    // Experiment H₂O: 45.1 cal/(mol·K). HF/STO-3G freqs are ~10% high,
    // which slightly UNDERESTIMATES vibrational entropy (modes too stiff),
    // but vibrational contribution is small at room T anyway. Translation
    // + rotation dominates.
    expect(S_cal).toBeGreaterThan(40);
    expect(S_cal).toBeLessThan(50);
  });

  test("Per-component breakdown sums correctly", () => {
    const r = thermochemistry(H2O, H2O_FREQS_CM_INV, { symmetryNumber: 2 });
    // U_thermal = ZPE + U_vib_thermal + U_rot + U_trans.
    const sum = r.zpe
              + r.breakdown.vibrationalU
              + r.breakdown.rotationalU
              + r.breakdown.translationalU;
    expect(Math.abs(r.thermalU - sum)).toBeLessThan(1e-12);
    // S_total = S_trans + S_rot + S_vib + S_elec.
    const sumS = r.breakdown.translationalS
               + r.breakdown.rotationalS
               + r.breakdown.vibrationalS
               + r.breakdown.electronicS;
    expect(Math.abs(r.entropy - sumS)).toBeLessThan(1e-12);
  });

  test("Gibbs correction is small but positive at 298 K", () => {
    const r = thermochemistry(H2O, H2O_FREQS_CM_INV, { symmetryNumber: 2 });
    const G_kcal = r.gibbsCorrection * HA_TO_KCAL_PER_MOL;
    // G(T) − E_e = thermal H − T·S. For H₂O at 298 K, this is
    // typically in the 0–5 kcal/mol range (entropy nearly cancels
    // enthalpy + ZPE; the residual is the chemical-accuracy "free
    // energy correction" reported in chemistry papers).
    expect(G_kcal).toBeGreaterThan(0);
    expect(G_kcal).toBeLessThan(10);
  });
});

describe("Thermochemistry — H₂ linear molecule", () => {
  test("H₂ at 298 K with linear: true gives finite ZPE + S", () => {
    const H2: Atom[] = [
      { symbol: "H", pos: [0, 0, -0.37] },
      { symbol: "H", pos: [0, 0,  0.37] },
    ];
    // HF/STO-3G H₂ stretch ≈ 5060 cm⁻¹.
    const freqs = new Float64Array([5060]);
    const r = thermochemistry(H2, freqs, {
      symmetryNumber: 2, linear: true,
    });
    expect(Number.isFinite(r.zpe)).toBe(true);
    expect(Number.isFinite(r.entropy)).toBe(true);
    expect(r.zpe).toBeGreaterThan(0);
    // H₂ experimental S(298 K, 1 atm) ≈ 31.1 cal/(mol·K).
    const S_cal = r.entropy * HA_PER_K_TO_CAL_PER_MOL_K;
    expect(S_cal).toBeGreaterThan(25);
    expect(S_cal).toBeLessThan(40);
  });
});

describe("Symmetry-number scaling of rotational entropy", () => {
  test("Doubling σ subtracts exactly R·ln(2) from S_rot", () => {
    const r1 = thermochemistry(H2O, H2O_FREQS_CM_INV, { symmetryNumber: 1 });
    const r2 = thermochemistry(H2O, H2O_FREQS_CM_INV, { symmetryNumber: 2 });
    const dS = r1.breakdown.rotationalS - r2.breakdown.rotationalS;
    // R·ln(2) in Hartree/K = 3.166811e-6 · 0.693 ≈ 2.195e-6.
    const expected = 3.166811563e-6 * Math.log(2);
    expect(Math.abs(dS - expected)).toBeLessThan(1e-12);
  });
});
