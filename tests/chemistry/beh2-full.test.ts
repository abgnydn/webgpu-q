// Full STO-3G BeH₂ FCI validation. Phase C v3 stage 2.
//
// Cross-checks:
//   1. Hermiticity of the sector-projected H.
//   2. FCI ground-state energy at R = 1.34 Å matches the PySCF
//      literature reference for BeH₂ STO-3G FCI (≈ -15.595 Ha).
//      We allow 1 mHa slack — the published value depends on the
//      exact basis-set tabulation and erf precision used.
//   3. Full STO-3G recovers ≥ 200 mHa of correlation vs s-only.
//   4. Dissociation curve has a minimum at R = 1.34 Å.
import { describe, expect, test } from "vitest";
import { buildBeH2Full } from "../../src/chemistry/beh2-full-builder.js";
import { lanczosGroundState } from "../../src/manybody/lanczos.js";

function fciEnergy(H: Float64Array, k: number): number {
  const matvec = (x: Float64Array, out: Float64Array) => {
    for (let i = 0; i < k; i++) {
      let s = 0;
      const row = i * k;
      for (let j = 0; j < k; j++) s += H[row + j]! * x[j]!;
      out[i] = s;
    }
  };
  const x0 = new Float64Array(k);
  for (let i = 0; i < k; i++) x0[i] = 1;
  const r = lanczosGroundState(matvec, x0, { maxIter: 100, tol: 1e-10 });
  return r.energy;
}

describe("BeH₂ full STO-3G — sector-projected H invariants", () => {
  test("sector dim = C(14, 6) = 3003", () => {
    const data = buildBeH2Full(1.34);
    expect(data.sector.k).toBe(3003);
    expect(data.sector.nQubits).toBe(14);
    expect(data.sector.particleNumber).toBe(6);
  });

  test("Hsec is real symmetric to f64 precision", () => {
    const data = buildBeH2Full(1.34);
    const { H, k } = data.sector;
    let maxAsym = 0;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        const d = Math.abs(H[i * k + j]! - H[j * k + i]!);
        if (d > maxAsym) maxAsym = d;
      }
    }
    expect(maxAsym).toBeLessThan(1e-12);
  });

  test("V_nn = (4·1/R + 4·1/R + 1·1/(2R)) Ha at R = 1.34 Å", () => {
    const data = buildBeH2Full(1.34);
    const expected = (4 / data.R_Bohr) + (4 / data.R_Bohr) + (1 / (2 * data.R_Bohr));
    expect(data.integrals.Vnn).toBeCloseTo(expected, 12);
  });
});

describe("BeH₂ full STO-3G — FCI energy vs literature", () => {
  test("R = 1.34 Å (experimental Be–H bond): E_FCI matches PySCF reference", () => {
    // PySCF STO-3G FCI ≈ -15.5949 Ha. We compute -15.59486 Ha;
    // the gap to the published value is well within the
    // basis-set / erf-precision tabulation slack.
    const data = buildBeH2Full(1.34);
    const E = fciEnergy(data.sector.H, data.sector.k);
    expect(E).toBeCloseTo(-15.5949, 3);  // within 0.5 mHa
  }, 60_000);

  test("full STO-3G recovers ≥ 200 mHa of correlation vs s-only", () => {
    // s-only BeH₂ at R = 1.34 = -15.349162 Ha (Phase C v2).
    // Full STO-3G should be lower by the Be 2p contribution.
    const data = buildBeH2Full(1.34);
    const E_full = fciEnergy(data.sector.H, data.sector.k);
    const E_sOnly = -15.349162;
    expect(E_full - E_sOnly).toBeLessThan(-0.2);  // ≥ 200 mHa more bound
  }, 60_000);

  test("dissociation curve has its minimum near the experimental Be–H bond R = 1.34 Å", () => {
    // Sample the well; minimum should sit at R ∈ [1.2, 1.5] Å.
    const Rs = [1.0, 1.2, 1.34, 1.5, 1.7];
    const energies = Rs.map((R) => {
      const d = buildBeH2Full(R);
      return fciEnergy(d.sector.H, d.sector.k);
    });
    let minIdx = 0;
    for (let i = 1; i < energies.length; i++) {
      if (energies[i]! < energies[minIdx]!) minIdx = i;
    }
    expect(Rs[minIdx]!).toBeGreaterThanOrEqual(1.2);
    expect(Rs[minIdx]!).toBeLessThanOrEqual(1.5);
  }, 90_000);
});
