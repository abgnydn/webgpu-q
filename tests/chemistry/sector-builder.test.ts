// Sector-direct H builder validation. Strategy:
//   1. Cross-check against the existing LiH dense+project path —
//      same FCI energy in the N=4 sector to f64 precision.
//   2. Hermiticity / shape sanity for the BeH₂-full sector H.
import { describe, expect, test } from "vitest";
import { computeLiHIntegrals } from "../../src/chemistry/lih-builder.js";
import { lowestInParticleSector } from "../../src/chemistry/sector.js";
import { buildLiHDense } from "../../src/chemistry/lih-builder.js";
import { buildSectorH } from "../../src/chemistry/sector-builder.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";

describe("Sector-direct H ≡ dense+project on LiH N=4", () => {
  test("FCI energy identical to f64 precision", () => {
    const R = 1.595;
    const I = computeLiHIntegrals(R);
    // Sector-direct path:
    const sec = buildSectorH(3, I.h_OAO, I.eri_OAO, I.Vnn, 4);
    const eig = eigsymmetric(sec.H, sec.k);
    const E_sector = eig.values[0]!;
    // Dense + project path (the existing one):
    const { H } = buildLiHDense(R);
    const E_dense = lowestInParticleSector(H, 6, 4).energy;
    expect(E_sector).toBeCloseTo(E_dense, 12);
  });

  test("sector dim matches C(2n, N_e) on LiH N=4", () => {
    const I = computeLiHIntegrals(1.595);
    const sec = buildSectorH(3, I.h_OAO, I.eri_OAO, I.Vnn, 4);
    // C(6, 4) = 15.
    expect(sec.k).toBe(15);
    expect(sec.nQubits).toBe(6);
    expect(sec.particleNumber).toBe(4);
  });

  test("Hsec is real symmetric on LiH N=4", () => {
    const I = computeLiHIntegrals(1.595);
    const sec = buildSectorH(3, I.h_OAO, I.eri_OAO, I.Vnn, 4);
    const { H, k } = sec;
    let maxAsym = 0;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        const d = Math.abs(H[i * k + j]! - H[j * k + i]!);
        if (d > maxAsym) maxAsym = d;
      }
    }
    expect(maxAsym).toBeLessThan(1e-12);
  });
});
