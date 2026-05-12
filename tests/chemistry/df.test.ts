// Density fitting (Cholesky DF) validation (Tier 2 stage 26).
// Strategy:
//   1. B-tensor reconstruction: |ERI_exact − B·B^T|_max ≤ threshold,
//      across H₂O, BeH₂ STO-3G systems and multiple thresholds.
//   2. J + K from B-tensor matches direct J + K from full ERI to the
//      same threshold-controlled precision.
//   3. Threshold convergence: tightening threshold strictly improves
//      reconstruction error (rank increases monotonically).
//   4. DF-HF energy: F = h + 2J − K built from DF reproduces the
//      exact HF energy to threshold precision when applied as a
//      post-processing on the converged HF density.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runMP2 } from "../../src/chemistry/mp2.js";
import {
  choleskyDecomposeERI,
  reconstructERI,
  buildJK_DF,
} from "../../src/chemistry/df.js";

const H2O: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
})();

const BEH2: Atom[] = [
  { symbol: "Be", pos: [0, 0, 0] },
  { symbol: "H",  pos: [0, 0, -1.34] },
  { symbol: "H",  pos: [0, 0,  1.34] },
];

function maxAbsDiff(A: Float64Array, B: Float64Array): number {
  let m = 0;
  for (let i = 0; i < A.length; i++) {
    const d = Math.abs(A[i]! - B[i]!);
    if (d > m) m = d;
  }
  return m;
}

describe("DF-MP2 — Tier 2 stage 34", () => {
  test("H₂O STO-3G — DF-MP2 matches exact MP2 to µHa", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    const mp2Direct = runMP2(hf, integrals);
    const mp2DF = runMP2(hf, integrals, { useDF: true });
    const err = Math.abs(mp2DF.correlationEnergy - mp2Direct.correlationEnergy);
    console.log(`[df-mp2-h2o] direct E_corr = ${mp2Direct.correlationEnergy.toExponential(6)}`);
    console.log(`[df-mp2-h2o] DF E_corr     = ${mp2DF.correlationEnergy.toExponential(6)}`);
    console.log(`[df-mp2-h2o] Δ              = ${err.toExponential(2)} Ha`);
    expect(err).toBeLessThan(1e-9);
  });

  test("BeH₂ STO-3G — DF-MP2 threshold trace converges", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(BEH2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 6);
    const direct = runMP2(hf, integrals);
    const trace: { tau: number; err: number }[] = [];
    for (const tau of [1e-3, 1e-6, 1e-9]) {
      const dfRes = runMP2(hf, integrals, { useDF: tau });
      trace.push({ tau, err: Math.abs(dfRes.correlationEnergy - direct.correlationEnergy) });
    }
    console.log(`[df-mp2-beh2-thresholds]`);
    for (const t of trace) console.log(`  τ=${t.tau}: Δ=${t.err.toExponential(2)}`);
    for (let i = 1; i < trace.length; i++) {
      expect(trace[i]!.err).toBeLessThanOrEqual(trace[i - 1]!.err * 1.01);
    }
  });
});

describe("DF-HF SCF integration — Tier 2 stage 29", () => {
  test("H₂O STO-3G — runRHFSCF with useDF=true matches direct HF to µHa", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hfDirect = runRHFSCF(integrals, 10);
    const hfDF = runRHFSCF(integrals, 10, { useDF: true });
    expect(hfDirect.converged).toBe(true);
    expect(hfDF.converged).toBe(true);
    const err = Math.abs(hfDF.energy - hfDirect.energy);
    console.log(`[df-hf-scf-h2o] direct = ${hfDirect.energy.toFixed(10)}, DF = ${hfDF.energy.toFixed(10)}, Δ = ${err.toExponential(2)}`);
    expect(err).toBeLessThan(1e-7);
    // Orbital energies should also agree closely.
    for (let i = 0; i < integrals.n; i++) {
      expect(Math.abs(hfDF.orbitalEnergies[i]! - hfDirect.orbitalEnergies[i]!)).toBeLessThan(1e-6);
    }
  });

  test("H₂O STO-3G — threshold knob controls DF-HF accuracy", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hfDirect = runRHFSCF(integrals, 10);
    const trace: { tau: number; err: number }[] = [];
    for (const tau of [1e-3, 1e-6, 1e-9]) {
      const hfDF = runRHFSCF(integrals, 10, { useDF: tau });
      const err = Math.abs(hfDF.energy - hfDirect.energy);
      trace.push({ tau, err });
    }
    console.log(`[df-hf-scf-thresholds]`);
    for (const t of trace) console.log(`  τ=${t.tau}: Δ=${t.err.toExponential(2)}`);
    // Each tighter threshold strictly improves (or doesn't worsen) the energy match.
    for (let i = 1; i < trace.length; i++) {
      expect(trace[i]!.err).toBeLessThanOrEqual(trace[i - 1]!.err * 1.01);
    }
  });
});

describe("Density fitting (CD-DF) — Tier 2 stage 26", () => {
  test("H₂O STO-3G — B·B^T reconstructs ERI to threshold precision", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = integrals.n;
    const N = n * n;
    for (const tau of [1e-3, 1e-6, 1e-10]) {
      const df = choleskyDecomposeERI(integrals.eri_AO, n, tau);
      const recon = reconstructERI(df);
      // Reshape eri_AO from [(μ*n+ν)*n+λ]·n+σ to [(μ·n+ν), (λ·n+σ)] matrix.
      const eri_flat = new Float64Array(N * N);
      for (let mu = 0; mu < n; mu++)
        for (let nu = 0; nu < n; nu++)
          for (let la = 0; la < n; la++)
            for (let si = 0; si < n; si++)
              eri_flat[(mu * n + nu) * N + (la * n + si)] =
                integrals.eri_AO[((mu * n + nu) * n + la) * n + si]!;
      const err = maxAbsDiff(eri_flat, recon);
      console.log(`[df-h2o-sto3g] τ=${tau}: nAux=${df.nAux} (of n²=${N}), maxErr=${err.toExponential(2)}`);
      expect(df.nAux).toBeLessThanOrEqual(N);
      expect(err).toBeLessThan(tau * 10); // factor 10 margin for accumulated roundoff
    }
  });

  test("H₂O STO-3G — DF J+K matches direct J+K on the HF density", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    expect(hf.converged).toBe(true);
    const n = integrals.n;
    const df = choleskyDecomposeERI(integrals.eri_AO, n, 1e-10);
    const { J: J_df, K: K_df } = buildJK_DF(df, hf.D);

    // Direct J, K from full ERI.
    const J_direct = new Float64Array(n * n);
    const K_direct = new Float64Array(n * n);
    for (let mu = 0; mu < n; mu++) {
      for (let nu = 0; nu < n; nu++) {
        let j = 0, k = 0;
        for (let la = 0; la < n; la++) {
          for (let si = 0; si < n; si++) {
            j += integrals.eri_AO[((mu * n + nu) * n + la) * n + si]! * hf.D[la * n + si]!;
            k += integrals.eri_AO[((mu * n + la) * n + nu) * n + si]! * hf.D[la * n + si]!;
          }
        }
        J_direct[mu * n + nu] = j;
        K_direct[mu * n + nu] = k;
      }
    }
    const jErr = maxAbsDiff(J_df, J_direct);
    const kErr = maxAbsDiff(K_df, K_direct);
    console.log(`[df-h2o-jk] maxErr  J=${jErr.toExponential(2)}, K=${kErr.toExponential(2)}`);
    expect(jErr).toBeLessThan(1e-9);
    expect(kErr).toBeLessThan(1e-9);
  });

  test("H₂O STO-3G — DF-HF energy matches exact HF to µHa precision", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(H2O);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    expect(hf.converged).toBe(true);
    const n = integrals.n;
    const df = choleskyDecomposeERI(integrals.eri_AO, n, 1e-10);
    const { J, K } = buildJK_DF(df, hf.D);
    // E_HF = ½ Σ_μν D[μν] · (h[μν] + F[μν])  with  F = h + J − ½ K (closed-shell).
    let E_elec_df = 0;
    for (let mu = 0; mu < n; mu++) {
      for (let nu = 0; nu < n; nu++) {
        const F_df = integrals.h_AO[mu * n + nu]! + J[mu * n + nu]! - 0.5 * K[mu * n + nu]!;
        E_elec_df += 0.5 * hf.D[mu * n + nu]! * (integrals.h_AO[mu * n + nu]! + F_df);
      }
    }
    const E_total_df = E_elec_df + integrals.Vnn;
    const err = Math.abs(E_total_df - hf.energy);
    console.log(`[df-h2o-hf] exact HF = ${hf.energy.toFixed(10)}, DF HF = ${E_total_df.toFixed(10)}, Δ = ${err.toExponential(2)}`);
    expect(err).toBeLessThan(1e-7);
  });

  test("BeH₂ STO-3G — threshold convergence: tighter τ → smaller error + larger nAux", () => {
    const { shells, nuclei } = moleculeToShellsNuclei(BEH2);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const n = integrals.n;
    const N = n * n;
    const eri_flat = new Float64Array(N * N);
    for (let mu = 0; mu < n; mu++)
      for (let nu = 0; nu < n; nu++)
        for (let la = 0; la < n; la++)
          for (let si = 0; si < n; si++)
            eri_flat[(mu * n + nu) * N + (la * n + si)] =
              integrals.eri_AO[((mu * n + nu) * n + la) * n + si]!;

    const taus = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10];
    const trace: { tau: number; nAux: number; err: number }[] = [];
    for (const tau of taus) {
      const df = choleskyDecomposeERI(integrals.eri_AO, n, tau);
      const err = maxAbsDiff(eri_flat, reconstructERI(df));
      trace.push({ tau, nAux: df.nAux, err });
    }
    console.log(`[df-beh2] convergence trace:`);
    for (const t of trace) {
      console.log(`  τ=${t.tau}: nAux=${t.nAux}, maxErr=${t.err.toExponential(2)}`);
    }
    // nAux is monotonic (non-decreasing) as τ tightens.
    for (let i = 1; i < trace.length; i++) {
      expect(trace[i]!.nAux).toBeGreaterThanOrEqual(trace[i - 1]!.nAux);
      // Each tighter τ reduces error (or at least doesn't increase).
      expect(trace[i]!.err).toBeLessThanOrEqual(trace[i - 1]!.err * 1.01);
    }
    // Final error reaches at least 1e-9 with τ=1e-10.
    expect(trace[trace.length - 1]!.err).toBeLessThan(1e-9);
  });
});
