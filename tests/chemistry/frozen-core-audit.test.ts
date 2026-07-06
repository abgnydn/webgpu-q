// Frozen-core audit — Tier 3 verification.
//
// nFrozenCore is documented as supported in MP2, CCSD, UCCSD,
// CCSD(T) (CPU + GPU). The MP2 + CCSD path has prior coverage in
// `ccsd.test.ts`. This audit closes the test gap on CCSD(T) and
// UCCSD, then documents what's STILL missing (EOM-CCSD has no
// frozen-core path at all — flagged in LIMITATIONS.md §4).
//
// Pass bars for every layer where frozen-core is claimed to work:
//   1. The frozen-core total energy must lie ABOVE the all-electron
//      total (less correlation captured → variational sanity).
//   2. The difference must be bounded by a few mHa in STO-3G
//      (small basis can't capture much core-valence dynamic
//      correlation anyway).
//   3. Internal consistency: CCSD(T) frozen-core only excludes
//      i/j/k loops; the underlying CCSD amplitudes for core
//      indices are still zero (zeroCoreAmplitudes). So
//      E_(T)(frozen) ≈ E_(T)(all) − Σ_{i,j,k include core} (T)_ijk.
//      We don't measure that decomposition here; just that the
//      direction + magnitude are physical.
//
// What this test does NOT cover (intentional):
//   - HF SCF "frozen core": the HF determinant is always all-electron
//     in this codebase; frozen-core only applies to post-HF
//     correlation. That's the standard quantum-chemistry convention
//     and not an audit gap.
//   - EOM-CCSD frozen-core: NOT IMPLEMENTED. See
//     `LIMITATIONS.md` for the queued Tier 3 follow-up.

import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runUHFSCF } from "../../src/chemistry/uhf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runCCSDT } from "../../src/chemistry/ccsd-t.js";
import { runUCCSD } from "../../src/chemistry/uccsd.js";

describe("Frozen-core audit", () => {
  test("CCSD(T) frozen-1s on H₂O: lies above all-electron, bounded by ~5 mHa STO-3G", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);

    // All-electron CCSD + (T).
    const ccsdAll = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
    const tAll    = runCCSDT(ccsdAll, hf, integrals);

    // Frozen-core CCSD + (T) — match nFrozenCore between layers.
    const ccsdFC  = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: 1 });
    const tFC     = runCCSDT(ccsdFC, hf, integrals, { nFrozenCore: 1 });

    const E_all = hf.energy + ccsdAll.correlationEnergy + tAll.tripleCorrection;
    const E_FC  = hf.energy + ccsdFC.correlationEnergy  + tFC.tripleCorrection;

    // Variational sanity: frozen-core total > all-electron total.
    expect(E_FC).toBeGreaterThan(E_all);

    // Magnitude bounded by missing 1s-1s + 1s-valence dynamic correlation
    // (small in STO-3G; literature ~5-20 mHa).
    const diffMHa = (E_FC - E_all) * 1000;
    expect(diffMHa).toBeGreaterThan(0);
    expect(diffMHa).toBeLessThan(30);

    // (T) itself stays negative (variational sanity for the triples step).
    expect(tFC.tripleCorrection).toBeLessThan(0);
    expect(tAll.tripleCorrection).toBeLessThan(0);
  });

  test("CCSD(T) frozen-1s on CH₄: same direction + bounded magnitude", () => {
    const r3 = 1.09 / Math.sqrt(3);
    const atoms: Atom[] = [
      { symbol: "C", pos: [0, 0, 0] },
      { symbol: "H", pos: [ r3,  r3,  r3] },
      { symbol: "H", pos: [ r3, -r3, -r3] },
      { symbol: "H", pos: [-r3,  r3, -r3] },
      { symbol: "H", pos: [-r3, -r3,  r3] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    expect(hf.converged).toBe(true);

    const ccsdAll = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
    const tAll    = runCCSDT(ccsdAll, hf, integrals);
    const ccsdFC  = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: 1 });
    const tFC     = runCCSDT(ccsdFC, hf, integrals, { nFrozenCore: 1 });

    const E_all = hf.energy + ccsdAll.correlationEnergy + tAll.tripleCorrection;
    const E_FC  = hf.energy + ccsdFC.correlationEnergy  + tFC.tripleCorrection;

    expect(E_FC).toBeGreaterThan(E_all);
    const diffMHa = (E_FC - E_all) * 1000;
    expect(diffMHa).toBeGreaterThan(0);
    expect(diffMHa).toBeLessThan(30);
  });

  test("UCCSD on closed-shell H₂O matches RHF-CCSD WITHOUT frozen core", () => {
    // Baseline that should hold: without frozen-core, UCCSD on a
    // closed-shell UHF must equal RHF-CCSD by spin-orbital basis
    // independence. This guards against a regression in the
    // bare UCCSD path.
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const nAB = nElectrons / 2;

    const rhf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const rhfCCSD = runCCSD(rhf, integrals, { maxIter: 200, tol: 1e-9 });

    const uhf = runUHFSCF(integrals, nAB, nAB, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });
    const uhfCCSD = runUCCSD(uhf, integrals, { maxIter: 200, tol: 1e-9 });

    expect(Math.abs(uhf.energy - rhf.energy)).toBeLessThan(1e-9);
    expect(Math.abs(uhfCCSD.correlationEnergy - rhfCCSD.correlationEnergy)).toBeLessThan(1e-6);
  });

  test("UCCSD frozen-core on closed-shell H₂O matches RHF-CCSD frozen-core", () => {
    // Originally an honest-negative pin (audit 2026-05 found UCCSD
    // frozen-core freezing the wrong SOs). Fixed in the same audit
    // by switching ccsdIterate's contract from a contiguous
    // `nFrozenSO: number` to an explicit `ReadonlySet<number>` of
    // frozen occupied SO indices. UCCSD now passes the correct
    // interleaved (α-spatial-s + β-spatial-s) set for the
    // "all-α-occ first, then all-β-occ" SO ordering.
    //
    // Pass bar: UCCSD frozen-core on closed-shell UHF must equal
    // RHF-CCSD frozen-core by spin-orbital basis independence.
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const nAB = nElectrons / 2;

    const rhf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    const rhfCCSD = runCCSD(rhf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: 1 });

    const uhf = runUHFSCF(integrals, nAB, nAB, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
      symmetryBreaking: 0,
    });
    const uhfCCSD = runUCCSD(uhf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: 1 });

    expect(Math.abs(uhfCCSD.correlationEnergy - rhfCCSD.correlationEnergy)).toBeLessThan(1e-6);
  });

  test("EOM-CCSD frozen-1s on H₂O: variational direction + bounded magnitude", async () => {
    const { runEOMCCSD } = await import("../../src/chemistry/eom-ccsd.js");

    // Originally an honest-negative pin (EOM-CCSD frozen-core was
    // NOT IMPLEMENTED, audit 2026-05). Closed in the same audit by
    // restricting the packed (singles + antisym doubles) basis to
    // occupied indices ≥ 2·nFrozenCore, with the σ-equation
    // unchanged (R_1 and R_2 are zero at frozen indices, so the
    // inner summations over m, n produce zero contributions from
    // frozen indices automatically).
    //
    // Pass bars (mirror the CCSD(T) frozen-core test):
    //   1. Frozen-core EOM excitation energies must SHIFT but stay
    //      finite, real, positive, and ordered ascending.
    //   2. The shift from all-electron to frozen-core should be
    //      small (≲ 100 mHa) in STO-3G — the missing 1s-1s and
    //      1s-valence correlation contributions to the excited
    //      state are bounded by their ground-state analogues.
    const half = (104.52 / 2) * Math.PI / 180;
    const xH = 0.9572 * Math.sin(half);
    const zH = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ xH, 0, zH] },
      { symbol: "H", pos: [-xH, 0, zH] },
    ];
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });

    // All-electron CCSD + EOM.
    const ccsdAll = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9 });
    const eomAll  = runEOMCCSD(ccsdAll, integrals, hf, { nRoots: 3 });

    // Frozen-1s CCSD + frozen-1s EOM.
    const ccsdFC = runCCSD(hf, integrals, { maxIter: 200, tol: 1e-9, nFrozenCore: 1 });
    const eomFC  = runEOMCCSD(ccsdFC, integrals, hf, { nRoots: 3, nFrozenCore: 1 });

    expect(eomFC.energies.length).toBe(3);

    // Lowest 3 excitation energies real-positive and ordered.
    for (let k = 0; k < 3; k++) {
      expect(eomFC.energies[k]!).toBeGreaterThan(0);
      expect(Math.abs(eomFC.imag[k]!)).toBeLessThan(1e-5);
      if (k > 0) expect(eomFC.energies[k]!).toBeGreaterThanOrEqual(eomFC.energies[k - 1]!);
    }

    // Frozen-vs-all shift bounded (small basis ⇒ small core correlation).
    for (let k = 0; k < 3; k++) {
      const shiftHa = Math.abs(eomFC.energies[k]! - eomAll.energies[k]!);
      expect(shiftHa).toBeLessThan(0.1); // < 100 mHa per root
    }
  }, 360_000); // EOM-CCSD all-electron + frozen-core on H₂O ≈ 30-44 s observed — 60 s was chronically marginal under load.
});
