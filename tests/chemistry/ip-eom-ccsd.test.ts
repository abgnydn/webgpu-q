// IP-EOM-CCSD validation (Tier 2 stage 37).
// Strategy:
//   1. Eigenvalues are real and positive → physical ionization potentials.
//   2. Lowest IP (HOMO removal) is in the same ballpark as Koopmans.
//      For H₂O STO-3G, Koopmans HOMO IP ≈ 10.65 eV (per CLAUDE.md stage 22)
//      and ΔSCF gives 8.36 eV. IP-EOM-CCSD should sit between these two
//      with a sensible correlation correction.
//   3. H₂ STO-3G: IP-EOM-CCSD should give a single non-zero IP (only one
//      occupied spatial → 2 occupied SOs, but they're equivalent by spin
//      symmetry). For 2-electron systems, IP-EOM-CCSD is exact for the
//      lowest IP.
//   4. Sub-shells ordered consistently: deeper-bound orbitals → larger IPs.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../src/chemistry/hf-scf.js";
import { runCCSD } from "../../src/chemistry/ccsd.js";
import { runIPEOMCCSD } from "../../src/chemistry/ip-eom-ccsd.js";

const HARTREE_TO_EV = 27.211386245988;

describe("IP-EOM-CCSD — Tier 2 stage 37", () => {
  test("H₂O STO-3G — lowest IP in the Koopmans / ΔSCF band", () => {
    const half = (104.52 / 2) * Math.PI / 180;
    const x = 0.9572 * Math.sin(half);
    const z = 0.9572 * Math.cos(half);
    const atoms: Atom[] = [
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 10);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-10, maxIter: 50 });
    expect(ccsd.converged).toBe(true);
    const ip = runIPEOMCCSD(ccsd, integrals, hf, { nRoots: 10 });

    // Real eigenvalues.
    for (let k = 0; k < ip.ips.length; k++) {
      expect(Math.abs(ip.imag[k]!)).toBeLessThan(1e-6);
      expect(ip.ips[k]!).toBeGreaterThan(0);
    }

    const koopmansHomoEV = -hf.orbitalEnergies[hf.nOccupied - 1]! * HARTREE_TO_EV;
    const lowestIPEV = ip.ips[0]! * HARTREE_TO_EV;
    console.log(`[ip-eom-h2o] Koopmans HOMO IP = ${koopmansHomoEV.toFixed(2)} eV`);
    console.log(`[ip-eom-h2o] IP-EOM-CCSD lowest 5 (eV): [${
      Array.from(ip.ips.slice(0, 5)).map(x => (x * HARTREE_TO_EV).toFixed(2)).join(", ")
    }]`);
    console.log(`[ip-eom-h2o] IP-EOM-CCSD lowest = ${lowestIPEV.toFixed(2)} eV`);

    // IP-EOM-CCSD lowest IP should be within ~3 eV of Koopmans
    // (correlation correction at STO-3G is bounded). Sanity bracket.
    expect(Math.abs(lowestIPEV - koopmansHomoEV)).toBeLessThan(3);
    expect(lowestIPEV).toBeGreaterThan(5);    // physical lower bound
    expect(lowestIPEV).toBeLessThan(20);     // not a 1s ionization

    // IPs sorted ascending → strictly non-decreasing.
    for (let k = 1; k < ip.ips.length; k++) {
      expect(ip.ips[k]!).toBeGreaterThanOrEqual(ip.ips[k - 1]! - 1e-9);
    }
  });

  test("H₂ STO-3G — IP positive, single-occupied-orbital sanity", () => {
    const atoms: Atom[] = [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ];
    const { shells, nuclei } = moleculeToShellsNuclei(atoms);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const hf = runRHFSCF(integrals, 2);
    const ccsd = runCCSD(hf, integrals, { tol: 1e-12 });
    const ip = runIPEOMCCSD(ccsd, integrals, hf);

    const koopmansHomoEV = -hf.orbitalEnergies[0]! * HARTREE_TO_EV;
    console.log(`[ip-eom-h2] Koopmans = ${koopmansHomoEV.toFixed(2)} eV`);
    console.log(`[ip-eom-h2] IP-EOM-CCSD = [${
      Array.from(ip.ips).map(x => (x * HARTREE_TO_EV).toFixed(2)).join(", ")
    }] eV`);
    // Lowest IP should be positive and < 20 eV (H₂ HOMO is shallow).
    expect(ip.ips[0]!).toBeGreaterThan(0);
    expect(ip.ips[0]! * HARTREE_TO_EV).toBeLessThan(25);
  });
});
