// The molecules period 3 actually buys.
//
// Element coverage is only interesting if it unlocks chemistry people
// teach. These four are the payoff for adding Cl, S and P:
//
//   CH3Cl  — the SN2 substrate. Chloromethane is the canonical example
//            in every introductory organic course; Cl was the single
//            element blocking it.
//   CH3SH  — methanethiol. Real organosulfur, not a toy hydride.
//   H2S    — the sulfur analogue of water; the classic "why is H2S a gas
//            and H2O a liquid" comparison.
//   PH3    — phosphine; the entry point to phosphates and the DNA
//            backbone.
//
// Each is checked against PySCF at STO-3G, Cartesian convention, on
// identical geometries. Bar is the same 0.1 mHa used by the element
// agreement test.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../../src/chemistry/hf-scf.js";

const D = Math.PI / 180;

function ch3cl(): Atom[] {
  const th = 108.0 * D, ch = 1.087;
  const out: Atom[] = [
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "Cl", pos: [0, 0, 1.785] },
  ];
  for (let k = 0; k < 3; k++) {
    const phi = 120 * k * D;
    const r = ch * Math.sin(Math.PI - th);
    const z = ch * Math.cos(Math.PI - th);
    out.push({ symbol: "H", pos: [r * Math.cos(phi), r * Math.sin(phi), z] });
  }
  return out;
}

function ch3sh(): Atom[] {
  const out: Atom[] = [
    { symbol: "S", pos: [0, 0, 0] },
    { symbol: "C", pos: [0, 0, 1.819] },
    { symbol: "H", pos: [1.336 * Math.sin(96.5 * D), 0, -1.336 * Math.cos(96.5 * D)] },
  ];
  for (let k = 0; k < 3; k++) {
    const phi = 120 * k * D;
    out.push({ symbol: "H", pos: [0.63 * Math.cos(phi), 0.63 * Math.sin(phi), 2.18] });
  }
  return out;
}

function bent(sym: Atom["symbol"], bond: number, angleDeg: number): Atom[] {
  const h = (angleDeg / 2) * D;
  const x = bond * Math.sin(h), z = bond * Math.cos(h);
  return [
    { symbol: sym, pos: [0, 0, 0] },
    { symbol: "H", pos: [x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
}

function pyramidal(sym: Atom["symbol"], bond: number, angleDeg: number): Atom[] {
  const ha = (angleDeg / 2) * D;
  const r = bond * Math.sin(ha) / Math.sin(60 * D);
  const h = Math.sqrt(Math.max(bond * bond - r * r, 0));
  const out: Atom[] = [{ symbol: sym, pos: [0, 0, 0] }];
  for (let k = 0; k < 3; k++) {
    const a = (90 + 120 * k) * D;
    out.push({ symbol: "H", pos: [r * Math.cos(a), r * Math.sin(a), -h] });
  }
  return out;
}

// PySCF RHF/STO-3G, mol.cart = True, conv_tol 1e-12.
const CASES: { name: string; atoms: Atom[]; nao: number; ref: number }[] = [
  { name: "CH3Cl (SN2 substrate)", atoms: ch3cl(), nao: 17, ref: -493.509047901 },
  { name: "CH3SH (methanethiol)", atoms: ch3sh(), nao: 18, ref: -432.209032619 },
  { name: "H2S", atoms: bent("S", 1.3356, 92.11), nao: 11, ref: -394.311555763 },
  { name: "PH3 (phosphine)", atoms: pyramidal("P", 1.42, 93.5), nao: 12, ref: -338.633614015 },
];

describe("Curriculum molecules unlocked by period 3", () => {
  for (const c of CASES) {
    test(`${c.name}: RHF/STO-3G within 0.1 mHa of PySCF`, () => {
      const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(c.atoms, "sto-3g");
      const ints = computeMolecularIntegrals(shells, nuclei);
      expect(ints.n).toBe(c.nao);
      const hf = runRHFSCF(ints, nElectrons, {
        useDIIS: true, maxIter: 300, energyTol: 1e-11, densityTol: 1e-9,
      });
      // A non-converged SCF is a failure, never a datum.
      expect(hf.converged).toBe(true);
      expect(Math.abs((hf.energy - c.ref) * 1000)).toBeLessThan(0.1);
    }, 300_000);
  }
});
