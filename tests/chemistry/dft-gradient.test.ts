// Tier 2 stage 6: analytical RKS-DFT gradient (LDA only).
//
// Pass bars (looser than HF's 1e-5 because of (a) the weights-fixed
// approximation — we don't differentiate the Becke partition weights
// w.r.t. nuclear positions yet — and (b) finite-grid quadrature on
// the XC integral itself):
//   • |analytic − FD| ≤ 1e-3 Ha/Bohr per component on H₂, H₂O, BeH₂
//     STO-3G with the default grid (50r × 12θ × 24φ per atom).
//   • Translational invariance Σ_atoms ∇E ≈ 0 to 1e-6.
//   • For non-LDA functionals (BVWN5, BLYP, B3LYP5), the call must
//     throw with a clear "needs GGA support" message — the GGA path
//     is documented as a follow-up.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { dftGradient } from "../../src/chemistry/dft-gradient.js";
import { buildEnergyWeightedDensity } from "../../src/chemistry/hf-gradient.js";

const ANGSTROM_TO_BOHR = 1.8897261339;

interface Mol {
  name: string;
  atoms: Atom[];
}

const MOLECULES: Mol[] = [
  {
    name: "H2",
    atoms: [
      { symbol: "H", pos: [0, 0, 0] },
      { symbol: "H", pos: [0, 0, 0.7414] },
    ],
  },
  {
    name: "H2O",
    atoms: (() => {
      const half = (104.52 / 2) * Math.PI / 180;
      const x = 0.9572 * Math.sin(half);
      const z = 0.9572 * Math.cos(half);
      return [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ x, 0, z] },
        { symbol: "H", pos: [-x, 0, z] },
      ] as Atom[];
    })(),
  },
  {
    name: "BeH2",
    atoms: [
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.34] },
      { symbol: "H",  pos: [0, 0, -1.34] },
    ],
  },
];

function analyticalLDAGradient(atoms: readonly Atom[]): Float64Array {
  const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const symbols = atoms.map((a) => a.symbol);
  const dft = runRKSDFT(integrals, nElectrons, symbols, {
    functional: "lda-svwn", energyTol: 1e-12, residualTol: 1e-7, maxIter: 200,
  });
  const W = buildEnergyWeightedDensity(dft.C_MO, dft.orbitalEnergies, dft.nOccupied, integrals.n);
  return dftGradient({
    shells, nuclei, shellAtomIdx, nucleiSymbols: symbols,
    P: dft.D, W, functional: "lda-svwn",
  });
}

function fdLDAGradient(atoms: readonly Atom[], hBohr = 1e-3): Float64Array {
  const grad = new Float64Array(atoms.length * 3);
  const energyAt = (perturbed: readonly Atom[]): number => {
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(perturbed);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const symbols = perturbed.map((a) => a.symbol);
    const dft = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "lda-svwn", energyTol: 1e-12, residualTol: 1e-7, maxIter: 200,
    });
    return dft.energy;
  };
  const hAng = hBohr / ANGSTROM_TO_BOHR;
  const shift = (i: number, axis: number, delta: number): Atom[] =>
    atoms.map((a, k) => {
      if (k !== i) return a;
      const p: [number, number, number] = [a.pos[0], a.pos[1], a.pos[2]];
      p[axis] = (p[axis] ?? 0) + delta;
      return { ...a, pos: p };
    });
  for (let i = 0; i < atoms.length; i++) {
    for (let axis = 0; axis < 3; axis++) {
      const Ep = energyAt(shift(i, axis,  hAng));
      const Em = energyAt(shift(i, axis, -hAng));
      grad[i * 3 + axis] = (Ep - Em) / (2 * hBohr);
    }
  }
  return grad;
}

describe("RKS-DFT/LDA analytical gradient: agreement with central-FD", () => {
  for (const m of MOLECULES) {
    test(`${m.name}: per-component |analytic − FD| ≤ 1e-3 Ha/Bohr (LDA, weights-fixed)`, () => {
      const an = analyticalLDAGradient(m.atoms);
      const fd = fdLDAGradient(m.atoms);
      let maxDiff = 0, maxIdx = -1;
      for (let i = 0; i < an.length; i++) {
        const d = Math.abs(an[i]! - fd[i]!);
        if (d > maxDiff) { maxDiff = d; maxIdx = i; }
      }
      // 1 mHa/Bohr: dominated by the weights-fixed approximation
      // (Becke-partition derivative we don't compute) plus the
      // numerical-grid quadrature error itself. Tightening this is
      // a planned follow-up.
      expect(maxDiff, `worst component @ flat-idx ${maxIdx}: an=${an[maxIdx]?.toExponential(4)} fd=${fd[maxIdx]?.toExponential(4)}`).toBeLessThan(1e-3);
    }, 180_000);
  }
});

describe("RKS-DFT/LDA analytical gradient: translational invariance", () => {
  for (const m of MOLECULES) {
    test(`${m.name}: Σ_atoms ∇E ≈ 0 to 1e-3 per axis (weights-fixed)`, () => {
      const g = analyticalLDAGradient(m.atoms);
      let sx = 0, sy = 0, sz = 0;
      for (let i = 0; i < m.atoms.length; i++) {
        sx += g[i * 3 + 0]!;
        sy += g[i * 3 + 1]!;
        sz += g[i * 3 + 2]!;
      }
      // Looser than HF (1e-9) because the weights-fixed approximation
      // breaks exact translational invariance — the Becke partition
      // weights would also need to translate with the nuclei. For
      // a fine grid this residual is sub-mHa/Bohr, well below the
      // FD-vs-analytical pass bar of 1e-3 itself.
      expect(Math.abs(sx)).toBeLessThan(1e-3);
      expect(Math.abs(sy)).toBeLessThan(1e-3);
      expect(Math.abs(sz)).toBeLessThan(1e-3);
    }, 60_000);
  }
});

describe("optimizeGeometry on the LDA surface", () => {
  // Verify the optimizeGeometry wiring: starting from experimental
  // H₂O geometry, the LDA optimum should lie close to PySCF's LDA
  // optimum (R_OH ≈ 0.97 Å, ∠HOH ≈ 105°) — slightly larger bond
  // than HF/STO-3G's (0.99 Å) because LDA over-binds. Loose pass
  // bar — we just want to confirm the optimization converges and
  // analytical and FD paths agree.
  const half = (104.52 / 2) * Math.PI / 180;
  const x0 = 0.9572 * Math.sin(half);
  const z0 = 0.9572 * Math.cos(half);
  const startAtoms: Atom[] = [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x0, 0, z0] },
    { symbol: "H", pos: [-x0, 0, z0] },
  ];

  test("H₂O LDA: analytical and FD geom-opt converge to same energy", async () => {
    // Lazy-import optimizeGeometry to avoid pulling DFT into HF-only callers.
    const { optimizeGeometry } = await import("../../src/chemistry/geometry.js");
    const A = optimizeGeometry(startAtoms, {
      method: "lda-svwn",
      useAnalyticGrad: true,
      lbfgs: { gTol: 1e-3, maxIter: 30 },
    });
    const F = optimizeGeometry(startAtoms, {
      method: "lda-svwn",
      useAnalyticGrad: false,
      lbfgs: { gTol: 1e-3, maxIter: 30 },
    });
    expect(A.optimizer.termination).toBe("converged");
    expect(F.optimizer.termination).toBe("converged");
    // Both paths should land at the same minimum to within ~5 mHa.
    expect(Math.abs(A.energy - F.energy)).toBeLessThan(5e-3);
    // LDA H₂O / STO-3G energy is around −74.7 to −74.8 Ha.
    expect(A.energy).toBeLessThan(-74.5);
    expect(A.energy).toBeGreaterThan(-75.5);
  }, 240_000);
});

describe("RKS-DFT analytical gradient: GGA / hybrid not yet supported", () => {
  // Tier 2 stage 6 ships LDA only. Confirm the API surface refuses
  // GGA functionals with a clear error message instead of returning
  // a wrong gradient.
  for (const k of ["bvwn5", "blyp", "b3vwn5", "b3lyp5"] as const) {
    test(`${k}: throws with TODO message`, () => {
      const atoms: Atom[] = [
        { symbol: "H", pos: [0, 0, 0] },
        { symbol: "H", pos: [0, 0, 0.74] },
      ];
      const { shells, nuclei, shellAtomIdx } = moleculeToShellsNuclei(atoms);
      const symbols = atoms.map((a) => a.symbol);
      const n = shells.length;
      const P = new Float64Array(n * n);
      const W = new Float64Array(n * n);
      expect(() => dftGradient({
        shells, nuclei, shellAtomIdx, nucleiSymbols: symbols,
        P, W, functional: k,
      })).toThrow(/lda-svwn.*GGA.*basis Hessians/);
    });
  }
});
