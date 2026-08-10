// Phase 2 (docs/RUN-PLAN-24H-ELEMENTS.md): analytical HF gradient vs
// central finite differences, for the ten elements added in this branch
// (B, Ne, Na, Mg, Al, Si, P, S, Cl, Ar).
//
// Geometries come from `pyscf-reference.json` — the same fixture that
// pins the RHF energies — so there is no second transcription of any
// coordinate.
//
// ── What bar is achievable, and why it is not 1e-6 ──────────────────
//
// The run plan proposed |analytic − FD| ≤ 1e-6 Ha/Bohr at h = 1e-4 Bohr.
// Measured (M2 Pro, STO-3G, SCF energyTol 1e-13 / densityTol 1e-12):
//
//   B/BH₃ 1.25e-7  Na/NaH 2.88e-7  Mg/MgH₂ 3.63e-7  Al/AlH₃ 3.00e-7
//   Si/SiH₄ 4.42e-7  P/PH₃ 1.245e-6  S/H₂S 2.57e-7  Cl/HCl 2.30e-7
//   (Ne, Ar are single atoms: exactly 0 by symmetry)
//
// For calibration, the eight elements that were already settled and
// PySCF-validated before this branch sit in the SAME band under the
// identical harness — H/H₂ 2.81e-7, Li/LiH 9.01e-8, Be/BeH₂ 3.60e-7,
// B/BH₃ 1.25e-7, C/CH₄ 1.45e-7, N/NH₃ 4.85e-7, O/H₂O 9.97e-8,
// F/HF 7.06e-7 — so nothing here is specific to the new elements.
//
// Nine of ten clear 1e-6; P/PH₃ measures 1.245e-6. That excess is NOT a
// gradient error — it is the finite-difference REFERENCE being the less
// accurate of the two objects:
//
//   • The FD is not noisy. An h-sweep on PH₃'s worst component gives
//     FD − analytic = 1.25e-6 at h = 3e-4, 1.24e-6 at 1e-4, 1.25e-6 at
//     3e-5, 1.24e-6 at 1e-5 — a clean plateau, so shrinking h does not
//     help and there is no round-off blow-up to hide behind.
//   • Cross-checked against PySCF's own analytic RHF gradient at the
//     same geometries and basis, our analytic gradient is within
//     4.8e-7 Ha/Bohr for every element (P/PH₃: 2.56e-7), whereas the
//     central FD of OUR energy is within only 1.5e-6 (P/PH₃: 1.50e-6).
//     The analytic gradient is ~6× closer to truth than the reference
//     it is being measured against.
//   • Root cause: our SCF energy carries an integral-precision error of
//     1e-7–8e-7 Ha relative to PySCF at these geometries. Differentiating
//     the energy differentiates that error too, giving a ~1e-6 Ha/Bohr
//     floor on any FD reference built from our own energies. Tightening
//     the SCF does not move it (loose 1e-11 and tight 1e-12 densityTol
//     give the identical 1.245e-6), and neither does disabling the
//     gradient's Schwarz screening.
//
// So the bar below is set at the measured capability of the REFERENCE
// (1.5e-6), not loosened to rescue the gradient. The sharp check on the
// gradient itself is translational invariance, which is a property of the
// analytical expression alone and is asserted at 1e-12 (measured ≤ 3.6e-14).
//
// That 1e-7–8e-7 Ha energy error almost certainly has a name: `boys0` in
// integrals-cg.ts evaluates erf with the Abramowitz–Stegun 7.1.26 rational
// fit, whose measured worst-case relative error is 2.66e-7 (the file admits
// ~1.5e-7 at integrals-cg.ts:195). It is the same floor that makes the run
// plan's Level-A bar of 1e-9 Ha unreachable. Replacing erfApprox with an
// f64-accurate erf is the one change that would let every FD-based gradient
// gate in this repo — HF, DFT and CPHF alike — be tightened below 1e-6.
import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { computeMolecularIntegrals } from "../../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom, type BasisName } from "../../../src/chemistry/atoms.js";
import { runRHFSCF } from "../../../src/chemistry/hf-scf.js";
import { hfGradient, buildEnergyWeightedDensity } from "../../../src/chemistry/hf-gradient.js";

/** Must match `atoms.ts` exactly, or the FD step is mis-scaled. */
const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

/** Central-difference step, in Bohr (run plan Gate 2). */
const FD_STEP_BOHR = 1e-4;

/** Measured FD-reference accuracy floor — see the header. */
const FD_BAR = 1.5e-6;

/** STO-3G reaches a 1e-12 density residual comfortably. */
const SCF_STO3G = {
  useDIIS: true, energyTol: 1e-13, densityTol: 1e-12, maxIter: 600,
} as const;

/** cc-pVDZ does NOT: H₂S and HCl (Cartesian 6d) stall above a 1e-12
 *  density residual and report `converged: false`. 1e-10 is the repo's
 *  standard heavy-basis tolerance and converges every case here. The
 *  looser SCF does not move the FD residual — measured identical to
 *  three digits on the cases that converge at both. */
const SCF_CCPVDZ = {
  useDIIS: true, energyTol: 1e-12, densityTol: 1e-10, maxIter: 600,
} as const;

function scfOpts(basis: BasisName): typeof SCF_STO3G | typeof SCF_CCPVDZ {
  return basis === "sto-3g" ? SCF_STO3G : SCF_CCPVDZ;
}

interface RefRow {
  element: string; molecule: string; basis: string; convention: string;
  atoms: { symbol: string; pos: number[] }[];
  ok: boolean;
}

const fixture = JSON.parse(
  readFileSync(new URL("./pyscf-reference.json", import.meta.url), "utf8"),
) as { rows: RefRow[] };

/** The ten elements this branch added. */
const NEW_ELEMENTS = ["B", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"] as const;

function fixtureRow(element: string): RefRow {
  const r = fixture.rows.find(
    (x) => x.element === element && x.basis === "sto-3g" && x.convention === "spherical",
  );
  if (!r) throw new Error(`no fixture row for ${element}`);
  return r;
}

function geometry(element: string): Atom[] {
  return fixtureRow(element).atoms.map((a) => ({
    symbol: a.symbol as Atom["symbol"],
    pos: [a.pos[0]!, a.pos[1]!, a.pos[2]!] as [number, number, number],
  }));
}

/** Analytical HF gradient (Ha/Bohr), flat [x0,y0,z0,x1,…]. */
function analyticalGradient(
  atoms: readonly Atom[], basis: BasisName, spherical: boolean,
): Float64Array {
  const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei, { spherical });
  const hf = runRHFSCF(integrals, nElectrons, scfOpts(basis));
  // A non-converged SCF is a failure, never a datum (run plan hard rule 2).
  if (!hf.converged) throw new Error("SCF did not converge");
  const W = buildEnergyWeightedDensity(hf.C_MO, hf.orbitalEnergies, hf.nOccupied, integrals.n);
  return hfGradient({
    shells, nuclei, shellAtomIdx, P: hf.D, W,
    // Spherical-d path: P / W arrive in the spherical-AO basis and must be
    // rotated back to Cartesian before contracting with derivative integrals.
    sphericalT: integrals.sphericalT,
  });
}

function energyAt(atoms: readonly Atom[], basis: BasisName, spherical: boolean): number {
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei, { spherical });
  const hf = runRHFSCF(integrals, nElectrons, scfOpts(basis));
  if (!hf.converged) throw new Error("SCF did not converge");
  return hf.energy;
}

function displaced(
  atoms: readonly Atom[], i: number, axis: number, deltaAngstrom: number,
): Atom[] {
  return atoms.map((a, k) => {
    if (k !== i) return a;
    const p: [number, number, number] = [a.pos[0], a.pos[1], a.pos[2]];
    p[axis] = (p[axis] ?? 0) + deltaAngstrom;
    return { ...a, pos: p };
  });
}

/** Central-FD gradient in Ha/Bohr; the step is applied in Bohr. */
function fdGradient(
  atoms: readonly Atom[], basis: BasisName, spherical: boolean, hBohr = FD_STEP_BOHR,
): Float64Array {
  const grad = new Float64Array(atoms.length * 3);
  const hAngstrom = hBohr / ANGSTROM_TO_BOHR;
  for (let i = 0; i < atoms.length; i++) {
    for (let axis = 0; axis < 3; axis++) {
      const ep = energyAt(displaced(atoms, i, axis, hAngstrom), basis, spherical);
      const em = energyAt(displaced(atoms, i, axis, -hAngstrom), basis, spherical);
      grad[i * 3 + axis] = (ep - em) / (2 * hBohr);
    }
  }
  return grad;
}

function worstComponent(
  a: Float64Array, b: Float64Array,
): { diff: number; index: number } {
  let diff = 0, index = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d > diff) { diff = d; index = i; }
  }
  return { diff, index };
}

describe("New elements: analytical HF gradient vs central FD (STO-3G)", () => {
  for (const element of NEW_ELEMENTS) {
    const row = fixtureRow(element);
    test(`${element} · ${row.molecule} · sto-3g: max |analytic − FD| ≤ ${FD_BAR} Ha/Bohr`, () => {
      const atoms = geometry(element);
      const an = analyticalGradient(atoms, "sto-3g", false);
      const fd = fdGradient(atoms, "sto-3g", false);
      const { diff, index } = worstComponent(an, fd);
      expect(
        diff,
        `${element}/${row.molecule} worst flat-component ${index}: ` +
        `analytic=${an[index]?.toExponential(6)} fd=${fd[index]?.toExponential(6)}`,
      ).toBeLessThan(FD_BAR);
    }, 300_000);
  }
});

describe("New elements: analytical HF gradient vs central FD (cc-pVDZ)", () => {
  // Two representative third-row cases at a basis that actually carries d
  // shells, so the d-derivative path and the spherical-d (5d) round-trip
  // through `hfGradient`'s `sphericalT` are both exercised. These cells are
  // 100-200 s each; that is why the cc-pVDZ set is deliberately small.
  //
  // Measured max |analytic − FD| at h = 1e-4:
  //   Cl/HCl  cart (n=24) 9.72e-7   sph (n=23) 9.89e-7
  //   S /H₂S  cart (n=29) 1.223e-6  sph (n=28) 1.210e-6
  const CASES: { element: string; spherical: boolean }[] = [
    { element: "Cl", spherical: false },
    { element: "Cl", spherical: true },
    { element: "S", spherical: false },
  ];
  for (const { element, spherical } of CASES) {
    const row = fixtureRow(element);
    const tag = spherical ? "spherical-d" : "cartesian-d";
    test(`${element} · ${row.molecule} · cc-pvdz · ${tag}: max |analytic − FD| ≤ ${FD_BAR} Ha/Bohr`, () => {
      const atoms = geometry(element);
      const an = analyticalGradient(atoms, "cc-pvdz", spherical);
      const fd = fdGradient(atoms, "cc-pvdz", spherical);
      const { diff, index } = worstComponent(an, fd);
      expect(
        diff,
        `${element}/${row.molecule} ${tag} worst flat-component ${index}: ` +
        `analytic=${an[index]?.toExponential(6)} fd=${fd[index]?.toExponential(6)}`,
      ).toBeLessThan(FD_BAR);
    }, 340_000);
  }
});

describe("New elements: gradient translational invariance (STO-3G)", () => {
  // Σ_atoms ∂E/∂R = 0 exactly, for any density — a property of the
  // analytical expression itself, independent of SCF or FD accuracy.
  // Measured worst case across the ten elements: 3.6e-14.
  for (const element of NEW_ELEMENTS) {
    const row = fixtureRow(element);
    test(`${element} · ${row.molecule}: |Σ_atoms ∇E| ≤ 1e-12 per axis`, () => {
      const atoms = geometry(element);
      const g = analyticalGradient(atoms, "sto-3g", false);
      for (let axis = 0; axis < 3; axis++) {
        let s = 0;
        for (let i = 0; i < atoms.length; i++) s += g[i * 3 + axis]!;
        expect(Math.abs(s), `axis ${axis}`).toBeLessThan(1e-12);
      }
    }, 120_000);
  }
});

describe("New elements: isolated-atom gradient vanishes by symmetry", () => {
  // Ne and Ar are the two single-atom cases. A free atom feels no force;
  // this is exact, so the bar is machine precision, not the FD floor.
  for (const element of ["Ne", "Ar"] as const) {
    test(`${element} atom: |∇E| = 0 in sto-3g and cc-pvdz, both conventions`, () => {
      const atoms = geometry(element);
      for (const basis of ["sto-3g", "cc-pvdz"] as const) {
        for (const spherical of [false, true]) {
          const g = analyticalGradient(atoms, basis, spherical);
          for (let i = 0; i < g.length; i++) {
            // Measured: identically 0.0, not merely small — the derivative
            // terms cancel exactly, component by component.
            expect(Math.abs(g[i]!), `${basis} ${spherical ? "sph" : "cart"} comp ${i}`)
              .toBeLessThan(1e-14);
          }
        }
      }
    }, 300_000);
  }
});
