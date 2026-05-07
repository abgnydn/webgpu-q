// ─────────────────────────────────────────────────────────────
// E17 — Electron-impact cross sections for water from a TDDFT /
// CIS spectrum, compared against published reference tables.
// Tier 2 stage 14.
//
// Hypothesis (per protocol):
//   For electron-on-water at T ∈ {100, 300, 1000, 3000, 10000} eV,
//   total ionization and electronic-excitation cross sections
//   derived from a Bethe-Born sum over the molecular oscillator-
//   strength distribution agree with published Itikawa-Mason 2005
//   reference values (the same values Geant4-DNA's G4EMLOW data
//   tables are built from) within ±15 % per cell.
//
// Method:
//   1. Run RKS-DFT B3LYP5 on H₂O at the experimental geometry
//      (R_OH = 0.9572 Å, ∠HOH = 104.52°). STO-3G + cc-pVDZ both,
//      to show the basis-set trend.
//   2. Run TDA on the converged KS reference. Get every singlet
//      excitation energy ω_n + oscillator strength f_n.
//   3. Bethe-Born sum: σ_inel(T) = (4π/T)·Σ_n (f_n/ω_n)·ln(αT/ω_n),
//      α = 4 (Inokuti-Itikawa standard), all in atomic units.
//   4. Partition by ionization potential I_p(H₂O) = 12.62 eV:
//      σ_exc = states with ω_n < I_p; σ_ion = states with ω_n > I_p.
//   5. Compare to Itikawa-Mason 2005 σ_ion + σ_exc tabulations
//      at the 5 reference energies (cm² → a₀² with `BOHR2_TO_CM2`).
//
// Honest scope notes (post-run, observed 2026-05-07):
//   • The TRK sum rule IS satisfied at cc-pVDZ — Σf ≈ 10.77 vs
//     N_e = 10. The total oscillator-strength budget is right.
//   • But the spectral DISTRIBUTION is wrong: a finite TDA-DFT
//     spectrum piles oscillator strength into discrete states close
//     to I_p, while the true distribution spreads it over the
//     continuum (ω → +∞). Bethe-Born's (f/ω)·ln(αT/ω) weight
//     amplifies low-ω contributions, so finite-state spectra
//     OVERSHOOT σ_ion by 2-6× (STO-3G→cc-pVDZ at the 5 ref
//     energies). σ_exc undershoots in STO-3G (only 1 state below
//     I_p) and overshoots in cc-pVDZ (4 states below I_p).
//   • Bethe asymptotic form is leading-log; at T ~ ω_n the Born
//     approximation itself breaks down. Results below ~ 100 eV
//     are not trustworthy at the ±15 % level even with a
//     converged spectrum.
//   • The published Itikawa-Mason 2005 values used here are hand-
//     tabulated from the paper; the actual G4EMLOW 8.8 tables in
//     webgpu-dna are upstream-equivalent but not re-derived here.
//   • For a converged comparison to Geant4-DNA / Itikawa-Mason,
//     we'd need a continuum representation: Stieltjes imaging,
//     SAC-CI in a continuum basis, B-spline / DVR continuum
//     orbitals, or direct optical-photoionization cross sections.
//     All genuinely substantial follow-ups.
//
// Pass bar: |σ_ours − σ_ref| / σ_ref ≤ 0.15 in each of 10 cells.
// FAIL is the documented expected outcome — ship as honest negative.
// ─────────────────────────────────────────────────────────────

import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { runRKSDFT } from "../../src/chemistry/dft/rks-scf.js";
import { runTDA } from "../../src/chemistry/tda-dft.js";
import {
  betheBornCrossSection,
  HARTREE_TO_EV,
  BOHR2_TO_CM2,
} from "../../src/chemistry/cross-sections.js";
import { initGPU } from "../../src/quantum.js";
import { captureEnv } from "../lib/env.js";
import type { Artifact, ArtifactMeta } from "../lib/runner.js";

/** Reference cross sections for e⁻ + H₂O (Itikawa-Mason 2005, J. Phys.
 *  Chem. Ref. Data 34, 1; the same data Geant4-DNA's G4EMLOW is built
 *  from). Total ionization and total electronic excitation, in cm². */
const ITIKAWA_2005_H2O: Record<number, { sigmaIon_cm2: number; sigmaExc_cm2: number }> = {
  100:   { sigmaIon_cm2: 2.21e-16, sigmaExc_cm2: 6.6e-17 },
  300:   { sigmaIon_cm2: 1.10e-16, sigmaExc_cm2: 2.6e-17 },
  1000:  { sigmaIon_cm2: 4.42e-17, sigmaExc_cm2: 8.7e-18 },
  3000:  { sigmaIon_cm2: 1.84e-17, sigmaExc_cm2: 3.0e-18 },
  10000: { sigmaIon_cm2: 6.49e-18, sigmaExc_cm2: 9.0e-19 },
};

/** Vertical ionization potential of H₂O (Lias et al. NIST, 12.62 eV). */
const H2O_IP_EV = 12.62;
const H2O_IP_HA = H2O_IP_EV / HARTREE_TO_EV;

const REFERENCE_ENERGIES_EV = [100, 300, 1000, 3000, 10000] as const;

const PASS_TOLERANCE = 0.15;     // ±15 % per cell

export interface E17Row {
  readonly basis: "sto-3g" | "cc-pvdz";
  readonly impactEnergyEV: number;
  /** σ_ion derived from our TDA spectrum (cm²). */
  readonly sigmaIon_cm2: number;
  /** σ_exc derived from our TDA spectrum (cm²). */
  readonly sigmaExc_cm2: number;
  /** Itikawa-Mason 2005 reference σ_ion (cm²). */
  readonly refSigmaIon_cm2: number;
  /** Itikawa-Mason 2005 reference σ_exc (cm²). */
  readonly refSigmaExc_cm2: number;
  /** |σ_ours − σ_ref| / σ_ref. */
  readonly relErrorIon: number;
  readonly relErrorExc: number;
  /** State counts that contributed to the ionization / excitation sums. */
  readonly nIonStates: number;
  readonly nExcStates: number;
}

export interface E17Summary {
  readonly basis: "sto-3g" | "cc-pvdz";
  readonly nSingletStates: number;
  readonly maxRelErrorIon: number;
  readonly maxRelErrorExc: number;
  readonly cellsWithinTolerance: number;     // out of 10
}

/** Build the H₂O equilibrium-geometry atom set. */
function h2oAtoms(): Atom[] {
  const half = (104.52 / 2) * Math.PI / 180;
  const x = 0.9572 * Math.sin(half);
  const z = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ x, 0, z] },
    { symbol: "H", pos: [-x, 0, z] },
  ];
}

export async function runE17(): Promise<Artifact<E17Row> & { summaries: E17Summary[] }> {
  // Env block — chemistry is CPU-only but we record GPU info to keep
  // artifact provenance uniform with the rest of the level ladder.
  const device = await initGPU();
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("E17: no adapter for env block");
  const env = await captureEnv(device, adapter);

  const atoms = h2oAtoms();
  const symbols = atoms.map((a) => a.symbol);

  const rows: E17Row[] = [];
  const summaries: E17Summary[] = [];

  for (const basis of ["sto-3g", "cc-pvdz"] as const) {
    console.log(`[E17] Building integrals (${basis}) + B3LYP5 SCF + TDA …`);
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
    // NOTE: cc-pVDZ uses Cartesian-d here (no spherical transform) — the
    // spherical-d MO transform desynchronizes the shells/MO basis with
    // the TDA-DFT XC-kernel grid path. Cartesian gives ~4 mHa larger
    // total energy (well-known slack), but the TDA spectrum is intact.
    const integrals = computeMolecularIntegrals(shells, nuclei);
    const dft = runRKSDFT(integrals, nElectrons, symbols, {
      functional: "b3lyp5", energyTol: 1e-10, maxIter: 200,
    });
    const tda = runTDA(integrals, dft, {
      method: "b3lyp5", nucleiSymbols: symbols,
    });
    const omegas = tda.singletEnergies;
    const f = tda.oscillatorStrengths;
    console.log(`[E17] ${basis}: ${omegas.length} singlet states, ` +
                `Σf = ${Array.from(f).reduce((a, b) => a + b, 0).toFixed(3)}`);

    let maxIon = 0, maxExc = 0, withinCells = 0;
    for (const T_eV of REFERENCE_ENERGIES_EV) {
      const T_Ha = T_eV / HARTREE_TO_EV;
      const cs = betheBornCrossSection(
        { excitationEnergiesHa: omegas, oscillatorStrengths: f, ionizationPotentialHa: H2O_IP_HA },
        T_Ha,
      );
      const ion_cm2 = cs.sigmaIon * BOHR2_TO_CM2;
      const exc_cm2 = cs.sigmaExc * BOHR2_TO_CM2;
      const ref = ITIKAWA_2005_H2O[T_eV]!;
      const errIon = Math.abs(ion_cm2 - ref.sigmaIon_cm2) / ref.sigmaIon_cm2;
      const errExc = Math.abs(exc_cm2 - ref.sigmaExc_cm2) / ref.sigmaExc_cm2;
      if (errIon > maxIon) maxIon = errIon;
      if (errExc > maxExc) maxExc = errExc;
      if (errIon <= PASS_TOLERANCE) withinCells++;
      if (errExc <= PASS_TOLERANCE) withinCells++;
      rows.push({
        basis,
        impactEnergyEV: T_eV,
        sigmaIon_cm2: ion_cm2,
        sigmaExc_cm2: exc_cm2,
        refSigmaIon_cm2: ref.sigmaIon_cm2,
        refSigmaExc_cm2: ref.sigmaExc_cm2,
        relErrorIon: errIon,
        relErrorExc: errExc,
        nIonStates: cs.nIon,
        nExcStates: cs.nExc,
      });
      console.log(
        `[E17] ${basis} T=${T_eV} eV: σ_ion=${ion_cm2.toExponential(3)} cm² ` +
        `(ref ${ref.sigmaIon_cm2.toExponential(3)}, ${(errIon * 100).toFixed(1)} %), ` +
        `σ_exc=${exc_cm2.toExponential(3)} cm² ` +
        `(ref ${ref.sigmaExc_cm2.toExponential(3)}, ${(errExc * 100).toFixed(1)} %)`,
      );
    }
    summaries.push({
      basis,
      nSingletStates: omegas.length,
      maxRelErrorIon: maxIon,
      maxRelErrorExc: maxExc,
      cellsWithinTolerance: withinCells,
    });
  }

  const allCells = REFERENCE_ENERGIES_EV.length * 2;     // 5 energies × {ion, exc}
  const totalWithin = summaries.reduce((acc, s) => acc + s.cellsWithinTolerance, 0);
  const possible = summaries.length * allCells;
  // Pass requires every cell within tolerance — which we don't expect.
  const status: Artifact<E17Row>["status"] =
    totalWithin === possible ? "pass" : "fail";

  const meta: ArtifactMeta = {
    protocol: "E17-cross-sections",
    hypothesis:
      "Electron-impact total ionization and electronic-excitation cross " +
      "sections for H₂O at T ∈ {100, 300, 1000, 3000, 10000} eV, derived " +
      "from a TDA-B3LYP5 oscillator-strength sum via Bethe-Born, agree " +
      "with Itikawa-Mason 2005 reference values within ±15 %.",
    passBar: "|σ_ours − σ_ref| / σ_ref ≤ 0.15 in every (energy, channel) cell",
    seed: "E17_CROSS_SECTIONS",
    warmup: 0,
    trials: 1,
  };

  // Diagnosis text — be explicit about what failed and why.
  const diagBits: string[] = [];
  for (const s of summaries) {
    const ionPct = (s.maxRelErrorIon * 100).toFixed(0);
    const excPct = (s.maxRelErrorExc * 100).toFixed(0);
    diagBits.push(
      `${s.basis}: ${s.nSingletStates} singlet states, ` +
      `${s.cellsWithinTolerance}/${allCells} cells within ±15 %, ` +
      `worst σ_ion error ${ionPct} %, worst σ_exc error ${excPct} %`,
    );
  }
  const honest =
    "expected (observed 2026-05-07): the TRK sum rule is satisfied " +
    "at cc-pVDZ (Σf ≈ 10.77 vs N_e = 10) but the discrete-state TDA " +
    "spectrum piles oscillator strength near I_p — Bethe-Born's " +
    "(f/ω)·ln(αT/ω) integrand weights low-ω heavily, so σ_ion " +
    "OVERSHOOTS Itikawa-Mason 2005 by 2-6×. σ_exc undershoots in " +
    "STO-3G (only 1 state below I_p) and overshoots in cc-pVDZ " +
    "(4 states below I_p). Genuine continuum representation " +
    "(Stieltjes imaging, SAC-CI, B-spline continuum) is needed — " +
    "substantial new infrastructure, deferred.";
  const diagnosis =
    status === "pass"
      ? `All cells within ±15 %. ${diagBits.join("; ")}.`
      : `Pass bar missed (as documented). ${diagBits.join("; ")}. ${honest}`;

  return { meta, env, rows, status, diagnosis, summaries };
}
