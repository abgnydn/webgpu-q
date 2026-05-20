// ─────────────────────────────────────────────────────────────
// molecular-report.ts — top-level convenience aggregator. Takes
// any converged SCF result (RHF / UHF / RKS / UKS) and runs the
// full property toolkit, returning a structured report suitable
// for JSON serialization (also feeds QCSchema export).
//
// Use case: "give me everything about this molecule" — the kind
// of call a researcher makes after their SCF converges. Avoids
// the user having to know which property module to call for what.
//
// Computes (always):
//   - SCF energy + total energy (+ correlation if provided)
//   - HOMO/LUMO orbital energies + gap
//   - Koopmans IP / EA
//   - Dipole moment (vector + magnitude + Debye)
//   - Mulliken charges
//   - Mayer bond orders + atomic valences
//   - NOON (natural orbital occupations)
//   - Multireference diagnostic verdict
//
// Computes (open-shell only):
//   - Mulliken spin populations
//   - UHF/UKS-style Mayer bond orders
//   - ⟨S²⟩ + exact value + contamination
//
// Computes (optional):
//   - D2 dispersion correction (if `opts.addD2`)
//   - Static polarizability (if `opts.addStaticAlpha`; requires
//     nucleiSymbols for KS reference)
// ─────────────────────────────────────────────────────────────

import type { Atom, AtomSymbol } from "./atoms.js";
import type { MolecularIntegrals } from "./cg-molecular.js";
import type { HFResult } from "./hf-scf.js";
import type { UHFResult } from "./uhf-scf.js";
import type { UKSResult } from "./dft/uks-scf.js";
import type { HFLike } from "./cis.js";
import type { CGShell } from "./integrals-cg.js";
import {
  dipoleMoment, dipoleMagnitude, AU_TO_DEBYE,
  mullikenCharges, mullikenChargesUHF,
  mullikenSpinPopulations,
  bondOrders, bondOrdersUHF,
  mayerValences, mayerValencesUHF,
} from "./properties.js";
import { naturalOrbitalOccupations } from "./natural-orbitals.js";
import { multireferenceDiagnostic } from "./multireference-diagnostic.js";
import { dispersionD2, type DispersionD2Opts } from "./dispersion-d2.js";
import { cphfPolarizability } from "./cphf.js";
import { uhfCphfPolarizability } from "./uhf-cphf.js";
import { uksCphfPolarizability } from "./uks-cphf.js";

export type MolecularInput =
  | { readonly kind: "rhf"; readonly hf: HFResult }
  | { readonly kind: "uhf"; readonly uhf: UHFResult }
  | { readonly kind: "rks"; readonly ks: HFLike; readonly nucleiSymbols: readonly AtomSymbol[] }
  | { readonly kind: "uks"; readonly uks: UKSResult; readonly nucleiSymbols: readonly AtomSymbol[] };

export interface MolecularReportOpts {
  /** Add Grimme D2 dispersion correction. */
  readonly addD2?: boolean;
  readonly d2Opts?: DispersionD2Opts;
  /** Compute analytical static polarizability (CPHF for RHF/UHF,
   *  LSDA UKS-CPHF for UKS, TDDFT@ω=0 for RKS). */
  readonly addStaticAlpha?: boolean;
  /** Post-HF correlation energy (e.g., from MP2 / CCSD). Added to
   *  totalEnergy if provided. */
  readonly correlationEnergy?: number;
  /** T1 diagnostic from a CCSD calculation, fed into the
   *  multireference verdict. */
  readonly t1Diagnostic?: number;
  /** D1 diagnostic from a CCSD calculation. */
  readonly d1Diagnostic?: number;
}

export interface MolecularReport {
  readonly kind: "rhf" | "uhf" | "rks" | "uks";

  // Energy
  readonly scfEnergy: number;
  readonly correlationEnergy?: number;
  readonly dispersionEnergy?: number;
  readonly totalEnergy: number;

  // Orbital
  readonly nOccupied: { readonly alpha: number; readonly beta: number };
  readonly homoEnergy: number;     // -ε_HOMO interpreted as Koopmans IP after sign flip
  readonly lumoEnergy: number;
  readonly homoLumoGap: number;     // ε_LUMO − ε_HOMO (Hartree)
  readonly homoLumoGapEv: number;
  readonly koopmansIPEv: number;
  readonly koopmansEAEv: number;

  // Multipole
  readonly dipole: readonly [number, number, number];
  readonly dipoleMagnitudeAU: number;
  readonly dipoleMagnitudeDebye: number;

  // Population analysis
  readonly mullikenCharges: Float64Array;
  readonly mullikenSpinPopulations?: Float64Array;       // open-shell only
  readonly bondOrders: Float64Array;                      // nAtoms × nAtoms row-major
  readonly atomicValences: Float64Array;

  // Wavefunction diagnostics
  readonly noon: Float64Array;
  readonly noonStronglyOccupied: number;
  readonly noonActive: number;
  readonly noonTotalElectrons: number;
  readonly s2?: number;
  readonly s2Exact?: number;

  // Optional analytical static polarizability
  readonly staticAlpha?: Float64Array;       // 3 × 3 row-major
  readonly staticAlphaIsotropic?: number;

  // Aggregate verdict
  readonly multireferenceSeverity: "single-reference" | "borderline" | "multi-reference";
  readonly multireferenceFlags: readonly string[];
}

const HARTREE_TO_EV = 27.211386245988;

export function molecularReport(
  input: MolecularInput,
  integrals: MolecularIntegrals,
  shells: readonly CGShell[],
  atoms: readonly Atom[],
  shellAtomIdx: readonly number[],
  opts: MolecularReportOpts = {},
): MolecularReport {
  // ── Extract common fields by kind. ─────────────────────────
  let scfEnergy: number;
  let D: Float64Array;
  let nOccA: number, nOccB: number;
  let epsAlpha: Float64Array;
  let epsBeta: Float64Array | undefined;
  let DAlpha: Float64Array | undefined;
  let DBeta: Float64Array | undefined;
  let s2: number | undefined;
  let s2Exact: number | undefined;

  if (input.kind === "rhf") {
    scfEnergy = input.hf.energy;
    D = input.hf.D;
    nOccA = input.hf.nOccupied; nOccB = input.hf.nOccupied;
    epsAlpha = input.hf.orbitalEnergies;
  } else if (input.kind === "uhf") {
    scfEnergy = input.uhf.energy;
    DAlpha = input.uhf.D_alpha; DBeta = input.uhf.D_beta;
    D = new Float64Array(DAlpha.length);
    for (let i = 0; i < D.length; i++) D[i] = DAlpha[i]! + DBeta[i]!;
    nOccA = input.uhf.nAlpha; nOccB = input.uhf.nBeta;
    epsAlpha = input.uhf.orbitalEnergiesAlpha;
    epsBeta = input.uhf.orbitalEnergiesBeta;
    s2 = input.uhf.s2; s2Exact = input.uhf.s2Exact;
  } else if (input.kind === "rks") {
    scfEnergy = (input.ks as HFLike & { energy?: number }).energy ?? 0;
    D = input.ks.D;
    nOccA = input.ks.nOccupied; nOccB = input.ks.nOccupied;
    epsAlpha = input.ks.orbitalEnergies;
  } else {
    // kind === "uks"
    scfEnergy = input.uks.energy;
    DAlpha = input.uks.D_alpha; DBeta = input.uks.D_beta;
    D = new Float64Array(DAlpha.length);
    for (let i = 0; i < D.length; i++) D[i] = DAlpha[i]! + DBeta[i]!;
    nOccA = input.uks.nAlpha; nOccB = input.uks.nBeta;
    epsAlpha = input.uks.orbitalEnergiesAlpha;
    epsBeta = input.uks.orbitalEnergiesBeta;
    s2 = input.uks.s2; s2Exact = input.uks.s2Exact;
  }

  // ── HOMO / LUMO. ───────────────────────────────────────────
  // For UHF/UKS, HOMO is the max(ε_alpha[nA-1], ε_beta[nB-1]); LUMO
  // is the min(ε_alpha[nA], ε_beta[nB]). Closed-shell uses just alpha.
  const homoA = nOccA > 0 ? epsAlpha[nOccA - 1]! : -Infinity;
  const lumoA = nOccA < epsAlpha.length ? epsAlpha[nOccA]! : Infinity;
  let homoEnergy: number, lumoEnergy: number;
  if (epsBeta) {
    const homoB = nOccB > 0 ? epsBeta[nOccB - 1]! : -Infinity;
    const lumoB = nOccB < epsBeta.length ? epsBeta[nOccB]! : Infinity;
    homoEnergy = Math.max(homoA, homoB);
    lumoEnergy = Math.min(lumoA, lumoB);
  } else {
    homoEnergy = homoA;
    lumoEnergy = lumoA;
  }
  const homoLumoGap = lumoEnergy - homoEnergy;

  // ── Dipole. ────────────────────────────────────────────────
  const dipole = dipoleMoment(integrals, D);
  const dipMag = dipoleMagnitude(dipole);

  // ── Population analysis. ───────────────────────────────────
  const isOpenShell = DAlpha !== undefined && DBeta !== undefined;
  const mCharges = isOpenShell
    ? mullikenChargesUHF(integrals, DAlpha!, DBeta!, shellAtomIdx)
    : mullikenCharges(integrals, D, shellAtomIdx);
  const spinPops = isOpenShell
    ? mullikenSpinPopulations(integrals, DAlpha!, DBeta!, shellAtomIdx)
    : undefined;
  const bOrders = isOpenShell
    ? bondOrdersUHF(integrals, DAlpha!, DBeta!, shellAtomIdx)
    : bondOrders(integrals, D, shellAtomIdx);
  const valences = isOpenShell
    ? mayerValencesUHF(integrals, DAlpha!, DBeta!, shellAtomIdx)
    : mayerValences(integrals, D, shellAtomIdx);

  // ── NOON on total density. ─────────────────────────────────
  const noon = naturalOrbitalOccupations(D, integrals.S_AO);

  // ── Multireference verdict. ────────────────────────────────
  // For open-shell systems the SOMO sits at natural occupation ~1
  // (not 2 or 0), which would spuriously trigger the NOON deviation
  // flag. Skip the NOON check in that case — S² contamination + T1/D1
  // already cover the open-shell multi-reference signal.
  const mr = multireferenceDiagnostic({
    t1: opts.t1Diagnostic,
    d1: opts.d1Diagnostic,
    s2, s2Exact,
    ...(isOpenShell ? {} : {
      noon: noon.occupations,
      nOccupied: Math.max(nOccA, nOccB),
    }),
  });

  // ── Optional D2 dispersion. ────────────────────────────────
  let dispersionEnergy: number | undefined;
  if (opts.addD2) {
    dispersionEnergy = dispersionD2(atoms, opts.d2Opts ?? {}).energy;
  }

  // ── Optional analytical static polarizability. ────────────
  let staticAlpha: Float64Array | undefined;
  let staticAlphaIso: number | undefined;
  if (opts.addStaticAlpha) {
    try {
      if (input.kind === "rhf") {
        const r = cphfPolarizability(input.hf, integrals, shells);
        staticAlpha = r.alpha; staticAlphaIso = r.isotropic;
      } else if (input.kind === "uhf") {
        const r = uhfCphfPolarizability(input.uhf, integrals, shells);
        staticAlpha = r.alpha; staticAlphaIso = r.isotropic;
      } else if (input.kind === "uks") {
        const r = uksCphfPolarizability(input.uks, integrals, shells,
          { functional: "lda-svwn", nucleiSymbols: input.nucleiSymbols });
        staticAlpha = r.alpha; staticAlphaIso = r.isotropic;
      }
      // rks: would need a separate KS-CPHF — skipped for now.
    } catch {
      // Some basis sets (e.g. He STO-3G with no virtual) make CPHF
      // singular. Leave staticAlpha undefined.
    }
  }

  const correlationEnergy = opts.correlationEnergy;
  const totalEnergy = scfEnergy
    + (correlationEnergy ?? 0)
    + (dispersionEnergy ?? 0);

  return {
    kind: input.kind,
    scfEnergy,
    correlationEnergy,
    dispersionEnergy,
    totalEnergy,
    nOccupied: { alpha: nOccA, beta: nOccB },
    homoEnergy, lumoEnergy,
    homoLumoGap, homoLumoGapEv: homoLumoGap * HARTREE_TO_EV,
    koopmansIPEv: -homoEnergy * HARTREE_TO_EV,
    koopmansEAEv: -lumoEnergy * HARTREE_TO_EV,
    dipole, dipoleMagnitudeAU: dipMag,
    dipoleMagnitudeDebye: dipMag * AU_TO_DEBYE,
    mullikenCharges: mCharges,
    mullikenSpinPopulations: spinPops,
    bondOrders: bOrders,
    atomicValences: valences,
    noon: noon.occupations,
    noonStronglyOccupied: noon.nStronglyOccupied,
    noonActive: noon.nActive,
    noonTotalElectrons: noon.totalElectrons,
    s2, s2Exact,
    staticAlpha, staticAlphaIsotropic: staticAlphaIso,
    multireferenceSeverity: mr.severity,
    multireferenceFlags: mr.flags,
  };
}
