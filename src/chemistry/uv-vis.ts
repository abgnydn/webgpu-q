// ─────────────────────────────────────────────────────────────
// uv-vis.ts — top-level UV-vis spectrum pipeline. Takes a list of
// atoms, runs HF + TDDFT (or HF + CIS or HF + TDHF, depending on
// opts), and returns:
//   - raw discrete excitations (energy, oscillator strength,
//     wavelength)
//   - a Gaussian-broadened continuous absorption spectrum
//   - identified peaks (local maxima)
//   - TRK sum rule check on the oscillator strengths
//
// One call from molecule to publication-ready UV-vis output.
// ─────────────────────────────────────────────────────────────

import type { Atom, AtomSymbol, BasisName } from "./atoms.js";
import { moleculeToShellsNuclei } from "./atoms.js";
import { computeMolecularIntegrals } from "./cg-molecular.js";
import { runRHFSCF } from "./hf-scf.js";
import { runRKSDFT } from "./dft/rks-scf.js";
import { runTDDFT, type TDAMethod } from "./tda-dft.js";
import {
  broadenSpectrum, findSpectrumPeaks, type SpectrumPoint,
} from "./spectrum.js";
import { trkSumRule, type TRKResult } from "./trk-sum-rule.js";
import { HARTREE_TO_EV, HARTREE_TO_NM } from "./units.js";

export interface UVVisOpts {
  /** Basis set. Default sto-3g. */
  readonly basis?: BasisName;
  /** Method: "hf" (TDHF / RPA) or a DFT functional name. Default "hf". */
  readonly method?: TDAMethod;
  /** Number of excited states to keep (lowest n). Default 10. */
  readonly nStates?: number;
  /** Gaussian broadening FWHM in eV for the spectrum. Default 0.3 eV. */
  readonly fwhm?: number;
  /** Energy grid step in eV. Default 0.01 eV. */
  readonly step?: number;
  /** Energy grid max in eV. Default = 1.5 × highest excitation. */
  readonly eMaxEv?: number;
}

export interface UVVisExcitation {
  /** Excitation energy in Hartree. */
  readonly energyHa: number;
  /** Excitation energy in eV. */
  readonly energyEv: number;
  /** Wavelength in nm. */
  readonly wavelengthNm: number;
  /** Dipole oscillator strength (dimensionless). */
  readonly oscillatorStrength: number;
}

export interface UVVisResult {
  readonly excitations: readonly UVVisExcitation[];
  /** Gaussian-broadened spectrum (energy in eV, absorption in
   *  arbitrary units). */
  readonly spectrum: readonly SpectrumPoint[];
  /** Peak energies in eV with their absorption values. */
  readonly peaks: readonly SpectrumPoint[];
  /** TRK sum-rule cross-check on the oscillator strengths. */
  readonly trk: TRKResult;
  /** SCF total energy (Ha). */
  readonly scfEnergy: number;
}

export function uvVisSpectrum(
  atoms: readonly Atom[],
  opts: UVVisOpts = {},
): UVVisResult {
  const basis = opts.basis ?? "sto-3g";
  const method = opts.method ?? "hf";
  const nStates = opts.nStates ?? 10;
  const fwhm = opts.fwhm ?? 0.3;
  const step = opts.step ?? 0.01;

  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei, { skipOAO: true });
  const nucleiSymbols = atoms.map((a) => a.symbol) as AtomSymbol[];

  // Run SCF.
  let hfRef;
  let scfEnergy: number;
  if (method === "hf") {
    const hf = runRHFSCF(integrals, nElectrons, {
      useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
    });
    hfRef = hf;
    scfEnergy = hf.energy;
  } else {
    const ks = runRKSDFT(integrals, nElectrons, nucleiSymbols, {
      functional: method,
      useDIIS: true, maxIter: 200, energyTol: 1e-9, residualTol: 1e-7,
    });
    hfRef = ks;
    scfEnergy = ks.energy;
  }

  // Run TDDFT.
  const tddft = runTDDFT(integrals, hfRef, {
    method, spin: "singlet", nucleiSymbols, nRoots: nStates,
  });

  // Build excitation list (use indices < nStates, pruning the array).
  const excList: UVVisExcitation[] = [];
  for (let n = 0; n < Math.min(tddft.singletEnergies.length, nStates); n++) {
    const Eha = tddft.singletEnergies[n]!;
    excList.push({
      energyHa: Eha,
      energyEv: Eha * HARTREE_TO_EV,
      wavelengthNm: HARTREE_TO_NM(Eha),
      oscillatorStrength: tddft.oscillatorStrengths[n]!,
    });
  }

  // Build broadened spectrum in eV.
  const excEv = excList.map((e) => ({
    energy: e.energyEv,
    oscillatorStrength: e.oscillatorStrength,
  }));
  const maxEv = excList[excList.length - 1]?.energyEv ?? 1;
  const eMaxEv = opts.eMaxEv ?? Math.max(maxEv * 1.5, 1);
  const spectrum = broadenSpectrum(excEv, { eMin: 0, eMax: eMaxEv, step, fwhm });
  const peaks = findSpectrumPeaks(spectrum, 0.01);

  // TRK check.
  const trk = trkSumRule(
    new Float64Array(excList.map((e) => e.oscillatorStrength)),
    nElectrons,
  );

  return { excitations: excList, spectrum, peaks, trk, scfEnergy };
}
