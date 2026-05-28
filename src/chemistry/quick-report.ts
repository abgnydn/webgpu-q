// ─────────────────────────────────────────────────────────────
// quick-report.ts — top-level "give me everything about this
// molecule" convenience. Takes a list of atoms (positions in Å)
// and runs the full standard property suite in one call:
//
//   1. Determine open- vs closed-shell from electron count + opts
//   2. Run RHF (or UHF if open-shell)
//   3. Compute the comprehensive molecularReport
//   4. (Optionally) add D2 dispersion + analytical static α
//
// Output: a MolecularReport object ready for JSON serialization
// (or rendering in a UI). The fastest path from atoms → ALL
// properties — ideal for demos, tutorials, and quick exploration.
//
// For more control (different functional, basis, post-HF method,
// EOM-CCSD spectra, etc.), users should build the calculation
// step-by-step using the individual modules.
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "./atoms.js";
import { moleculeToShellsNuclei } from "./atoms.js";
import { computeMolecularIntegrals } from "./cg-molecular.js";
import { runRHFSCF, type HFOpts } from "./hf-scf.js";
import { runUHFSCF, type UHFOpts } from "./uhf-scf.js";
import { molecularReport, type MolecularReport } from "./molecular-report.js";

export interface QuickReportOpts {
  /** Basis set. Default sto-3g. */
  readonly basis?: BasisName;
  /** Number of α electrons (open-shell). If omitted, assumed closed-shell. */
  readonly nAlpha?: number;
  /** Number of β electrons (open-shell). Required iff nAlpha given. */
  readonly nBeta?: number;
  /** Forwarded HF / UHF SCF options. */
  readonly hf?: HFOpts;
  readonly uhf?: UHFOpts;
  /** Add D2 dispersion correction. */
  readonly addD2?: boolean;
  /** Compute analytical static polarizability. */
  readonly addStaticAlpha?: boolean;
}

export function quickReport(
  atoms: readonly Atom[],
  opts: QuickReportOpts = {},
): MolecularReport {
  const basis = opts.basis ?? "sto-3g";
  const { shells, nuclei, nElectrons, shellAtomIdx } = moleculeToShellsNuclei(atoms, basis);
  const integrals = computeMolecularIntegrals(shells, nuclei, { skipOAO: true });

  const openShell = opts.nAlpha !== undefined || opts.nBeta !== undefined;
  if (openShell && (opts.nAlpha === undefined || opts.nBeta === undefined)) {
    throw new Error("quickReport: both nAlpha and nBeta required for open-shell");
  }
  if (openShell && (opts.nAlpha! + opts.nBeta!) !== nElectrons) {
    throw new Error(
      `quickReport: nAlpha + nBeta = ${opts.nAlpha! + opts.nBeta!} ≠ ` +
      `nElectrons = ${nElectrons} (derived from atoms)`,
    );
  }

  const defaultHF: HFOpts = {
    useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
  };
  const defaultUHF: UHFOpts = {
    useDIIS: true, maxIter: 200, energyTol: 1e-10, densityTol: 1e-8,
  };

  if (openShell) {
    const uhf = runUHFSCF(integrals, opts.nAlpha!, opts.nBeta!, { ...defaultUHF, ...(opts.uhf ?? {}) });
    return molecularReport(
      { kind: "uhf", uhf }, integrals, shells, atoms, shellAtomIdx,
      { addD2: opts.addD2, addStaticAlpha: opts.addStaticAlpha },
    );
  }
  const hf = runRHFSCF(integrals, nElectrons, { ...defaultHF, ...(opts.hf ?? {}) });
  return molecularReport(
    { kind: "rhf", hf }, integrals, shells, atoms, shellAtomIdx,
    { addD2: opts.addD2, addStaticAlpha: opts.addStaticAlpha },
  );
}
