// ─────────────────────────────────────────────────────────────
// qcschema.ts — emit QCSchema-compatible JSON for a converged
// calculation. QCSchema is MolSSI's quantum-chemistry interchange
// standard (https://github.com/MolSSI/QCSchema).
//
// We emit an `AtomicResult` v1 record for the "energy" driver,
// matching the layout consumed by QCFractal, QCEngine, MolSSI's
// QCArchive, and downstream cheminformatics tooling (ASE,
// cclib, pymatgen).
//
// Schema (abbreviated):
//
//   {
//     "schema_name": "qcschema_output",
//     "schema_version": 1,
//     "molecule": {
//       "schema_name": "qcschema_molecule",
//       "schema_version": 2,
//       "symbols": ["O", "H", "H"],
//       "geometry": [x_O, y_O, z_O, x_H1, y_H1, z_H1, ...]  // bohr
//     },
//     "driver": "energy",
//     "model": { "method": "hf" | "uhf" | "mp2" | ..., "basis": "sto-3g" },
//     "return_result": <total energy>,
//     "properties": {
//       "return_energy": <total energy>,
//       "scf_total_energy": <SCF total>,
//       "scf_one_electron_energy": <h matrix trace contribution>,
//       "scf_two_electron_energy": <J/K contribution>,
//       "nuclear_repulsion_energy": <Vnn>,
//       "calcinfo_natom": N,
//       "calcinfo_nbasis": n,
//       "calcinfo_nmo": n_MOs,
//       "calcinfo_nalpha": n_α,
//       "calcinfo_nbeta": n_β
//     },
//     "success": true,
//     "extras": { ... free-form (we put webgpu-q-specific things here) }
//   }
//
// Scope:
//   - Energy driver only (no gradient / Hessian / properties driver).
//   - Closed-shell (RHF/RKS) and open-shell (UHF/UKS) entry points.
//   - Optional correlation energy + post-HF method tag for MP2/CCSD.
//
// References:
//   https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html
//   https://github.com/MolSSI/QCElemental
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "./atoms.js";
import { Z_FOR } from "./atoms.js";

const ANGSTROM_PER_BOHR = 1.8897261245650618;

export interface QCSchemaCommon {
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  readonly method: string;        // "hf" / "uhf" / "rks-b3lyp5" / "mp2" / "ccsd" / "ccsd(t)" / ...
  /** Optional one-electron energy. */
  readonly oneElectron?: number;
  /** Optional two-electron (J + K) energy. */
  readonly twoElectron?: number;
  /** Nuclear repulsion energy. */
  readonly nuclearRepulsion: number;
  /** Total energy (return_result). */
  readonly totalEnergy: number;
  /** Number of basis functions. */
  readonly nBasis: number;
  /** Number of MOs. */
  readonly nMO: number;
  /** True for converged SCF + post-HF. */
  readonly success: boolean;
  /** Optional extra fields (any JSON). */
  readonly extras?: Record<string, unknown>;
}

export interface QCSchemaClosedShell extends QCSchemaCommon {
  readonly nElectrons: number;
  readonly nOccupied: number;
  /** SCF energy (E_HF or E_RKS). Distinct from totalEnergy when
   *  a post-HF correlation is added on top. */
  readonly scfEnergy: number;
  /** Optional post-HF correlation energy (MP2, CCSD, CCSD(T)). */
  readonly correlationEnergy?: number;
}

export interface QCSchemaOpenShell extends QCSchemaCommon {
  readonly nAlpha: number;
  readonly nBeta: number;
  readonly scfEnergy: number;
  readonly correlationEnergy?: number;
  /** Optional ⟨S²⟩ expectation. */
  readonly s2?: number;
}

/**
 * Render a closed-shell calculation as a QCSchema AtomicResult JSON string.
 */
export function toQCSchemaClosedShell(input: QCSchemaClosedShell): string {
  const obj = {
    schema_name: "qcschema_output",
    schema_version: 1,
    molecule: buildMolecule(input.atoms),
    driver: "energy",
    model: { method: input.method, basis: input.basis },
    return_result: input.totalEnergy,
    properties: {
      return_energy: input.totalEnergy,
      scf_total_energy: input.scfEnergy,
      ...(input.oneElectron !== undefined && { scf_one_electron_energy: input.oneElectron }),
      ...(input.twoElectron !== undefined && { scf_two_electron_energy: input.twoElectron }),
      nuclear_repulsion_energy: input.nuclearRepulsion,
      ...(input.correlationEnergy !== undefined && {
        scf_correlation_energy: input.correlationEnergy,
        mp2_correlation_energy: input.method.startsWith("mp2") ? input.correlationEnergy : undefined,
        ccsd_correlation_energy: input.method.startsWith("ccsd") ? input.correlationEnergy : undefined,
      }),
      calcinfo_natom: input.atoms.length,
      calcinfo_nbasis: input.nBasis,
      calcinfo_nmo: input.nMO,
      calcinfo_nalpha: input.nOccupied,
      calcinfo_nbeta: input.nOccupied,
    },
    success: input.success,
    extras: {
      webgpu_q_version: "0.3.0",
      ...(input.extras ?? {}),
    },
  };
  return JSON.stringify(obj, null, 2);
}

/**
 * Render an open-shell (UHF / UKS / UCCSD / etc.) calculation as a
 * QCSchema AtomicResult JSON string.
 */
export function toQCSchemaOpenShell(input: QCSchemaOpenShell): string {
  const obj = {
    schema_name: "qcschema_output",
    schema_version: 1,
    molecule: {
      ...buildMolecule(input.atoms),
      molecular_multiplicity: input.nAlpha - input.nBeta + 1,
    },
    driver: "energy",
    model: { method: input.method, basis: input.basis },
    return_result: input.totalEnergy,
    properties: {
      return_energy: input.totalEnergy,
      scf_total_energy: input.scfEnergy,
      ...(input.oneElectron !== undefined && { scf_one_electron_energy: input.oneElectron }),
      ...(input.twoElectron !== undefined && { scf_two_electron_energy: input.twoElectron }),
      nuclear_repulsion_energy: input.nuclearRepulsion,
      ...(input.correlationEnergy !== undefined && { scf_correlation_energy: input.correlationEnergy }),
      calcinfo_natom: input.atoms.length,
      calcinfo_nbasis: input.nBasis,
      calcinfo_nmo: input.nMO,
      calcinfo_nalpha: input.nAlpha,
      calcinfo_nbeta: input.nBeta,
      ...(input.s2 !== undefined && { scf_s_squared: input.s2 }),
    },
    success: input.success,
    extras: {
      webgpu_q_version: "0.3.0",
      ...(input.extras ?? {}),
    },
  };
  return JSON.stringify(obj, null, 2);
}

// ─────────────────────────────────────────────────────────────

function buildMolecule(atoms: readonly Atom[]): {
  schema_name: string;
  schema_version: number;
  symbols: string[];
  geometry: number[];
  atomic_numbers: number[];
} {
  const symbols: string[] = [];
  const geometry: number[] = [];
  const atomic_numbers: number[] = [];
  for (const at of atoms) {
    symbols.push(at.symbol);
    geometry.push(at.pos[0] * ANGSTROM_PER_BOHR);
    geometry.push(at.pos[1] * ANGSTROM_PER_BOHR);
    geometry.push(at.pos[2] * ANGSTROM_PER_BOHR);
    atomic_numbers.push(at.ghost ? 0 : Z_FOR[at.symbol]);
  }
  return {
    schema_name: "qcschema_molecule",
    schema_version: 2,
    symbols,
    geometry,
    atomic_numbers,
  };
}
