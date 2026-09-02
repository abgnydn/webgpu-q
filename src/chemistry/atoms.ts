// ─────────────────────────────────────────────────────────────
// atoms.ts — atom registry. Maps a chemical symbol to (Z, full
// STO-3G shell list at a given center). Keeps the per-molecule
// builder code thin: every molecule is just a list of (symbol,
// position) tuples that this module turns into the CG shells +
// nuclei the chemistry pipeline expects.
// ─────────────────────────────────────────────────────────────

import {
  STO3G_H_1S, STO3G_HE_1S,
  STO3G_LI_1S, STO3G_LI_2S,
  STO3G_BE_1S, STO3G_BE_2S, STO3G_BE_2P,
  STO3G_B_1S, STO3G_B_2S, STO3G_B_2P,
  STO3G_NE_1S, STO3G_NE_2S, STO3G_NE_2P,
  CCPVDZ_B_1S, CCPVDZ_B_2S, CCPVDZ_B_2S_P,
  CCPVDZ_B_2P, CCPVDZ_B_2P_P, CCPVDZ_B_3D,
  CCPVDZ_NE_1S, CCPVDZ_NE_2S, CCPVDZ_NE_2S_P,
  CCPVDZ_NE_2P, CCPVDZ_NE_2P_P, CCPVDZ_NE_3D,
  AUG_CCPVDZ_B_DIFFUSE_S, AUG_CCPVDZ_B_DIFFUSE_P, AUG_CCPVDZ_B_DIFFUSE_D,
  AUG_CCPVDZ_NE_DIFFUSE_S, AUG_CCPVDZ_NE_DIFFUSE_P, AUG_CCPVDZ_NE_DIFFUSE_D,
  STO3G_C_1S, STO3G_C_2S, STO3G_C_2P,
  STO3G_N_1S, STO3G_N_2S, STO3G_N_2P,
  STO3G_O_1S, STO3G_O_2S, STO3G_O_2P,
  STO3G_F_1S, STO3G_F_2S, STO3G_F_2P,
  CCPVDZ_H_1S, CCPVDZ_H_2S, CCPVDZ_H_2P,
  CCPVDZ_HE_1S, CCPVDZ_HE_2S, CCPVDZ_HE_2P,
  CCPVDZ_LI_1S, CCPVDZ_LI_2S, CCPVDZ_LI_2S_P,
  CCPVDZ_LI_2P, CCPVDZ_LI_2P_P, CCPVDZ_LI_3D,
  CCPVDZ_BE_1S, CCPVDZ_BE_2S, CCPVDZ_BE_2S_P,
  CCPVDZ_BE_2P, CCPVDZ_BE_2P_P, CCPVDZ_BE_3D,
  CCPVDZ_C_1S, CCPVDZ_C_2S, CCPVDZ_C_2S_P,
  CCPVDZ_C_2P, CCPVDZ_C_2P_P, CCPVDZ_C_3D,
  CCPVDZ_N_1S, CCPVDZ_N_2S, CCPVDZ_N_2S_P,
  CCPVDZ_N_2P, CCPVDZ_N_2P_P, CCPVDZ_N_3D,
  CCPVDZ_O_1S, CCPVDZ_O_2S, CCPVDZ_O_2S_P,
  CCPVDZ_O_2P, CCPVDZ_O_2P_P, CCPVDZ_O_3D,
  CCPVDZ_F_1S, CCPVDZ_F_2S, CCPVDZ_F_2S_P,
  CCPVDZ_F_2P, CCPVDZ_F_2P_P, CCPVDZ_F_3D,
  AUG_CCPVDZ_H_DIFFUSE_S, AUG_CCPVDZ_H_DIFFUSE_P,
  AUG_CCPVDZ_HE_DIFFUSE_S, AUG_CCPVDZ_HE_DIFFUSE_P,
  AUG_CCPVDZ_LI_DIFFUSE_S, AUG_CCPVDZ_LI_DIFFUSE_P, AUG_CCPVDZ_LI_DIFFUSE_D,
  AUG_CCPVDZ_BE_DIFFUSE_S, AUG_CCPVDZ_BE_DIFFUSE_P, AUG_CCPVDZ_BE_DIFFUSE_D,
  AUG_CCPVDZ_C_DIFFUSE_S,  AUG_CCPVDZ_C_DIFFUSE_P,  AUG_CCPVDZ_C_DIFFUSE_D,
  AUG_CCPVDZ_N_DIFFUSE_S,  AUG_CCPVDZ_N_DIFFUSE_P,  AUG_CCPVDZ_N_DIFFUSE_D,
  AUG_CCPVDZ_O_DIFFUSE_S, AUG_CCPVDZ_O_DIFFUSE_P, AUG_CCPVDZ_O_DIFFUSE_D,
  AUG_CCPVDZ_F_DIFFUSE_S,  AUG_CCPVDZ_F_DIFFUSE_P,  AUG_CCPVDZ_F_DIFFUSE_D,
  STO3G_NA_1S, STO3G_NA_2S, STO3G_NA_3S, STO3G_NA_2P, STO3G_NA_3P,
  STO3G_MG_1S, STO3G_MG_2S, STO3G_MG_3S, STO3G_MG_2P, STO3G_MG_3P,
  STO3G_AL_1S, STO3G_AL_2S, STO3G_AL_3S, STO3G_AL_2P, STO3G_AL_3P,
  STO3G_SI_1S, STO3G_SI_2S, STO3G_SI_3S, STO3G_SI_2P, STO3G_SI_3P,
  STO3G_P_1S, STO3G_P_2S, STO3G_P_3S, STO3G_P_2P, STO3G_P_3P,
  STO3G_S_1S, STO3G_S_2S, STO3G_S_3S, STO3G_S_2P, STO3G_S_3P,
  STO3G_CL_1S, STO3G_CL_2S, STO3G_CL_3S, STO3G_CL_2P, STO3G_CL_3P,
  STO3G_AR_1S, STO3G_AR_2S, STO3G_AR_3S, STO3G_AR_2P, STO3G_AR_3P,
  CCPVDZ_NA_1S, CCPVDZ_NA_2S, CCPVDZ_NA_3S, CCPVDZ_NA_3S_P,
  CCPVDZ_NA_2P, CCPVDZ_NA_3P, CCPVDZ_NA_3P_P, CCPVDZ_NA_3D,
  CCPVDZ_MG_1S, CCPVDZ_MG_2S, CCPVDZ_MG_3S, CCPVDZ_MG_3S_P,
  CCPVDZ_MG_2P, CCPVDZ_MG_3P, CCPVDZ_MG_3P_P, CCPVDZ_MG_3D,
  CCPVDZ_AL_1S, CCPVDZ_AL_2S, CCPVDZ_AL_3S, CCPVDZ_AL_3S_P,
  CCPVDZ_AL_2P, CCPVDZ_AL_3P, CCPVDZ_AL_3P_P, CCPVDZ_AL_3D,
  CCPVDZ_SI_1S, CCPVDZ_SI_2S, CCPVDZ_SI_3S, CCPVDZ_SI_3S_P,
  CCPVDZ_SI_2P, CCPVDZ_SI_3P, CCPVDZ_SI_3P_P, CCPVDZ_SI_3D,
  CCPVDZ_P_1S, CCPVDZ_P_2S, CCPVDZ_P_3S, CCPVDZ_P_3S_P,
  CCPVDZ_P_2P, CCPVDZ_P_3P, CCPVDZ_P_3P_P, CCPVDZ_P_3D,
  CCPVDZ_S_1S, CCPVDZ_S_2S, CCPVDZ_S_3S, CCPVDZ_S_3S_P,
  CCPVDZ_S_2P, CCPVDZ_S_3P, CCPVDZ_S_3P_P, CCPVDZ_S_3D,
  CCPVDZ_CL_1S, CCPVDZ_CL_2S, CCPVDZ_CL_3S, CCPVDZ_CL_3S_P,
  CCPVDZ_CL_2P, CCPVDZ_CL_3P, CCPVDZ_CL_3P_P, CCPVDZ_CL_3D,
  CCPVDZ_AR_1S, CCPVDZ_AR_2S, CCPVDZ_AR_3S, CCPVDZ_AR_3S_P,
  CCPVDZ_AR_2P, CCPVDZ_AR_3P, CCPVDZ_AR_3P_P, CCPVDZ_AR_3D,
  AUG_CCPVDZ_NA_DIFFUSE_S, AUG_CCPVDZ_NA_DIFFUSE_P, AUG_CCPVDZ_NA_DIFFUSE_D,
  AUG_CCPVDZ_MG_DIFFUSE_S, AUG_CCPVDZ_MG_DIFFUSE_P, AUG_CCPVDZ_MG_DIFFUSE_D,
  AUG_CCPVDZ_AL_DIFFUSE_S, AUG_CCPVDZ_AL_DIFFUSE_P, AUG_CCPVDZ_AL_DIFFUSE_D,
  AUG_CCPVDZ_SI_DIFFUSE_S, AUG_CCPVDZ_SI_DIFFUSE_P, AUG_CCPVDZ_SI_DIFFUSE_D,
  AUG_CCPVDZ_P_DIFFUSE_S, AUG_CCPVDZ_P_DIFFUSE_P, AUG_CCPVDZ_P_DIFFUSE_D,
  AUG_CCPVDZ_S_DIFFUSE_S, AUG_CCPVDZ_S_DIFFUSE_P, AUG_CCPVDZ_S_DIFFUSE_D,
  AUG_CCPVDZ_CL_DIFFUSE_S, AUG_CCPVDZ_CL_DIFFUSE_P, AUG_CCPVDZ_CL_DIFFUSE_D,
  AUG_CCPVDZ_AR_DIFFUSE_S, AUG_CCPVDZ_AR_DIFFUSE_P, AUG_CCPVDZ_AR_DIFFUSE_D,
} from "./integrals.js";
import { type CGShell, makeCGShell } from "./integrals-cg.js";
import { type Nucleus } from "./cg-molecular.js";

export type AtomSymbol =
  | "H" | "He" | "Li" | "Be" | "B" | "C" | "N" | "O" | "F" | "Ne"
  | "Na" | "Mg" | "Al" | "Si" | "P" | "S" | "Cl" | "Ar";
export type BasisName = "sto-3g" | "cc-pvdz" | "aug-cc-pvdz";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

export interface Atom {
  readonly symbol: AtomSymbol;
  /** Position in Ångströms (will be converted to Bohr). */
  readonly pos: readonly [number, number, number];
  /** If true, this atom contributes its basis functions to the calculation
   *  but **no nuclear charge** (Z=0) and **no electrons**. Used by
   *  Boys-Bernardi counterpoise correction to compute a fragment's energy
   *  in the full dimer basis — the "ghost" augments the AO space without
   *  changing the molecular Hamiltonian's nuclear / electron count.
   *  Default false. */
  readonly ghost?: boolean;
}

/** Atomic number for each supported atom. */
export const Z_FOR: Readonly<Record<AtomSymbol, number>> = {
  H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10,
  Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18,
};

/** Number of electrons in the neutral atom. */
export const N_ELECTRONS_FOR: Readonly<Record<AtomSymbol, number>> = {
  H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10,
  Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18,
};

/**
 * Canonical frozen-core count for each atom.
 *
 * For first-row chemistry the convention is to freeze the 1s core
 * orbital of every heavy atom (Li → Ne). Hydrogen has nothing to
 * freeze. Freezing core orbitals from the correlation treatment
 * gives 2-3× speedup on CCSD/CCSD(T) with sub-µHa impact on
 * relative energies (the same chemistry observable that any
 * comparative calculation cares about).
 */
export const FROZEN_CORE_FOR: Readonly<Record<AtomSymbol, number>> = {
  H: 0, He: 0, Li: 1, Be: 1, B: 1, C: 1, N: 1, O: 1, F: 1, Ne: 1,
  // Third row freezes the full neon core (1s + 2s + 2p = 5 orbitals),
  // not just 1s — the standard convention for Na-Ar.
  Na: 5, Mg: 5, Al: 5, Si: 5, P: 5, S: 5, Cl: 5, Ar: 5,
};

/** Default frozen-core count for a molecule (sum of per-atom 1s cores). */
export function defaultFrozenCore(atoms: readonly Atom[]): number {
  let n = 0;
  for (const a of atoms) n += FROZEN_CORE_FOR[a.symbol];
  return n;
}

/**
 * Return the CG-shell list for a given atom + basis set. Default
 * basis is STO-3G for backward compatibility with v3-v5 callers.
 *
 * STO-3G:
 *   H     → [1s]
 *   Li    → [1s, 2s]                          (no 2p in this codebase)
 *   Be    → [1s, 2s, 2p_x, 2p_y, 2p_z]
 *   C/N/O → [1s, 2s, 2p_x, 2p_y, 2p_z]
 *
 * cc-pVDZ (Phase E stage 2):
 *   H → [1s, 2s, 2p_x, 2p_y, 2p_z]            (5 funcs)
 *   O → [1s, 2s, 2s', 2p, 2p', 3d_xx, 3d_yy,
 *        3d_zz, 3d_xy, 3d_xz, 3d_yz]          (14 funcs)
 *   (Other atoms not yet covered for cc-pVDZ.)
 */
export function atomShells(
  symbol: AtomSymbol,
  pos_bohr: readonly [number, number, number],
  basis: BasisName = "sto-3g",
): CGShell[] {
  if (basis === "cc-pvdz") return atomShellsCcPvdz(symbol, pos_bohr);
  if (basis === "aug-cc-pvdz") {
    return [...atomShellsCcPvdz(symbol, pos_bohr), ...atomShellsAugDiffuse(symbol, pos_bohr)];
  }
  switch (symbol) {
    case "H":
      return [makeCGShell(STO3G_H_1S, pos_bohr, [0, 0, 0], "H:1s")];
    case "He":
      return [makeCGShell(STO3G_HE_1S, pos_bohr, [0, 0, 0], "He:1s")];
    case "Li":
      return [
        makeCGShell(STO3G_LI_1S, pos_bohr, [0, 0, 0], "Li:1s"),
        makeCGShell(STO3G_LI_2S, pos_bohr, [0, 0, 0], "Li:2s"),
      ];
    case "Be":
      return [
        makeCGShell(STO3G_BE_1S, pos_bohr, [0, 0, 0], "Be:1s"),
        makeCGShell(STO3G_BE_2S, pos_bohr, [0, 0, 0], "Be:2s"),
        makeCGShell(STO3G_BE_2P, pos_bohr, [1, 0, 0], "Be:2p_x"),
        makeCGShell(STO3G_BE_2P, pos_bohr, [0, 1, 0], "Be:2p_y"),
        makeCGShell(STO3G_BE_2P, pos_bohr, [0, 0, 1], "Be:2p_z"),
      ];
    case "B":
      return [
        makeCGShell(STO3G_B_1S, pos_bohr, [0, 0, 0], "B:1s"),
        makeCGShell(STO3G_B_2S, pos_bohr, [0, 0, 0], "B:2s"),
        makeCGShell(STO3G_B_2P, pos_bohr, [1, 0, 0], "B:2p_x"),
        makeCGShell(STO3G_B_2P, pos_bohr, [0, 1, 0], "B:2p_y"),
        makeCGShell(STO3G_B_2P, pos_bohr, [0, 0, 1], "B:2p_z"),
      ];
    case "C":
      return [
        makeCGShell(STO3G_C_1S, pos_bohr, [0, 0, 0], "C:1s"),
        makeCGShell(STO3G_C_2S, pos_bohr, [0, 0, 0], "C:2s"),
        makeCGShell(STO3G_C_2P, pos_bohr, [1, 0, 0], "C:2p_x"),
        makeCGShell(STO3G_C_2P, pos_bohr, [0, 1, 0], "C:2p_y"),
        makeCGShell(STO3G_C_2P, pos_bohr, [0, 0, 1], "C:2p_z"),
      ];
    case "N":
      return [
        makeCGShell(STO3G_N_1S, pos_bohr, [0, 0, 0], "N:1s"),
        makeCGShell(STO3G_N_2S, pos_bohr, [0, 0, 0], "N:2s"),
        makeCGShell(STO3G_N_2P, pos_bohr, [1, 0, 0], "N:2p_x"),
        makeCGShell(STO3G_N_2P, pos_bohr, [0, 1, 0], "N:2p_y"),
        makeCGShell(STO3G_N_2P, pos_bohr, [0, 0, 1], "N:2p_z"),
      ];
    case "O":
      return [
        makeCGShell(STO3G_O_1S, pos_bohr, [0, 0, 0], "O:1s"),
        makeCGShell(STO3G_O_2S, pos_bohr, [0, 0, 0], "O:2s"),
        makeCGShell(STO3G_O_2P, pos_bohr, [1, 0, 0], "O:2p_x"),
        makeCGShell(STO3G_O_2P, pos_bohr, [0, 1, 0], "O:2p_y"),
        makeCGShell(STO3G_O_2P, pos_bohr, [0, 0, 1], "O:2p_z"),
      ];
    case "F":
      return [
        makeCGShell(STO3G_F_1S, pos_bohr, [0, 0, 0], "F:1s"),
        makeCGShell(STO3G_F_2S, pos_bohr, [0, 0, 0], "F:2s"),
        makeCGShell(STO3G_F_2P, pos_bohr, [1, 0, 0], "F:2p_x"),
        makeCGShell(STO3G_F_2P, pos_bohr, [0, 1, 0], "F:2p_y"),
        makeCGShell(STO3G_F_2P, pos_bohr, [0, 0, 1], "F:2p_z"),
      ];
    case "Ne":
      return [
        makeCGShell(STO3G_NE_1S, pos_bohr, [0, 0, 0], "Ne:1s"),
        makeCGShell(STO3G_NE_2S, pos_bohr, [0, 0, 0], "Ne:2s"),
        makeCGShell(STO3G_NE_2P, pos_bohr, [1, 0, 0], "Ne:2p_x"),
        makeCGShell(STO3G_NE_2P, pos_bohr, [0, 1, 0], "Ne:2p_y"),
        makeCGShell(STO3G_NE_2P, pos_bohr, [0, 0, 1], "Ne:2p_z"),
      ];
    case "Na":
      return [
        makeCGShell(STO3G_NA_1S, pos_bohr, [0, 0, 0], "Na:1s"),
        makeCGShell(STO3G_NA_2S, pos_bohr, [0, 0, 0], "Na:2s"),
        makeCGShell(STO3G_NA_3S, pos_bohr, [0, 0, 0], "Na:3s"),
        makeCGShell(STO3G_NA_2P, pos_bohr, [1, 0, 0], "Na:2p_x"),
        makeCGShell(STO3G_NA_2P, pos_bohr, [0, 1, 0], "Na:2p_y"),
        makeCGShell(STO3G_NA_2P, pos_bohr, [0, 0, 1], "Na:2p_z"),
        makeCGShell(STO3G_NA_3P, pos_bohr, [1, 0, 0], "Na:3p_x"),
        makeCGShell(STO3G_NA_3P, pos_bohr, [0, 1, 0], "Na:3p_y"),
        makeCGShell(STO3G_NA_3P, pos_bohr, [0, 0, 1], "Na:3p_z"),
      ];
    case "Mg":
      return [
        makeCGShell(STO3G_MG_1S, pos_bohr, [0, 0, 0], "Mg:1s"),
        makeCGShell(STO3G_MG_2S, pos_bohr, [0, 0, 0], "Mg:2s"),
        makeCGShell(STO3G_MG_3S, pos_bohr, [0, 0, 0], "Mg:3s"),
        makeCGShell(STO3G_MG_2P, pos_bohr, [1, 0, 0], "Mg:2p_x"),
        makeCGShell(STO3G_MG_2P, pos_bohr, [0, 1, 0], "Mg:2p_y"),
        makeCGShell(STO3G_MG_2P, pos_bohr, [0, 0, 1], "Mg:2p_z"),
        makeCGShell(STO3G_MG_3P, pos_bohr, [1, 0, 0], "Mg:3p_x"),
        makeCGShell(STO3G_MG_3P, pos_bohr, [0, 1, 0], "Mg:3p_y"),
        makeCGShell(STO3G_MG_3P, pos_bohr, [0, 0, 1], "Mg:3p_z"),
      ];
    case "Al":
      return [
        makeCGShell(STO3G_AL_1S, pos_bohr, [0, 0, 0], "Al:1s"),
        makeCGShell(STO3G_AL_2S, pos_bohr, [0, 0, 0], "Al:2s"),
        makeCGShell(STO3G_AL_3S, pos_bohr, [0, 0, 0], "Al:3s"),
        makeCGShell(STO3G_AL_2P, pos_bohr, [1, 0, 0], "Al:2p_x"),
        makeCGShell(STO3G_AL_2P, pos_bohr, [0, 1, 0], "Al:2p_y"),
        makeCGShell(STO3G_AL_2P, pos_bohr, [0, 0, 1], "Al:2p_z"),
        makeCGShell(STO3G_AL_3P, pos_bohr, [1, 0, 0], "Al:3p_x"),
        makeCGShell(STO3G_AL_3P, pos_bohr, [0, 1, 0], "Al:3p_y"),
        makeCGShell(STO3G_AL_3P, pos_bohr, [0, 0, 1], "Al:3p_z"),
      ];
    case "Si":
      return [
        makeCGShell(STO3G_SI_1S, pos_bohr, [0, 0, 0], "Si:1s"),
        makeCGShell(STO3G_SI_2S, pos_bohr, [0, 0, 0], "Si:2s"),
        makeCGShell(STO3G_SI_3S, pos_bohr, [0, 0, 0], "Si:3s"),
        makeCGShell(STO3G_SI_2P, pos_bohr, [1, 0, 0], "Si:2p_x"),
        makeCGShell(STO3G_SI_2P, pos_bohr, [0, 1, 0], "Si:2p_y"),
        makeCGShell(STO3G_SI_2P, pos_bohr, [0, 0, 1], "Si:2p_z"),
        makeCGShell(STO3G_SI_3P, pos_bohr, [1, 0, 0], "Si:3p_x"),
        makeCGShell(STO3G_SI_3P, pos_bohr, [0, 1, 0], "Si:3p_y"),
        makeCGShell(STO3G_SI_3P, pos_bohr, [0, 0, 1], "Si:3p_z"),
      ];
    case "P":
      return [
        makeCGShell(STO3G_P_1S, pos_bohr, [0, 0, 0], "P:1s"),
        makeCGShell(STO3G_P_2S, pos_bohr, [0, 0, 0], "P:2s"),
        makeCGShell(STO3G_P_3S, pos_bohr, [0, 0, 0], "P:3s"),
        makeCGShell(STO3G_P_2P, pos_bohr, [1, 0, 0], "P:2p_x"),
        makeCGShell(STO3G_P_2P, pos_bohr, [0, 1, 0], "P:2p_y"),
        makeCGShell(STO3G_P_2P, pos_bohr, [0, 0, 1], "P:2p_z"),
        makeCGShell(STO3G_P_3P, pos_bohr, [1, 0, 0], "P:3p_x"),
        makeCGShell(STO3G_P_3P, pos_bohr, [0, 1, 0], "P:3p_y"),
        makeCGShell(STO3G_P_3P, pos_bohr, [0, 0, 1], "P:3p_z"),
      ];
    case "S":
      return [
        makeCGShell(STO3G_S_1S, pos_bohr, [0, 0, 0], "S:1s"),
        makeCGShell(STO3G_S_2S, pos_bohr, [0, 0, 0], "S:2s"),
        makeCGShell(STO3G_S_3S, pos_bohr, [0, 0, 0], "S:3s"),
        makeCGShell(STO3G_S_2P, pos_bohr, [1, 0, 0], "S:2p_x"),
        makeCGShell(STO3G_S_2P, pos_bohr, [0, 1, 0], "S:2p_y"),
        makeCGShell(STO3G_S_2P, pos_bohr, [0, 0, 1], "S:2p_z"),
        makeCGShell(STO3G_S_3P, pos_bohr, [1, 0, 0], "S:3p_x"),
        makeCGShell(STO3G_S_3P, pos_bohr, [0, 1, 0], "S:3p_y"),
        makeCGShell(STO3G_S_3P, pos_bohr, [0, 0, 1], "S:3p_z"),
      ];
    case "Cl":
      return [
        makeCGShell(STO3G_CL_1S, pos_bohr, [0, 0, 0], "Cl:1s"),
        makeCGShell(STO3G_CL_2S, pos_bohr, [0, 0, 0], "Cl:2s"),
        makeCGShell(STO3G_CL_3S, pos_bohr, [0, 0, 0], "Cl:3s"),
        makeCGShell(STO3G_CL_2P, pos_bohr, [1, 0, 0], "Cl:2p_x"),
        makeCGShell(STO3G_CL_2P, pos_bohr, [0, 1, 0], "Cl:2p_y"),
        makeCGShell(STO3G_CL_2P, pos_bohr, [0, 0, 1], "Cl:2p_z"),
        makeCGShell(STO3G_CL_3P, pos_bohr, [1, 0, 0], "Cl:3p_x"),
        makeCGShell(STO3G_CL_3P, pos_bohr, [0, 1, 0], "Cl:3p_y"),
        makeCGShell(STO3G_CL_3P, pos_bohr, [0, 0, 1], "Cl:3p_z"),
      ];
    case "Ar":
      return [
        makeCGShell(STO3G_AR_1S, pos_bohr, [0, 0, 0], "Ar:1s"),
        makeCGShell(STO3G_AR_2S, pos_bohr, [0, 0, 0], "Ar:2s"),
        makeCGShell(STO3G_AR_3S, pos_bohr, [0, 0, 0], "Ar:3s"),
        makeCGShell(STO3G_AR_2P, pos_bohr, [1, 0, 0], "Ar:2p_x"),
        makeCGShell(STO3G_AR_2P, pos_bohr, [0, 1, 0], "Ar:2p_y"),
        makeCGShell(STO3G_AR_2P, pos_bohr, [0, 0, 1], "Ar:2p_z"),
        makeCGShell(STO3G_AR_3P, pos_bohr, [1, 0, 0], "Ar:3p_x"),
        makeCGShell(STO3G_AR_3P, pos_bohr, [0, 1, 0], "Ar:3p_y"),
        makeCGShell(STO3G_AR_3P, pos_bohr, [0, 0, 1], "Ar:3p_z"),
      ];
  }
}

/** cc-pVDZ shells. H–Ar, all 18 (row 3 via `heavyShellsRow3`). */
function atomShellsCcPvdz(symbol: AtomSymbol, pos: readonly [number, number, number]): CGShell[] {
  /** Build the 14-function cc-pVDZ heavy-atom shell stack
   *  (3s + 2p + 1d → 1+1+1+3+3+6 = 15 Cartesian d functions actually).
   *  Same layout that's been validated against PySCF on H₂O. */
  type ShellData = { readonly alpha: readonly number[]; readonly c: readonly number[] };
  function heavyShells(
    sym: string,
    s1: ShellData, s2: ShellData, s2p: ShellData,
    p1: ShellData, p2: ShellData, d1: ShellData,
  ): CGShell[] {
    return [
      makeCGShell(s1,  pos, [0, 0, 0], `${sym}:1s`),
      makeCGShell(s2,  pos, [0, 0, 0], `${sym}:2s`),
      makeCGShell(s2p, pos, [0, 0, 0], `${sym}:2s'`),
      makeCGShell(p1,  pos, [1, 0, 0], `${sym}:2p_x`),
      makeCGShell(p1,  pos, [0, 1, 0], `${sym}:2p_y`),
      makeCGShell(p1,  pos, [0, 0, 1], `${sym}:2p_z`),
      makeCGShell(p2,  pos, [1, 0, 0], `${sym}:2p'_x`),
      makeCGShell(p2,  pos, [0, 1, 0], `${sym}:2p'_y`),
      makeCGShell(p2,  pos, [0, 0, 1], `${sym}:2p'_z`),
      // 6 Cartesian d functions: xx, yy, zz, xy, xz, yz.
      makeCGShell(d1, pos, [2, 0, 0], `${sym}:3d_xx`),
      makeCGShell(d1, pos, [0, 2, 0], `${sym}:3d_yy`),
      makeCGShell(d1, pos, [0, 0, 2], `${sym}:3d_zz`),
      makeCGShell(d1, pos, [1, 1, 0], `${sym}:3d_xy`),
      makeCGShell(d1, pos, [1, 0, 1], `${sym}:3d_xz`),
      makeCGShell(d1, pos, [0, 1, 1], `${sym}:3d_yz`),
    ];
  }
  /** Third-row cc-pVDZ stack: 4s + 3p + 1d -> 4 + 9 + 6 = 19 Cartesian
   *  functions (18 spherical). Kept separate from `heavyShells` rather
   *  than generalising it, so the row-2 path emits byte-identical shells
   *  and no pinned first-row number can move. The d-set must stay a
   *  contiguous canonical 6-tuple — `buildSphericalDTransform` in
   *  cg-molecular.ts throws otherwise. */
  function heavyShellsRow3(
    sym: string,
    s1: ShellData, s2: ShellData, s3: ShellData, s3p: ShellData,
    p1: ShellData, p2: ShellData, p3p: ShellData, d1: ShellData,
  ): CGShell[] {
    return [
      makeCGShell(s1,  pos, [0, 0, 0], `${sym}:1s`),
      makeCGShell(s2,  pos, [0, 0, 0], `${sym}:2s`),
      makeCGShell(s3,  pos, [0, 0, 0], `${sym}:3s`),
      makeCGShell(s3p, pos, [0, 0, 0], `${sym}:3s'`),
      makeCGShell(p1,  pos, [1, 0, 0], `${sym}:2p_x`),
      makeCGShell(p1,  pos, [0, 1, 0], `${sym}:2p_y`),
      makeCGShell(p1,  pos, [0, 0, 1], `${sym}:2p_z`),
      makeCGShell(p2,  pos, [1, 0, 0], `${sym}:3p_x`),
      makeCGShell(p2,  pos, [0, 1, 0], `${sym}:3p_y`),
      makeCGShell(p2,  pos, [0, 0, 1], `${sym}:3p_z`),
      makeCGShell(p3p, pos, [1, 0, 0], `${sym}:3p'_x`),
      makeCGShell(p3p, pos, [0, 1, 0], `${sym}:3p'_y`),
      makeCGShell(p3p, pos, [0, 0, 1], `${sym}:3p'_z`),
      makeCGShell(d1, pos, [2, 0, 0], `${sym}:3d_xx`),
      makeCGShell(d1, pos, [0, 2, 0], `${sym}:3d_yy`),
      makeCGShell(d1, pos, [0, 0, 2], `${sym}:3d_zz`),
      makeCGShell(d1, pos, [1, 1, 0], `${sym}:3d_xy`),
      makeCGShell(d1, pos, [1, 0, 1], `${sym}:3d_xz`),
      makeCGShell(d1, pos, [0, 1, 1], `${sym}:3d_yz`),
    ];
  }
  switch (symbol) {
    case "H":
      return [
        makeCGShell(CCPVDZ_H_1S, pos, [0, 0, 0], "H:1s"),
        makeCGShell(CCPVDZ_H_2S, pos, [0, 0, 0], "H:2s"),
        makeCGShell(CCPVDZ_H_2P, pos, [1, 0, 0], "H:2p_x"),
        makeCGShell(CCPVDZ_H_2P, pos, [0, 1, 0], "H:2p_y"),
        makeCGShell(CCPVDZ_H_2P, pos, [0, 0, 1], "H:2p_z"),
      ];
    case "Li": return heavyShells("Li", CCPVDZ_LI_1S, CCPVDZ_LI_2S, CCPVDZ_LI_2S_P, CCPVDZ_LI_2P, CCPVDZ_LI_2P_P, CCPVDZ_LI_3D);
    case "Be": return heavyShells("Be", CCPVDZ_BE_1S, CCPVDZ_BE_2S, CCPVDZ_BE_2S_P, CCPVDZ_BE_2P, CCPVDZ_BE_2P_P, CCPVDZ_BE_3D);
    case "B":  return heavyShells("B",  CCPVDZ_B_1S,  CCPVDZ_B_2S,  CCPVDZ_B_2S_P,  CCPVDZ_B_2P,  CCPVDZ_B_2P_P,  CCPVDZ_B_3D);
    case "C":  return heavyShells("C",  CCPVDZ_C_1S,  CCPVDZ_C_2S,  CCPVDZ_C_2S_P,  CCPVDZ_C_2P,  CCPVDZ_C_2P_P,  CCPVDZ_C_3D);
    case "N":  return heavyShells("N",  CCPVDZ_N_1S,  CCPVDZ_N_2S,  CCPVDZ_N_2S_P,  CCPVDZ_N_2P,  CCPVDZ_N_2P_P,  CCPVDZ_N_3D);
    case "O":  return heavyShells("O",  CCPVDZ_O_1S,  CCPVDZ_O_2S,  CCPVDZ_O_2S_P,  CCPVDZ_O_2P,  CCPVDZ_O_2P_P,  CCPVDZ_O_3D);
    case "F":  return heavyShells("F",  CCPVDZ_F_1S,  CCPVDZ_F_2S,  CCPVDZ_F_2S_P,  CCPVDZ_F_2P,  CCPVDZ_F_2P_P,  CCPVDZ_F_3D);
    case "Ne": return heavyShells("Ne", CCPVDZ_NE_1S, CCPVDZ_NE_2S, CCPVDZ_NE_2S_P, CCPVDZ_NE_2P, CCPVDZ_NE_2P_P, CCPVDZ_NE_3D);
    case "Na": return heavyShellsRow3("Na", CCPVDZ_NA_1S, CCPVDZ_NA_2S, CCPVDZ_NA_3S, CCPVDZ_NA_3S_P, CCPVDZ_NA_2P, CCPVDZ_NA_3P, CCPVDZ_NA_3P_P, CCPVDZ_NA_3D);
    case "Mg": return heavyShellsRow3("Mg", CCPVDZ_MG_1S, CCPVDZ_MG_2S, CCPVDZ_MG_3S, CCPVDZ_MG_3S_P, CCPVDZ_MG_2P, CCPVDZ_MG_3P, CCPVDZ_MG_3P_P, CCPVDZ_MG_3D);
    case "Al": return heavyShellsRow3("Al", CCPVDZ_AL_1S, CCPVDZ_AL_2S, CCPVDZ_AL_3S, CCPVDZ_AL_3S_P, CCPVDZ_AL_2P, CCPVDZ_AL_3P, CCPVDZ_AL_3P_P, CCPVDZ_AL_3D);
    case "Si": return heavyShellsRow3("Si", CCPVDZ_SI_1S, CCPVDZ_SI_2S, CCPVDZ_SI_3S, CCPVDZ_SI_3S_P, CCPVDZ_SI_2P, CCPVDZ_SI_3P, CCPVDZ_SI_3P_P, CCPVDZ_SI_3D);
    case "P": return heavyShellsRow3("P", CCPVDZ_P_1S, CCPVDZ_P_2S, CCPVDZ_P_3S, CCPVDZ_P_3S_P, CCPVDZ_P_2P, CCPVDZ_P_3P, CCPVDZ_P_3P_P, CCPVDZ_P_3D);
    case "S": return heavyShellsRow3("S", CCPVDZ_S_1S, CCPVDZ_S_2S, CCPVDZ_S_3S, CCPVDZ_S_3S_P, CCPVDZ_S_2P, CCPVDZ_S_3P, CCPVDZ_S_3P_P, CCPVDZ_S_3D);
    case "Cl": return heavyShellsRow3("Cl", CCPVDZ_CL_1S, CCPVDZ_CL_2S, CCPVDZ_CL_3S, CCPVDZ_CL_3S_P, CCPVDZ_CL_2P, CCPVDZ_CL_3P, CCPVDZ_CL_3P_P, CCPVDZ_CL_3D);
    case "Ar": return heavyShellsRow3("Ar", CCPVDZ_AR_1S, CCPVDZ_AR_2S, CCPVDZ_AR_3S, CCPVDZ_AR_3S_P, CCPVDZ_AR_2P, CCPVDZ_AR_3P, CCPVDZ_AR_3P_P, CCPVDZ_AR_3D);
    case "He":
      return [
        makeCGShell(CCPVDZ_HE_1S, pos, [0, 0, 0], "He:1s"),
        makeCGShell(CCPVDZ_HE_2S, pos, [0, 0, 0], "He:2s"),
        makeCGShell(CCPVDZ_HE_2P, pos, [1, 0, 0], "He:2p_x"),
        makeCGShell(CCPVDZ_HE_2P, pos, [0, 1, 0], "He:2p_y"),
        makeCGShell(CCPVDZ_HE_2P, pos, [0, 0, 1], "He:2p_z"),
      ];
  }
}

/**
 * Augmentation-only shells for aug-cc-pVDZ — one diffuse function per
 * angular momentum class (s, p, d), appended to the existing cc-pVDZ
 * shell list. H–Ar coverage, all 18 elements.
 */
function atomShellsAugDiffuse(symbol: AtomSymbol, pos: readonly [number, number, number]): CGShell[] {
  type ShellData = { readonly alpha: readonly number[]; readonly c: readonly number[] };
  function heavyDiffuse(sym: string, s: ShellData, p: ShellData, d: ShellData): CGShell[] {
    return [
      makeCGShell(s, pos, [0, 0, 0], `${sym}:aug-s`),
      makeCGShell(p, pos, [1, 0, 0], `${sym}:aug-p_x`),
      makeCGShell(p, pos, [0, 1, 0], `${sym}:aug-p_y`),
      makeCGShell(p, pos, [0, 0, 1], `${sym}:aug-p_z`),
      makeCGShell(d, pos, [2, 0, 0], `${sym}:aug-d_xx`),
      makeCGShell(d, pos, [0, 2, 0], `${sym}:aug-d_yy`),
      makeCGShell(d, pos, [0, 0, 2], `${sym}:aug-d_zz`),
      makeCGShell(d, pos, [1, 1, 0], `${sym}:aug-d_xy`),
      makeCGShell(d, pos, [1, 0, 1], `${sym}:aug-d_xz`),
      makeCGShell(d, pos, [0, 1, 1], `${sym}:aug-d_yz`),
    ];
  }
  switch (symbol) {
    case "H":
      return [
        makeCGShell(AUG_CCPVDZ_H_DIFFUSE_S, pos, [0, 0, 0], "H:aug-s"),
        makeCGShell(AUG_CCPVDZ_H_DIFFUSE_P, pos, [1, 0, 0], "H:aug-p_x"),
        makeCGShell(AUG_CCPVDZ_H_DIFFUSE_P, pos, [0, 1, 0], "H:aug-p_y"),
        makeCGShell(AUG_CCPVDZ_H_DIFFUSE_P, pos, [0, 0, 1], "H:aug-p_z"),
      ];
    case "Li": return heavyDiffuse("Li", AUG_CCPVDZ_LI_DIFFUSE_S, AUG_CCPVDZ_LI_DIFFUSE_P, AUG_CCPVDZ_LI_DIFFUSE_D);
    case "Be": return heavyDiffuse("Be", AUG_CCPVDZ_BE_DIFFUSE_S, AUG_CCPVDZ_BE_DIFFUSE_P, AUG_CCPVDZ_BE_DIFFUSE_D);
    case "B":  return heavyDiffuse("B",  AUG_CCPVDZ_B_DIFFUSE_S,  AUG_CCPVDZ_B_DIFFUSE_P,  AUG_CCPVDZ_B_DIFFUSE_D);
    case "C":  return heavyDiffuse("C",  AUG_CCPVDZ_C_DIFFUSE_S,  AUG_CCPVDZ_C_DIFFUSE_P,  AUG_CCPVDZ_C_DIFFUSE_D);
    case "N":  return heavyDiffuse("N",  AUG_CCPVDZ_N_DIFFUSE_S,  AUG_CCPVDZ_N_DIFFUSE_P,  AUG_CCPVDZ_N_DIFFUSE_D);
    case "O":  return heavyDiffuse("O",  AUG_CCPVDZ_O_DIFFUSE_S,  AUG_CCPVDZ_O_DIFFUSE_P,  AUG_CCPVDZ_O_DIFFUSE_D);
    case "F":  return heavyDiffuse("F",  AUG_CCPVDZ_F_DIFFUSE_S,  AUG_CCPVDZ_F_DIFFUSE_P,  AUG_CCPVDZ_F_DIFFUSE_D);
    case "Ne": return heavyDiffuse("Ne", AUG_CCPVDZ_NE_DIFFUSE_S, AUG_CCPVDZ_NE_DIFFUSE_P, AUG_CCPVDZ_NE_DIFFUSE_D);
    case "Na": return heavyDiffuse("Na", AUG_CCPVDZ_NA_DIFFUSE_S, AUG_CCPVDZ_NA_DIFFUSE_P, AUG_CCPVDZ_NA_DIFFUSE_D);
    case "Mg": return heavyDiffuse("Mg", AUG_CCPVDZ_MG_DIFFUSE_S, AUG_CCPVDZ_MG_DIFFUSE_P, AUG_CCPVDZ_MG_DIFFUSE_D);
    case "Al": return heavyDiffuse("Al", AUG_CCPVDZ_AL_DIFFUSE_S, AUG_CCPVDZ_AL_DIFFUSE_P, AUG_CCPVDZ_AL_DIFFUSE_D);
    case "Si": return heavyDiffuse("Si", AUG_CCPVDZ_SI_DIFFUSE_S, AUG_CCPVDZ_SI_DIFFUSE_P, AUG_CCPVDZ_SI_DIFFUSE_D);
    case "P": return heavyDiffuse("P", AUG_CCPVDZ_P_DIFFUSE_S, AUG_CCPVDZ_P_DIFFUSE_P, AUG_CCPVDZ_P_DIFFUSE_D);
    case "S": return heavyDiffuse("S", AUG_CCPVDZ_S_DIFFUSE_S, AUG_CCPVDZ_S_DIFFUSE_P, AUG_CCPVDZ_S_DIFFUSE_D);
    case "Cl": return heavyDiffuse("Cl", AUG_CCPVDZ_CL_DIFFUSE_S, AUG_CCPVDZ_CL_DIFFUSE_P, AUG_CCPVDZ_CL_DIFFUSE_D);
    case "Ar": return heavyDiffuse("Ar", AUG_CCPVDZ_AR_DIFFUSE_S, AUG_CCPVDZ_AR_DIFFUSE_P, AUG_CCPVDZ_AR_DIFFUSE_D);
    case "He":
      return [
        makeCGShell(AUG_CCPVDZ_HE_DIFFUSE_S, pos, [0, 0, 0], "He:aug-s"),
        makeCGShell(AUG_CCPVDZ_HE_DIFFUSE_P, pos, [1, 0, 0], "He:aug-p_x"),
        makeCGShell(AUG_CCPVDZ_HE_DIFFUSE_P, pos, [0, 1, 0], "He:aug-p_y"),
        makeCGShell(AUG_CCPVDZ_HE_DIFFUSE_P, pos, [0, 0, 1], "He:aug-p_z"),
      ];
  }
}

/** Convert a list of atoms (positions in Å) into CG shells (positions in Bohr) and Nuclei. */
export function moleculeToShellsNuclei(
  atoms: readonly Atom[],
  basis: BasisName = "sto-3g",
): {
  shells: CGShell[];
  nuclei: Nucleus[];
  nElectrons: number;
  /** Atom index that owns each shell (length = shells.length). Required for analytical
   *  HF gradients — when the nucleus moves, every shell whose AO is anchored on it
   *  contributes to the basis-side derivative. Atom index aligns with `nuclei[i]`. */
  shellAtomIdx: number[];
} {
  const shells: CGShell[] = [];
  const nuclei: Nucleus[] = [];
  const shellAtomIdx: number[] = [];
  let nElectrons = 0;
  for (let ai = 0; ai < atoms.length; ai++) {
    const a = atoms[ai]!;
    const pos_bohr: [number, number, number] = [
      a.pos[0] * ANGSTROM_TO_BOHR,
      a.pos[1] * ANGSTROM_TO_BOHR,
      a.pos[2] * ANGSTROM_TO_BOHR,
    ];
    const newShells = atomShells(a.symbol, pos_bohr, basis);
    shells.push(...newShells);
    for (let s = 0; s < newShells.length; s++) shellAtomIdx.push(ai);
    // Ghost atoms contribute their AO basis but no nuclear charge and
    // no electrons (Boys-Bernardi counterpoise). Push a Z=0 nucleus
    // entry so the per-atom index stays aligned with shellAtomIdx but
    // V_ne contributions from this site vanish, and V_nn pair terms
    // involving a ghost are zero.
    const isGhost = a.ghost === true;
    nuclei.push({ Z: isGhost ? 0 : Z_FOR[a.symbol], pos: pos_bohr });
    if (!isGhost) nElectrons += N_ELECTRONS_FOR[a.symbol];
  }
  return { shells, nuclei, nElectrons, shellAtomIdx };
}
