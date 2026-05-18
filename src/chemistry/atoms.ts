// ─────────────────────────────────────────────────────────────
// atoms.ts — atom registry. Maps a chemical symbol to (Z, full
// STO-3G shell list at a given center). Keeps the per-molecule
// builder code thin: every molecule is just a list of (symbol,
// position) tuples that this module turns into the CG shells +
// nuclei the chemistry pipeline expects.
// ─────────────────────────────────────────────────────────────

import {
  STO3G_H_1S,
  STO3G_LI_1S, STO3G_LI_2S,
  STO3G_BE_1S, STO3G_BE_2S, STO3G_BE_2P,
  STO3G_C_1S, STO3G_C_2S, STO3G_C_2P,
  STO3G_N_1S, STO3G_N_2S, STO3G_N_2P,
  STO3G_O_1S, STO3G_O_2S, STO3G_O_2P,
  STO3G_F_1S, STO3G_F_2S, STO3G_F_2P,
  CCPVDZ_H_1S, CCPVDZ_H_2S, CCPVDZ_H_2P,
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
  AUG_CCPVDZ_LI_DIFFUSE_S, AUG_CCPVDZ_LI_DIFFUSE_P, AUG_CCPVDZ_LI_DIFFUSE_D,
  AUG_CCPVDZ_BE_DIFFUSE_S, AUG_CCPVDZ_BE_DIFFUSE_P, AUG_CCPVDZ_BE_DIFFUSE_D,
  AUG_CCPVDZ_C_DIFFUSE_S,  AUG_CCPVDZ_C_DIFFUSE_P,  AUG_CCPVDZ_C_DIFFUSE_D,
  AUG_CCPVDZ_N_DIFFUSE_S,  AUG_CCPVDZ_N_DIFFUSE_P,  AUG_CCPVDZ_N_DIFFUSE_D,
  AUG_CCPVDZ_O_DIFFUSE_S, AUG_CCPVDZ_O_DIFFUSE_P, AUG_CCPVDZ_O_DIFFUSE_D,
  AUG_CCPVDZ_F_DIFFUSE_S,  AUG_CCPVDZ_F_DIFFUSE_P,  AUG_CCPVDZ_F_DIFFUSE_D,
} from "./integrals.js";
import { type CGShell, makeCGShell } from "./integrals-cg.js";
import { type Nucleus } from "./cg-molecular.js";

export type AtomSymbol = "H" | "Li" | "Be" | "C" | "N" | "O" | "F";
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
  H: 1, Li: 3, Be: 4, C: 6, N: 7, O: 8, F: 9,
};

/** Number of electrons in the neutral atom. */
export const N_ELECTRONS_FOR: Readonly<Record<AtomSymbol, number>> = {
  H: 1, Li: 3, Be: 4, C: 6, N: 7, O: 8, F: 9,
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
  H: 0, Li: 1, Be: 1, C: 1, N: 1, O: 1, F: 1,
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
  }
}

/** cc-pVDZ shells. H + first-row (Li, Be, C, N, O, F). */
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
    case "C":  return heavyShells("C",  CCPVDZ_C_1S,  CCPVDZ_C_2S,  CCPVDZ_C_2S_P,  CCPVDZ_C_2P,  CCPVDZ_C_2P_P,  CCPVDZ_C_3D);
    case "N":  return heavyShells("N",  CCPVDZ_N_1S,  CCPVDZ_N_2S,  CCPVDZ_N_2S_P,  CCPVDZ_N_2P,  CCPVDZ_N_2P_P,  CCPVDZ_N_3D);
    case "O":  return heavyShells("O",  CCPVDZ_O_1S,  CCPVDZ_O_2S,  CCPVDZ_O_2S_P,  CCPVDZ_O_2P,  CCPVDZ_O_2P_P,  CCPVDZ_O_3D);
    case "F":  return heavyShells("F",  CCPVDZ_F_1S,  CCPVDZ_F_2S,  CCPVDZ_F_2S_P,  CCPVDZ_F_2P,  CCPVDZ_F_2P_P,  CCPVDZ_F_3D);
  }
}

/**
 * Augmentation-only shells for aug-cc-pVDZ — one diffuse function per
 * angular momentum class (s, p, d), appended to the existing cc-pVDZ
 * shell list. H + first-row coverage (H, Li, Be, C, N, O, F).
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
    case "C":  return heavyDiffuse("C",  AUG_CCPVDZ_C_DIFFUSE_S,  AUG_CCPVDZ_C_DIFFUSE_P,  AUG_CCPVDZ_C_DIFFUSE_D);
    case "N":  return heavyDiffuse("N",  AUG_CCPVDZ_N_DIFFUSE_S,  AUG_CCPVDZ_N_DIFFUSE_P,  AUG_CCPVDZ_N_DIFFUSE_D);
    case "O":  return heavyDiffuse("O",  AUG_CCPVDZ_O_DIFFUSE_S,  AUG_CCPVDZ_O_DIFFUSE_P,  AUG_CCPVDZ_O_DIFFUSE_D);
    case "F":  return heavyDiffuse("F",  AUG_CCPVDZ_F_DIFFUSE_S,  AUG_CCPVDZ_F_DIFFUSE_P,  AUG_CCPVDZ_F_DIFFUSE_D);
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
