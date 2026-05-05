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
} from "./integrals.js";
import { type CGShell, makeCGShell } from "./integrals-cg.js";
import { type Nucleus } from "./cg-molecular.js";

export type AtomSymbol = "H" | "Li" | "Be" | "C" | "N" | "O";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

export interface Atom {
  readonly symbol: AtomSymbol;
  /** Position in Ångströms (will be converted to Bohr). */
  readonly pos: readonly [number, number, number];
}

/** Atomic number for each supported atom. */
export const Z_FOR: Readonly<Record<AtomSymbol, number>> = {
  H: 1, Li: 3, Be: 4, C: 6, N: 7, O: 8,
};

/** Number of electrons in the neutral atom. */
export const N_ELECTRONS_FOR: Readonly<Record<AtomSymbol, number>> = {
  H: 1, Li: 3, Be: 4, C: 6, N: 7, O: 8,
};

/**
 * Return the full STO-3G CG-shell list for a given atom centered at
 * `pos_bohr`. For first-row (H) we get [1s]. For second-row (Li, Be,
 * C, N, O) we get [1s, 2s, 2p_x, 2p_y, 2p_z]. Lithium intentionally
 * gets only 1s and 2s — its STO-3G 2p contraction is missing from
 * Pople's table for this atom.
 */
export function atomShells(symbol: AtomSymbol, pos_bohr: readonly [number, number, number]): CGShell[] {
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
  }
}

/** Convert a list of atoms (positions in Å) into CG shells (positions in Bohr) and Nuclei. */
export function moleculeToShellsNuclei(atoms: readonly Atom[]): {
  shells: CGShell[];
  nuclei: Nucleus[];
  nElectrons: number;
} {
  const shells: CGShell[] = [];
  const nuclei: Nucleus[] = [];
  let nElectrons = 0;
  for (const a of atoms) {
    const pos_bohr: [number, number, number] = [
      a.pos[0] * ANGSTROM_TO_BOHR,
      a.pos[1] * ANGSTROM_TO_BOHR,
      a.pos[2] * ANGSTROM_TO_BOHR,
    ];
    shells.push(...atomShells(a.symbol, pos_bohr));
    nuclei.push({ Z: Z_FOR[a.symbol], pos: pos_bohr });
    nElectrons += N_ELECTRONS_FOR[a.symbol];
  }
  return { shells, nuclei, nElectrons };
}
