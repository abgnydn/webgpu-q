// ─────────────────────────────────────────────────────────────
// beh2-full-builder.ts — BeH₂ in full STO-3G (Be 1s + 2s + 2p
// + 2 H 1s = 7 spatial / 14 spin-orbitals). Phase C v3 stage 2.
//
// This is the first molecule in the project that includes a
// p-shell. The full Hilbert space (2^14 = 16384) won't fit as a
// dense matrix in a browser tab (2 GB), so we project on the fly
// to the N=6 sector (C(14, 6) = 3003-dim → 72 MB) using
// `buildSectorH` from sector-builder.ts.
//
// Geometry: linear H–Be–H, Be at origin, H at (0, 0, ±R).
// HF reference: closed-shell singlet, 6 electrons in the 3
// lowest spatial OAO orbitals (Be 1s, Be 2s, and one of the
// symmetric H combinations after Löwdin).
// ─────────────────────────────────────────────────────────────

import {
  STO3G_BE_1S, STO3G_BE_2S, STO3G_BE_2P, STO3G_H_1S,
} from "./integrals.js";
import { type CGShell, makeCGShell } from "./integrals-cg.js";
import { computeMolecularIntegrals, type MolecularIntegrals, type Nucleus } from "./cg-molecular.js";
import { buildSectorH, type SectorH } from "./sector-builder.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

type Center = readonly [number, number, number];

export interface BeH2FullData {
  /** Be–H bond length (Bohr). */
  readonly R_Bohr: number;
  /** Atom centers — [0]=Be, [1]=H_left, [2]=H_right. */
  readonly centers: readonly [Center, Center, Center];
  /** 7 atomic shells in spatial-orbital order:
   *  0: Be 1s, 1: Be 2s, 2: Be 2p_x, 3: Be 2p_y, 4: Be 2p_z,
   *  5: H_left 1s, 6: H_right 1s. */
  readonly shells: readonly CGShell[];
  /** Molecular integrals (S, h, ERI in both AO and OAO bases + Vnn). */
  readonly integrals: MolecularIntegrals;
  /** Sector-projected Hamiltonian for the N=6 (neutral) sector. */
  readonly sector: SectorH;
}

export const BEH2_FULL_N_SPATIAL = 7;
export const BEH2_FULL_N_QUBITS = 14;
export const BEH2_FULL_PARTICLE_NUMBER = 6;

/** Build the full STO-3G BeH₂ Hamiltonian, projected to the N=6 sector. */
export function buildBeH2Full(R_angstrom: number): BeH2FullData {
  const R = R_angstrom * ANGSTROM_TO_BOHR;
  const Be: Center = [0, 0, 0];
  const HL: Center = [0, 0, -R];
  const HR: Center = [0, 0, +R];

  const shells: CGShell[] = [
    makeCGShell(STO3G_BE_1S, Be, [0, 0, 0], "Be:1s"),
    makeCGShell(STO3G_BE_2S, Be, [0, 0, 0], "Be:2s"),
    makeCGShell(STO3G_BE_2P, Be, [1, 0, 0], "Be:2p_x"),
    makeCGShell(STO3G_BE_2P, Be, [0, 1, 0], "Be:2p_y"),
    makeCGShell(STO3G_BE_2P, Be, [0, 0, 1], "Be:2p_z"),
    makeCGShell(STO3G_H_1S, HL, [0, 0, 0], "H_L:1s"),
    makeCGShell(STO3G_H_1S, HR, [0, 0, 0], "H_R:1s"),
  ];
  const nuclei: Nucleus[] = [
    { Z: 4, pos: Be },
    { Z: 1, pos: HL },
    { Z: 1, pos: HR },
  ];

  const integrals = computeMolecularIntegrals(shells, nuclei);
  const sector = buildSectorH(
    BEH2_FULL_N_SPATIAL,
    integrals.h_OAO,
    integrals.eri_OAO,
    integrals.Vnn,
    BEH2_FULL_PARTICLE_NUMBER,
  );

  return { R_Bohr: R, centers: [Be, HL, HR] as const, shells, integrals, sector };
}
