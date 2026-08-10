// ─────────────────────────────────────────────────────────────
// periodic-table.ts — atomic property database for the elements
// supported by webgpu-q (H, He, Li, Be, B, C, N, O, F, Ne,
// Na, Mg, Al, Si, P, S, Cl, Ar) — periods 1-3 complete.
//
// Data sources:
//   - Atomic number, isotope mass: NIST 2024.
//   - Electronegativity: Pauling scale, NIST chemistry webbook.
//   - Covalent radius: Pyykkö & Atsumi 2009 (CEU 15, 12770).
//   - Van der Waals radius: Bondi 1964 / Alvarez 2013.
//   - Period / group / family: standard periodic table.
// ─────────────────────────────────────────────────────────────

import type { AtomSymbol } from "./atoms.js";

export type ElementGroup = "alkali" | "alkaline-earth" | "halogen" | "noble" | "main";

export interface ElementProperties {
  readonly symbol: AtomSymbol;
  readonly name: string;
  readonly atomicNumber: number;
  /** Most-abundant isotope mass in amu (atomic mass units). */
  readonly mass: number;
  /** Pauling electronegativity (dimensionless). */
  readonly electronegativity: number | null;
  /** Covalent radius in Å (single bond), Pyykkö-Atsumi 2009. */
  readonly covalentRadius: number;
  /** Van der Waals radius in Å, Bondi 1964 / Alvarez 2013. */
  readonly vdwRadius: number;
  /** Period (row) in the periodic table. */
  readonly period: 1 | 2 | 3 | 4 | 5 | 6 | 7;
  /** Group (column) — main-group convention 1-18. */
  readonly group: number;
  readonly family: ElementGroup;
}

export const PERIODIC_TABLE: Readonly<Record<AtomSymbol, ElementProperties>> = {
  H: {
    symbol: "H", name: "Hydrogen", atomicNumber: 1,
    mass: 1.00782503207, electronegativity: 2.20,
    covalentRadius: 0.32, vdwRadius: 1.20,
    period: 1, group: 1, family: "main",
  },
  He: {
    symbol: "He", name: "Helium", atomicNumber: 2,
    mass: 4.002603254, electronegativity: null,
    covalentRadius: 0.46, vdwRadius: 1.40,
    period: 1, group: 18, family: "noble",
  },
  Li: {
    symbol: "Li", name: "Lithium", atomicNumber: 3,
    mass: 7.0160034366, electronegativity: 0.98,
    covalentRadius: 1.33, vdwRadius: 1.82,
    period: 2, group: 1, family: "alkali",
  },
  Be: {
    symbol: "Be", name: "Beryllium", atomicNumber: 4,
    mass: 9.012183065, electronegativity: 1.57,
    covalentRadius: 1.02, vdwRadius: 1.53,
    period: 2, group: 2, family: "alkaline-earth",
  },
  B: {
    symbol: "B", name: "Boron", atomicNumber: 5,
    mass: 11.00930536, electronegativity: 2.04,
    covalentRadius: 0.85, vdwRadius: 1.92,
    period: 2, group: 13, family: "main",
  },
  C: {
    symbol: "C", name: "Carbon", atomicNumber: 6,
    mass: 12.0, electronegativity: 2.55,
    covalentRadius: 0.75, vdwRadius: 1.70,
    period: 2, group: 14, family: "main",
  },
  N: {
    symbol: "N", name: "Nitrogen", atomicNumber: 7,
    mass: 14.0030740048, electronegativity: 3.04,
    covalentRadius: 0.71, vdwRadius: 1.55,
    period: 2, group: 15, family: "main",
  },
  O: {
    symbol: "O", name: "Oxygen", atomicNumber: 8,
    mass: 15.99491461956, electronegativity: 3.44,
    covalentRadius: 0.63, vdwRadius: 1.52,
    period: 2, group: 16, family: "main",
  },
  F: {
    symbol: "F", name: "Fluorine", atomicNumber: 9,
    mass: 18.998403163, electronegativity: 3.98,
    covalentRadius: 0.64, vdwRadius: 1.47,
    period: 2, group: 17, family: "halogen",
  },
  Ne: {
    symbol: "Ne", name: "Neon", atomicNumber: 10,
    mass: 19.9924401762, electronegativity: null,
    covalentRadius: 0.67, vdwRadius: 1.54,
    period: 2, group: 18, family: "noble",
  },
  Na: {
    symbol: "Na", name: "Sodium", atomicNumber: 11,
    mass: 22.989769282, electronegativity: 0.93,
    covalentRadius: 1.55, vdwRadius: 2.27,
    period: 3, group: 1, family: "alkali",
  },
  Mg: {
    symbol: "Mg", name: "Magnesium", atomicNumber: 12,
    mass: 23.985041697, electronegativity: 1.31,
    covalentRadius: 1.39, vdwRadius: 1.73,
    period: 3, group: 2, family: "alkaline-earth",
  },
  Al: {
    symbol: "Al", name: "Aluminium", atomicNumber: 13,
    mass: 26.98153853, electronegativity: 1.61,
    covalentRadius: 1.26, vdwRadius: 1.84,
    period: 3, group: 13, family: "main",
  },
  Si: {
    symbol: "Si", name: "Silicon", atomicNumber: 14,
    mass: 27.97692653465, electronegativity: 1.90,
    covalentRadius: 1.16, vdwRadius: 2.10,
    period: 3, group: 14, family: "main",
  },
  P: {
    symbol: "P", name: "Phosphorus", atomicNumber: 15,
    mass: 30.97376199842, electronegativity: 2.19,
    covalentRadius: 1.11, vdwRadius: 1.80,
    period: 3, group: 15, family: "main",
  },
  S: {
    symbol: "S", name: "Sulfur", atomicNumber: 16,
    mass: 31.9720711744, electronegativity: 2.58,
    covalentRadius: 1.03, vdwRadius: 1.80,
    period: 3, group: 16, family: "main",
  },
  Cl: {
    symbol: "Cl", name: "Chlorine", atomicNumber: 17,
    mass: 34.968852682, electronegativity: 3.16,
    covalentRadius: 0.99, vdwRadius: 1.75,
    period: 3, group: 17, family: "halogen",
  },
  Ar: {
    symbol: "Ar", name: "Argon", atomicNumber: 18,
    mass: 39.9623831237, electronegativity: null,
    covalentRadius: 0.96, vdwRadius: 1.88,
    period: 3, group: 18, family: "noble",
  },
};

/** Look up element data. Throws on unknown symbol. */
export function elementInfo(symbol: AtomSymbol): ElementProperties {
  const info = PERIODIC_TABLE[symbol];
  if (!info) throw new Error(`elementInfo: unknown element ${symbol}`);
  return info;
}

/** Molecular weight in g/mol = sum of isotope-weighted atomic masses. */
export function molecularWeight(symbols: readonly AtomSymbol[]): number {
  let m = 0;
  for (const sym of symbols) m += PERIODIC_TABLE[sym].mass;
  return m;
}
