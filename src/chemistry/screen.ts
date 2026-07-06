// ─────────────────────────────────────────────────────────────
// screen.ts — parameterized molecular screening over a library.
//
// The reusable "run the property suite on every molecule in a set
// and rank them" primitive. Given a list of molecules (as Atom[]
// or raw XYZ text) it runs quickReport per molecule, collects a
// standardized row (energy, HOMO–LUMO gap, dipole, Koopmans IP,
// multireference verdict), and returns a ranked artifact.
//
// This is the headless entry point a driver (Playwright / Modal /
// the swarm) calls to turn one molecule → full report into a batch
// screen. Pure + synchronous (CPU/WASM path via quickReport); the
// swarm distributes it by handing disjoint molecule slices to tabs.
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "./atoms.js";
import { parseXYZ } from "./xyz.js";
import { quickReport, type QuickReportOpts } from "./quick-report.js";

/** One molecule to screen. Provide `atoms` OR `xyz` (raw XYZ text). */
export interface ScreenMolecule {
  readonly name: string;
  readonly atoms?: readonly Atom[];
  readonly xyz?: string;
  /** Open-shell: both required together (forwarded to quickReport). */
  readonly nAlpha?: number;
  readonly nBeta?: number;
}

export type ScreenRankBy = "gap" | "dipole" | "energy" | "ip";

export interface ScreenOpts {
  /** Basis set. Default sto-3g. */
  readonly basis?: BasisName;
  /** Descriptor to rank by. Default "gap" (HOMO–LUMO gap, eV). */
  readonly rankBy?: ScreenRankBy;
  /** Sort direction. Default true (largest first). Errored rows always sink. */
  readonly descending?: boolean;
  /** Add D2 dispersion to each report. */
  readonly addD2?: boolean;
}

export interface ScreenRow {
  readonly name: string;
  readonly status: "ok" | "error";
  readonly nAtoms?: number;
  readonly totalEnergy?: number;
  readonly homoLumoGapEv?: number;
  readonly dipoleDebye?: number;
  readonly koopmansIPEv?: number;
  readonly multireference?: string;
  readonly error?: string;
}

export interface ScreenArtifact {
  readonly meta: {
    readonly kind: "screen";
    readonly basis: BasisName;
    readonly rankBy: ScreenRankBy;
    readonly descending: boolean;
    readonly count: number;
  };
  /** Rows, ranked by the chosen descriptor (errored rows at the bottom). */
  readonly rows: readonly ScreenRow[];
  /** "pass" when every molecule ran; "partial" when one or more errored. */
  readonly status: "pass" | "partial";
  readonly diagnosis?: string;
}

function rankKey(row: ScreenRow, rankBy: ScreenRankBy): number {
  switch (rankBy) {
    case "gap":    return row.homoLumoGapEv ?? 0;
    case "dipole": return row.dipoleDebye ?? 0;
    case "ip":     return row.koopmansIPEv ?? 0;
    case "energy": return row.totalEnergy ?? 0;
  }
}

/**
 * Screen a library of molecules and return a ranked artifact. A failing
 * molecule (bad geometry, SCF divergence, unsupported element) becomes an
 * `error` row instead of aborting the whole batch, and the artifact status
 * flips to "partial" — so one bad input never loses the rest of the screen.
 */
export function screenMolecules(
  molecules: readonly ScreenMolecule[],
  opts: ScreenOpts = {},
): ScreenArtifact {
  const basis = opts.basis ?? "sto-3g";
  const rankBy = opts.rankBy ?? "gap";
  const descending = opts.descending ?? true;

  const rows: ScreenRow[] = molecules.map((m): ScreenRow => {
    try {
      const atoms = m.atoms ?? (m.xyz !== undefined ? parseXYZ(m.xyz).atoms : undefined);
      if (!atoms || atoms.length === 0) {
        throw new Error("no atoms — provide `atoms` or non-empty `xyz`");
      }
      const qopts: QuickReportOpts = {
        basis,
        addD2: opts.addD2,
        ...(m.nAlpha !== undefined ? { nAlpha: m.nAlpha } : {}),
        ...(m.nBeta !== undefined ? { nBeta: m.nBeta } : {}),
      };
      const r = quickReport(atoms, qopts);
      return {
        name: m.name,
        status: "ok",
        nAtoms: atoms.length,
        totalEnergy: r.totalEnergy,
        homoLumoGapEv: r.homoLumoGapEv,
        dipoleDebye: r.dipoleMagnitudeDebye,
        koopmansIPEv: r.koopmansIPEv,
        multireference: r.multireferenceSeverity,
      };
    } catch (e) {
      return { name: m.name, status: "error", error: e instanceof Error ? e.message : String(e) };
    }
  });

  // Rank ok rows by the descriptor; errored rows always sink to the bottom.
  const sink = descending ? -Infinity : Infinity;
  const ranked = [...rows].sort((a, b) => {
    const ka = a.status === "ok" ? rankKey(a, rankBy) : sink;
    const kb = b.status === "ok" ? rankKey(b, rankBy) : sink;
    return descending ? kb - ka : ka - kb;
  });

  const nErr = rows.reduce((k, r) => k + (r.status === "error" ? 1 : 0), 0);
  return {
    meta: { kind: "screen", basis, rankBy, descending, count: molecules.length },
    rows: ranked,
    status: nErr === 0 ? "pass" : "partial",
    ...(nErr > 0 ? { diagnosis: `${nErr}/${molecules.length} molecule(s) failed to screen` } : {}),
  };
}
