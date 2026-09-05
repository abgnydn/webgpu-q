// ─────────────────────────────────────────────────────────────
// molecular-graph-2d.ts — projection of a 3D molecule onto a 2D
// SVG. Atoms are filled circles colored by element; bonds are
// derived from interatomic distance vs. a covalent-radius sum
// threshold (no inputs required beyond the atom list).
//
// Optional overlays:
//   - charges: Mulliken atomic charges (+/− with a divergent
//     color ring around each atom + numeric label).
//   - spin: open-shell spin populations (Sz per atom).
//   - labels: explicit per-atom strings (overrides element symbol).
// ─────────────────────────────────────────────────────────────

import {
  CHART_COLORS, chartHeader, chartFooter, escapeXML, fmtNum,
  type ChartFrame,
} from "./chart-base.js";

export interface AtomLite {
  readonly symbol: string;
  /** Cartesian position in Angstroms (or Bohr — units only affect bond detection threshold). */
  readonly pos: readonly [number, number, number];
}

export interface MoleculeGraph2DOpts {
  readonly title?: string;
  readonly charges?: ReadonlyArray<number>;
  readonly spinPopulations?: ReadonlyArray<number>;
  readonly atomLabels?: ReadonlyArray<string>;
  /** Project onto: xy (default), xz, or yz. */
  readonly projection?: "xy" | "xz" | "yz";
  /** Covalent bond radius cutoff (Å). Default 1.8. */
  readonly bondCutoffAngstrom?: number;
  readonly frame?: ChartFrame;
}

const ATOM_COLOR: Record<string, string> = {
  H:  "#e2e8f0",
  He: "#fbbf24",
  Li: "#c084fc",
  Be: "#22d3ee",
  B:  "#ffb56e",
  C:  "#64748b",
  N:  "#6ea8ff",
  O:  "#f87171",
  F:  "#34d399",
  Ne: "#a78bfa",
  Na: "#c084fc",
  Mg: "#22d3ee",
  Al: "#94a3b8",
  Si: "#facc15",
  P:  "#ffb56e",
  S:  "#fbbf24",
  Cl: "#34d399",
  Ar: "#a78bfa",     // noble — same hue as Ne, matching the group-inherits-colour pattern
};

const ATOM_RADIUS: Record<string, number> = {
  H: 6, He: 6,
  Li: 10, Be: 9, B: 9, C: 9, N: 9, O: 9, F: 9, Ne: 9,
  Na: 12, Mg: 12, Al: 12, Si: 12, P: 12, S: 12, Cl: 11, Ar: 11,
};

export function moleculeGraph2D(
  atoms: readonly AtomLite[],
  opts: MoleculeGraph2DOpts = {},
): string {
  const frame: ChartFrame = opts.frame ?? {
    width: 520, height: 380,
    margin: { top: 50, right: 30, bottom: 30, left: 30 },
  };
  const title = opts.title ?? "Molecule";
  const projection = opts.projection ?? "xy";
  const bondCutoff = opts.bondCutoffAngstrom ?? 1.8;

  const { open, plotW, plotH } = chartHeader(frame, title);

  if (atoms.length === 0) {
    return open
      + `<text x="${plotW / 2}" y="${plotH / 2}" font-size="14" fill="${CHART_COLORS.textFaded}" text-anchor="middle">empty molecule</text>`
      + chartFooter();
  }

  // Project to (u, v) coords.
  const project = (p: readonly [number, number, number]): [number, number] => {
    if (projection === "xz") return [p[0], p[2]];
    if (projection === "yz") return [p[1], p[2]];
    return [p[0], p[1]];
  };
  const coords2D = atoms.map((a) => project(a.pos));

  // Compute bounds; flip y so it points up.
  let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (const [u, v] of coords2D) {
    if (u < uMin) uMin = u; if (u > uMax) uMax = u;
    if (v < vMin) vMin = v; if (v > vMax) vMax = v;
  }
  const span = Math.max(uMax - uMin, vMax - vMin, 0.5);
  const cx = (uMin + uMax) / 2;
  const cy = (vMin + vMax) / 2;
  const pad = 30;
  const usable = Math.min(plotW, plotH) - 2 * pad;
  const scale = usable / span;
  const screen = (i: number): [number, number] => {
    const [u, v] = coords2D[i]!;
    const sx = plotW / 2 + (u - cx) * scale;
    const sy = plotH / 2 - (v - cy) * scale; // flip y
    return [sx, sy];
  };

  let body = `<text x="${plotW / 2}" y="-28" font-size="14" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle">${escapeXML(title)}</text>`;
  body += `<text x="${plotW / 2}" y="-10" font-size="10" fill="${CHART_COLORS.textFaded}" text-anchor="middle" font-family="ui-monospace, monospace">${atoms.length} atoms · ${projection} projection</text>`;

  // Detect bonds: full 3D distance, not the projected one (more accurate).
  let bondSvg = "";
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      const ai = atoms[i]!.pos, aj = atoms[j]!.pos;
      const dx = ai[0] - aj[0], dy = ai[1] - aj[1], dz = ai[2] - aj[2];
      const R = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (R < bondCutoff) {
        const [x1, y1] = screen(i);
        const [x2, y2] = screen(j);
        bondSvg += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${CHART_COLORS.textMid}" stroke-width="3" stroke-linecap="round" opacity="0.65"/>`;
      }
    }
  }
  body += bondSvg;

  // Determine charge color range (divergent).
  const charges = opts.charges;
  let chargeAbsMax = 0;
  if (charges) {
    for (const q of charges) {
      if (Math.abs(q) > chargeAbsMax) chargeAbsMax = Math.abs(q);
    }
    if (chargeAbsMax === 0) chargeAbsMax = 1;
  }

  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i]!;
    const [x, y] = screen(i);
    const r = ATOM_RADIUS[a.symbol] ?? 10;
    const color = ATOM_COLOR[a.symbol] ?? CHART_COLORS.textMid;
    // Charge ring (if charges provided).
    if (charges) {
      const q = charges[i] ?? 0;
      const t = q / chargeAbsMax; // -1..+1
      // red for positive, cyan for negative.
      const ring = t > 0 ? CHART_COLORS.red : CHART_COLORS.cyan;
      const ringW = 2 + 5 * Math.abs(t);
      body += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r + ringW + 1}" fill="none" stroke="${ring}" stroke-opacity="${(0.25 + 0.6 * Math.abs(t)).toFixed(2)}" stroke-width="${ringW.toFixed(2)}"/>`;
    }
    body += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${color}" stroke="#020617" stroke-width="1.5"/>`;
    const label = opts.atomLabels?.[i] ?? a.symbol;
    body += `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" font-size="${a.symbol === "H" ? 10 : 11}" fill="#020617" font-weight="700" text-anchor="middle" font-family="ui-monospace, monospace">${escapeXML(label)}</text>`;
    // Numeric overlay below the atom.
    if (charges) {
      const q = charges[i] ?? 0;
      body += `<text x="${x.toFixed(1)}" y="${(y + r + 14).toFixed(1)}" font-size="10" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${q >= 0 ? "+" : ""}${fmtNum(q, 3)}</text>`;
    } else if (opts.spinPopulations) {
      const sz = opts.spinPopulations[i] ?? 0;
      body += `<text x="${x.toFixed(1)}" y="${(y + r + 14).toFixed(1)}" font-size="10" fill="${sz > 0 ? CHART_COLORS.amber : CHART_COLORS.cyan}" text-anchor="middle" font-family="ui-monospace, monospace">${fmtNum(sz, 3)}</text>`;
    }
  }

  // Legend (only when overlay present).
  if (charges) {
    body += `<g transform="translate(0 ${plotH + 10})">`;
    body += `<circle cx="8" cy="-5" r="6" fill="none" stroke="${CHART_COLORS.red}" stroke-width="3"/>`;
    body += `<text x="22" y="-1" font-size="10" fill="${CHART_COLORS.textMid}">δ+ Mulliken</text>`;
    body += `<circle cx="118" cy="-5" r="6" fill="none" stroke="${CHART_COLORS.cyan}" stroke-width="3"/>`;
    body += `<text x="132" y="-1" font-size="10" fill="${CHART_COLORS.textMid}">δ−  ·  range [${fmtNum(-chargeAbsMax, 3)}, ${fmtNum(chargeAbsMax, 3)}] e</text>`;
    body += `</g>`;
  }

  return open + body + chartFooter();
}

/** Convenience wrapper: bare atom diagram with element symbols. */
export function moleculeStructure(
  atoms: readonly AtomLite[],
  opts: Omit<MoleculeGraph2DOpts, "charges" | "spinPopulations"> = {},
): string {
  return moleculeGraph2D(atoms, opts);
}
