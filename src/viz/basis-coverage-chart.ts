// ─────────────────────────────────────────────────────────────
// basis-coverage-chart.ts — per-atom horizontal stacked bars
// showing how many s / p / d / f / g basis functions each atom
// contributes. Useful for explaining the size of a calculation.
// ─────────────────────────────────────────────────────────────

import {
  CHART_COLORS, chartHeader, chartFooter, escapeXML, fmtNum,
  type ChartFrame,
} from "./chart-base.js";

export interface AtomBasis {
  readonly symbol: string;
  /** Per-shell counts: [n_s, n_p, n_d, n_f, n_g]. Trailing zeros may be omitted. */
  readonly counts: ReadonlyArray<number>;
}

export interface BasisCoverageOpts {
  readonly title?: string;
  readonly subtitle?: string;
  readonly frame?: ChartFrame;
}

const SHELL_COLORS = [
  CHART_COLORS.cyan,
  CHART_COLORS.amber,
  CHART_COLORS.lavender,
  CHART_COLORS.lime,
  CHART_COLORS.pink,
];
const SHELL_LABELS = ["s", "p", "d", "f", "g"];

export function basisCoverageChart(
  atoms: readonly AtomBasis[],
  opts: BasisCoverageOpts = {},
): string {
  const frame: ChartFrame = opts.frame ?? {
    width: 640, height: Math.max(140, 60 + atoms.length * 32),
    margin: { top: 60, right: 30, bottom: 40, left: 90 },
  };
  const title = opts.title ?? "Basis-set coverage";

  const { open, plotW, plotH } = chartHeader(frame, title);
  if (atoms.length === 0) {
    return open
      + `<text x="${plotW / 2}" y="${plotH / 2}" font-size="14" fill="${CHART_COLORS.textFaded}" text-anchor="middle">no atoms</text>`
      + chartFooter();
  }

  // Compute totals + max for bar scaling.
  let maxTotal = 0;
  let grandTotal = 0;
  const totals = atoms.map((a) => {
    let t = 0;
    for (const c of a.counts) t += c;
    if (t > maxTotal) maxTotal = t;
    grandTotal += t;
    return t;
  });
  if (maxTotal === 0) maxTotal = 1;

  let body = `<text x="${plotW / 2}" y="-38" font-size="14" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle">${escapeXML(title)}</text>`;
  body += `<text x="${plotW / 2}" y="-22" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${atoms.length} atoms · ${grandTotal} basis functions${opts.subtitle ? ` · ${escapeXML(opts.subtitle)}` : ""}</text>`;

  // Legend (shell types).
  let legX = 0;
  for (let s = 0; s < SHELL_LABELS.length; s++) {
    body += `<rect x="${legX}" y="-10" width="10" height="10" rx="2" fill="${SHELL_COLORS[s]!}"/>`;
    body += `<text x="${legX + 14}" y="-1" font-size="10" fill="${CHART_COLORS.textMid}" font-family="ui-monospace, monospace">${SHELL_LABELS[s]}</text>`;
    legX += 36;
  }

  const rowH = plotH / atoms.length;
  const barH = Math.min(rowH * 0.6, 20);
  const offY = (rowH - barH) / 2;

  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i]!;
    const total = totals[i]!;
    const yTop = i * rowH + offY;
    body += `<text x="-10" y="${yTop + barH / 2 + 4}" font-size="11" fill="${CHART_COLORS.text}" text-anchor="end" font-family="ui-monospace, monospace">${escapeXML(a.symbol)} #${i}</text>`;
    // Each shell stacked horizontally.
    let xCursor = 0;
    for (let s = 0; s < a.counts.length; s++) {
      const c = a.counts[s] ?? 0;
      if (c <= 0) continue;
      const w = (c / maxTotal) * plotW;
      const color = SHELL_COLORS[s] ?? CHART_COLORS.textFaded;
      body += `<rect x="${xCursor.toFixed(1)}" y="${yTop.toFixed(1)}" width="${w.toFixed(1)}" height="${barH.toFixed(1)}" fill="${color}" fill-opacity="0.85"/>`;
      if (w > 16) {
        body += `<text x="${(xCursor + w / 2).toFixed(1)}" y="${(yTop + barH / 2 + 4).toFixed(1)}" font-size="10" fill="#020617" font-weight="700" text-anchor="middle" font-family="ui-monospace, monospace">${c}${SHELL_LABELS[s]}</text>`;
      }
      xCursor += w;
    }
    // Total at the end.
    body += `<text x="${(xCursor + 6).toFixed(1)}" y="${(yTop + barH / 2 + 4).toFixed(1)}" font-size="10" fill="${CHART_COLORS.textMid}" font-family="ui-monospace, monospace">${total} fn</text>`;
  }

  return open + body + chartFooter();
}

/** Helper: AO shell-count summary string e.g. "5s 4p 1d". */
export function basisSummary(counts: ReadonlyArray<number>): string {
  const parts: string[] = [];
  for (let s = 0; s < counts.length; s++) {
    const c = counts[s] ?? 0;
    if (c > 0) parts.push(`${c}${SHELL_LABELS[s] ?? "?"}`);
  }
  return parts.length === 0 ? "—" : parts.join(" ");
}

/** Helper: fmtNum re-export consumers may want. */
export { fmtNum };
