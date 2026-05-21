// ─────────────────────────────────────────────────────────────
// convergence-trace.ts — log-scale convergence trace for any
// iterative solver (SCF / DIIS / CCSD / EOM-CCSD / DMRG / VQE).
//
// Multiple metrics may be overlaid; the convention is that one
// curve is the canonical convergence quantity (often |ΔE|) and
// optional secondary curves are |density-error| / commutator norms /
// residuals.
// ─────────────────────────────────────────────────────────────

import {
  CHART_COLORS, SERIES_PALETTE, chartHeader, chartFooter,
  escapeXML, fmtNum, linearScale, niceTicks, polyline,
  type ChartFrame,
} from "./chart-base.js";

export interface ConvergenceSeries {
  readonly name: string;
  /** Per-iteration values; non-positive entries are clamped to 1e-16 for the log axis. */
  readonly values: ReadonlyArray<number>;
}

export interface ConvergenceTraceOpts {
  readonly title?: string;
  readonly threshold?: number;
  readonly xLabel?: string;
  readonly frame?: ChartFrame;
}

export function convergenceTrace(
  seriesList: readonly ConvergenceSeries[],
  opts: ConvergenceTraceOpts = {},
): string {
  const frame: ChartFrame = opts.frame ?? {
    width: 700, height: 340,
    margin: { top: 50, right: 130, bottom: 50, left: 60 },
  };
  const title = opts.title ?? "Convergence";
  const xLabel = opts.xLabel ?? "iteration";

  const { open, plotW, plotH } = chartHeader(frame, title);
  if (seriesList.length === 0) {
    return open
      + `<text x="${plotW / 2}" y="${plotH / 2}" font-size="14" fill="${CHART_COLORS.textFaded}" text-anchor="middle">no series</text>`
      + chartFooter();
  }

  let xMax = 0;
  let logMin = Infinity, logMax = -Infinity;
  for (const s of seriesList) {
    if (s.values.length > xMax) xMax = s.values.length;
    for (const v of s.values) {
      const clamped = Math.max(Math.abs(v), 1e-16);
      const lg = Math.log10(clamped);
      if (lg < logMin) logMin = lg;
      if (lg > logMax) logMax = lg;
    }
  }
  if (!Number.isFinite(logMin) || !Number.isFinite(logMax)) {
    logMin = -10; logMax = 0;
  }
  if (logMin === logMax) { logMin -= 1; logMax += 1; }

  const scaleX = linearScale(0, Math.max(xMax - 1, 1), 0, plotW);
  const scaleY = linearScale(logMin, logMax, plotH, 0);

  // Title.
  let body = `<text x="${plotW / 2}" y="-28" font-size="14" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle">${escapeXML(title)}</text>`;
  body += `<text x="${plotW / 2}" y="-10" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">log₁₀ scale · ${xMax} iterations</text>`;

  // Grid + axis (integer powers of 10).
  const yTicks: number[] = [];
  for (let p = Math.ceil(logMin); p <= Math.floor(logMax); p++) yTicks.push(p);
  for (const p of yTicks) {
    body += `<line x1="0" y1="${scaleY(p)}" x2="${plotW}" y2="${scaleY(p)}" stroke="${CHART_COLORS.textFaded}" stroke-opacity="0.15"/>`;
    body += `<text x="-8" y="${scaleY(p) + 4}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="end" font-family="ui-monospace, monospace">10${supscript(p)}</text>`;
  }
  body += `<text x="0" y="0" font-size="12" fill="${CHART_COLORS.text}" font-weight="600" transform="translate(-46 ${plotH / 2}) rotate(-90)" text-anchor="middle">|metric|</text>`;

  // Threshold line.
  if (opts.threshold !== undefined && opts.threshold > 0) {
    const tY = scaleY(Math.log10(opts.threshold));
    body += `<line x1="0" y1="${tY}" x2="${plotW}" y2="${tY}" stroke="${CHART_COLORS.amber}" stroke-dasharray="6 4" stroke-width="1.5"/>`;
    body += `<text x="${plotW + 4}" y="${tY + 4}" font-size="10" fill="${CHART_COLORS.amber}" font-family="ui-monospace, monospace">τ = ${fmtNum(opts.threshold, 2)}</text>`;
  }

  // X axis.
  const xTicks = niceTicks(0, Math.max(xMax - 1, 1), 6).filter((t) => Number.isInteger(t));
  body += `<line x1="0" y1="${plotH}" x2="${plotW}" y2="${plotH}" stroke="${CHART_COLORS.textFaded}"/>`;
  for (const t of xTicks) {
    const x = scaleX(t);
    body += `<line x1="${x}" y1="${plotH}" x2="${x}" y2="${plotH + 5}" stroke="${CHART_COLORS.textFaded}"/>`;
    body += `<text x="${x}" y="${plotH + 20}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${t}</text>`;
  }
  body += `<text x="${plotW / 2}" y="${plotH + 38}" font-size="12" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="600">${escapeXML(xLabel)}</text>`;

  // Series.
  for (let si = 0; si < seriesList.length; si++) {
    const s = seriesList[si]!;
    const color = SERIES_PALETTE[si % SERIES_PALETTE.length]!;
    const data: Array<[number, number]> = [];
    for (let i = 0; i < s.values.length; i++) {
      const clamped = Math.max(Math.abs(s.values[i]!), 1e-16);
      data.push([i, Math.log10(clamped)]);
    }
    body += polyline(data, scaleX, scaleY, color, 2);
    // Dot markers for clarity at low iteration counts.
    if (s.values.length <= 30) {
      for (const [x, y] of data) {
        body += `<circle cx="${scaleX(x).toFixed(1)}" cy="${scaleY(y).toFixed(1)}" r="2.5" fill="${color}"/>`;
      }
    }
  }

  // Legend.
  const legendX = plotW + 16;
  for (let si = 0; si < seriesList.length; si++) {
    const color = SERIES_PALETTE[si % SERIES_PALETTE.length]!;
    const y = si * 22;
    body += `<rect x="${legendX}" y="${y}" width="14" height="3" fill="${color}"/>`;
    body += `<text x="${legendX + 20}" y="${y + 5}" font-size="11" fill="${CHART_COLORS.text}" font-family="ui-monospace, monospace">${escapeXML(seriesList[si]!.name)}</text>`;
  }

  return open + body + chartFooter();
}

function supscript(p: number): string {
  const map: Record<string, string> = {
    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
  };
  return String(p).split("").map((c) => map[c] ?? c).join("");
}
