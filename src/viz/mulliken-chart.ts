// ─────────────────────────────────────────────────────────────
// mulliken-chart.ts — bar chart of Mulliken atomic charges
// (sums to zero by construction). Color encodes sign:
// red for positive (electron-deficient), cyan for negative.
//
// For an atom-overlay visualization, see molecular-graph-2d.ts
// with `charges` option.
// ─────────────────────────────────────────────────────────────

import {
  CHART_COLORS, chartHeader, chartFooter, escapeXML, fmtNum,
  linearScale, niceTicks, type ChartFrame,
} from "./chart-base.js";

export interface MullikenChartOpts {
  readonly title?: string;
  readonly atomLabels?: readonly string[];
  readonly frame?: ChartFrame;
}

export function mullikenChart(
  charges: ReadonlyArray<number> | Float64Array,
  opts: MullikenChartOpts = {},
): string {
  const frame: ChartFrame = opts.frame ?? {
    width: 640, height: 320,
    margin: { top: 50, right: 30, bottom: 60, left: 60 },
  };
  const title = opts.title ?? "Mulliken atomic charges";
  const labels = opts.atomLabels;
  const n = charges.length;

  const { open, plotW, plotH } = chartHeader(frame, title);
  if (n === 0) {
    return open
      + `<text x="${plotW / 2}" y="${plotH / 2}" font-size="14" fill="${CHART_COLORS.textFaded}" text-anchor="middle">no atoms</text>`
      + chartFooter();
  }

  // Find sum (should be ~0 for neutral); show as caption.
  let total = 0, qMin = Infinity, qMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const q = charges[i]!;
    total += q;
    if (q < qMin) qMin = q; if (q > qMax) qMax = q;
  }
  // Symmetric range around 0.
  const absMax = Math.max(Math.abs(qMin), Math.abs(qMax), 0.05);
  const yMin = -absMax * 1.15, yMax = absMax * 1.15;

  let body = `<text x="${plotW / 2}" y="-28" font-size="14" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle">${escapeXML(title)}</text>`;
  body += `<text x="${plotW / 2}" y="-10" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">Σq = ${fmtNum(total, 3)} e   ·   range [${fmtNum(qMin, 3)}, ${fmtNum(qMax, 3)}]</text>`;

  const scaleY = linearScale(yMin, yMax, plotH, 0);
  const ticks = niceTicks(yMin, yMax, 5);

  // Grid + axis.
  for (const t of ticks) {
    body += `<line x1="0" y1="${scaleY(t)}" x2="${plotW}" y2="${scaleY(t)}" stroke="${CHART_COLORS.textFaded}" stroke-opacity="${t === 0 ? 0.5 : 0.15}"/>`;
    body += `<text x="-8" y="${scaleY(t) + 4}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="end" font-family="ui-monospace, monospace">${fmtNum(t, 3)}</text>`;
  }
  body += `<text x="0" y="0" font-size="12" fill="${CHART_COLORS.text}" font-weight="600" transform="translate(-44 ${plotH / 2}) rotate(-90)" text-anchor="middle">q (e)</text>`;

  // Bars.
  const barGap = 8;
  const slot = plotW / n;
  const barW = Math.min(slot - barGap, 40);
  const zeroY = scaleY(0);
  for (let i = 0; i < n; i++) {
    const q = charges[i]!;
    const x = i * slot + (slot - barW) / 2;
    const y = scaleY(q);
    const h = Math.abs(y - zeroY);
    const top = q >= 0 ? y : zeroY;
    const color = q >= 0 ? CHART_COLORS.red : CHART_COLORS.cyan;
    body += `<rect x="${x.toFixed(1)}" y="${top.toFixed(1)}" width="${barW.toFixed(1)}" height="${Math.max(h, 1).toFixed(1)}" fill="${color}" fill-opacity="0.85" rx="3"/>`;
    const lbl = labels?.[i] ?? `${i}`;
    body += `<text x="${(x + barW / 2).toFixed(1)}" y="${plotH + 18}" font-size="11" fill="${CHART_COLORS.text}" text-anchor="middle" font-family="ui-monospace, monospace">${escapeXML(lbl)}</text>`;
    // Numeric label above/below the bar.
    const numY = q >= 0 ? (y - 4) : (y + 12);
    body += `<text x="${(x + barW / 2).toFixed(1)}" y="${numY.toFixed(1)}" font-size="10" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${q >= 0 ? "+" : ""}${fmtNum(q, 3)}</text>`;
  }
  // Atom label legend below bars.
  body += `<text x="${plotW / 2}" y="${plotH + 44}" font-size="11" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="600">atom</text>`;

  return open + body + chartFooter();
}
