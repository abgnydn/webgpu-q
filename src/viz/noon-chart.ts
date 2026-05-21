// ─────────────────────────────────────────────────────────────
// noon-chart.ts — Natural Orbital Occupation Number bar chart.
// Each NOON gets a vertical bar from 0 to its occupation (0-2 scale).
// Closed-shell single-determinant systems show a clean "cliff" from
// 2 → 0; multireference systems show fractional bars in the active
// region — instant visual diagnostic of multi-reference character.
//
// Optional multireference verdict badge (from `multireferenceDiagnostic`)
// rendered top-right with severity color:
//   green  → single-reference
//   yellow → borderline
//   red    → multi-reference
// ─────────────────────────────────────────────────────────────

import {
  chartHeader, chartFooter, linearScale, niceTicks,
  yAxis, gridY, escapeXML, CHART_COLORS, DEFAULT_FRAME,
  type ChartFrame,
} from "./chart-base.js";

export interface NOONChartOpts {
  readonly frame?: ChartFrame;
  readonly title?: string;
  readonly nOccupied?: number;       // show vertical divider after this index
  readonly severity?: "single-reference" | "borderline" | "multi-reference";
  readonly flags?: readonly string[];
}

export function noonChart(
  occupations: ReadonlyArray<number> | Float64Array,
  opts: NOONChartOpts = {},
): string {
  const frame = opts.frame ?? DEFAULT_FRAME;
  const title = opts.title ?? "Natural orbital occupations (NOON)";
  const n = occupations.length;

  const { open, plotW, plotH } = chartHeader(frame, title);
  // Y axis: 0 to 2.0 (occupation per natural orbital).
  const scaleY = linearScale(0, 2.1, plotH, 0);
  // X axis: discrete bars.
  const barGap = 2;
  const barW = Math.max(2, (plotW - barGap * (n - 1)) / n);

  let body = "";
  const yTicks = niceTicks(0, 2.1, 5);
  body += gridY(scaleY, plotW, yTicks);
  body += yAxis(scaleY, plotH, yTicks, "n_p (electrons)", (v) => v.toFixed(1));

  // X axis (orbital index).
  body += `<line x1="0" y1="${plotH}" x2="${plotW}" y2="${plotH}" stroke="${CHART_COLORS.textFaded}"/>`;
  body += `<text x="${plotW / 2}" y="${plotH + 30}" font-size="12" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="600">natural orbital index (sorted descending)</text>`;

  // Bars.
  for (let i = 0; i < n; i++) {
    const x = i * (barW + barGap);
    const v = occupations[i] ?? 0;
    const y = scaleY(v);
    const h = plotH - y;
    // Color by classification: occupied (>1.95) cyan, active (0.05–1.95) amber, virtual (<0.05) faded.
    let color: string = CHART_COLORS.textFaded;
    if (v > 1.95) color = CHART_COLORS.cyan;
    else if (v > 0.05) color = CHART_COLORS.amber;
    body += `<rect x="${x}" y="${y}" width="${barW}" height="${h}" fill="${color}" fill-opacity="0.85"/>`;
  }

  // Optional vertical divider after the doubly-occupied region.
  if (opts.nOccupied !== undefined && opts.nOccupied > 0 && opts.nOccupied < n) {
    const x = opts.nOccupied * (barW + barGap) - barGap / 2;
    body += `<line x1="${x}" y1="0" x2="${x}" y2="${plotH}" stroke="${CHART_COLORS.text}" stroke-dasharray="3,3" stroke-opacity="0.4"/>`;
    body += `<text x="${x + 4}" y="14" font-size="10" fill="${CHART_COLORS.textMid}">↑ HOMO</text>`;
  }

  // Title.
  body += `<text x="${plotW / 2}" y="-12" font-size="13" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="700">${escapeXML(title)}</text>`;

  // Severity badge.
  if (opts.severity) {
    let color: string = CHART_COLORS.lime, label = "single-reference";
    if (opts.severity === "borderline") { color = CHART_COLORS.yellow; label = "borderline"; }
    if (opts.severity === "multi-reference") { color = CHART_COLORS.red; label = "multi-reference"; }
    const badgeX = plotW - 150;
    body += `<g transform="translate(${badgeX} -2)">`;
    body += `<rect x="0" y="-14" width="146" height="20" rx="10" fill="${CHART_COLORS.bg1}" stroke="${color}" stroke-opacity="0.7"/>`;
    body += `<circle cx="12" cy="-4" r="4" fill="${color}"/>`;
    body += `<text x="22" y="0" font-size="11" font-weight="700" fill="${color}">${escapeXML(label)}</text>`;
    body += `</g>`;
  }

  return open + body + chartFooter();
}
