// ─────────────────────────────────────────────────────────────
// uv-vis-chart.ts — Gaussian-broadened absorption spectrum from
// TDDFT / EOM-CCSD output. Plots σ_abs(ω) vs ω with peak labels
// at local maxima.
// ─────────────────────────────────────────────────────────────

import {
  chartHeader, chartFooter, linearScale, niceTicks,
  xAxis, yAxis, gridY, polyline, escapeXML,
  CHART_COLORS, DEFAULT_FRAME,
  type ChartFrame,
} from "./chart-base.js";
import type { SpectrumPoint } from "../chemistry/spectrum.js";

export interface UVVisChartOpts {
  readonly frame?: ChartFrame;
  readonly title?: string;
  /** Show peak labels (default true). */
  readonly showPeaks?: boolean;
  /** Energy unit label. Default "ω (eV)". */
  readonly xLabel?: string;
  /** Fill area under curve. */
  readonly fillArea?: boolean;
}

export function uvVisChart(
  spectrum: readonly SpectrumPoint[],
  peaks: readonly SpectrumPoint[],
  opts: UVVisChartOpts = {},
): string {
  const frame = opts.frame ?? DEFAULT_FRAME;
  const title = opts.title ?? "UV-vis absorption spectrum";
  const xLabel = opts.xLabel ?? "ω (eV)";
  const showPeaks = opts.showPeaks ?? true;
  const fillArea = opts.fillArea ?? true;

  if (spectrum.length === 0) {
    const { open } = chartHeader(frame, `${title} (empty)`);
    return open + `<text x="${frame.width / 2 - 60}" y="${frame.height / 2 - 30}" fill="${CHART_COLORS.textMid}">no spectrum data</text>` + chartFooter();
  }

  const xMin = spectrum[0]!.energy;
  const xMax = spectrum[spectrum.length - 1]!.energy;
  let yMax = 0;
  for (const p of spectrum) if (p.absorption > yMax) yMax = p.absorption;
  yMax = yMax * 1.15 || 1;

  const { open, plotW, plotH } = chartHeader(frame, title);
  const scaleX = linearScale(xMin, xMax, 0, plotW);
  const scaleY = linearScale(0, yMax, plotH, 0);
  const xTicks = niceTicks(xMin, xMax, 6);
  const yTicks = niceTicks(0, yMax, 5);

  let body = "";
  body += gridY(scaleY, plotW, yTicks);
  body += xAxis(scaleX, plotH, plotW, xTicks, xLabel, (v) => v.toFixed(1));
  body += yAxis(scaleY, plotH, yTicks, "absorption (a.u.)", (v) => v.toFixed(1));
  body += `<text x="${plotW / 2}" y="-12" font-size="13" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="700">${escapeXML(title)}</text>`;

  // Filled area under curve.
  if (fillArea) {
    const pts = spectrum.map((p) => `${scaleX(p.energy).toFixed(2)},${scaleY(p.absorption).toFixed(2)}`).join(" ");
    body += `<polygon points="0,${plotH} ${pts} ${plotW},${plotH}" fill="${CHART_COLORS.cyan}" fill-opacity="0.15"/>`;
  }

  // Line.
  body += polyline(
    spectrum.map((p) => [p.energy, p.absorption] as const),
    scaleX, scaleY, CHART_COLORS.cyan, 2,
  );

  // Peaks with labels.
  if (showPeaks) {
    for (const peak of peaks) {
      const x = scaleX(peak.energy);
      const y = scaleY(peak.absorption);
      body += `<circle cx="${x}" cy="${y}" r="4" fill="${CHART_COLORS.amber}" stroke="${CHART_COLORS.bg1}" stroke-width="1.5"/>`;
      body += `<text x="${x}" y="${y - 10}" font-size="10" fill="${CHART_COLORS.amber}" text-anchor="middle" font-weight="600" font-family="ui-monospace, monospace">${peak.energy.toFixed(2)}</text>`;
    }
  }

  return open + body + chartFooter();
}
