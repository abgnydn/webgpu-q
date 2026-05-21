// ─────────────────────────────────────────────────────────────
// alpha-omega-chart.ts — α(ω) frequency-dependent polarizability
// dispersion curve. Plots the isotropic α (in a.u. or Å³) vs photon
// frequency ω (Ha or eV) from 0 up to just below the first pole.
//
// The signature divergence as ω → ω_1 (first electronic excitation)
// is *the* picture of where excited states live. Multiple series
// supported for method comparison (RHF · UHF · RKS · UKS-DFT).
// ─────────────────────────────────────────────────────────────

import {
  chartHeader, chartFooter, linearScale, niceTicks,
  xAxis, yAxis, gridY, polyline, escapeXML,
  CHART_COLORS, SERIES_PALETTE, DEFAULT_FRAME,
  type ChartFrame,
} from "./chart-base.js";

export interface AlphaOmegaSeries {
  readonly name: string;
  readonly data: ReadonlyArray<readonly [number, number]>;   // [(ω, α_iso)]
}

export interface AlphaOmegaChartOpts {
  readonly frame?: ChartFrame;
  readonly title?: string;
  /** Energy unit label for x-axis. Default "ω (Ha)". */
  readonly xLabel?: string;
  /** α unit label for y-axis. Default "α_iso (a.u.)". */
  readonly yLabel?: string;
}

/**
 * Render α(ω) dispersion curve(s) as an SVG line chart.
 * Each series is a list of (ω, α_iso) data points; multiple series
 * are drawn in different colors with a legend.
 */
export function alphaOmegaChart(
  series: readonly AlphaOmegaSeries[],
  opts: AlphaOmegaChartOpts = {},
): string {
  const frame = opts.frame ?? DEFAULT_FRAME;
  const xLabel = opts.xLabel ?? "ω (Ha)";
  const yLabel = opts.yLabel ?? "α_iso (a.u.)";
  const title = opts.title ?? "Dynamic polarizability α(ω)";

  // Auto-determine plot domain.
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const s of series) {
    for (const [x, y] of s.data) {
      if (x < xMin) xMin = x; if (x > xMax) xMax = x;
      if (y < yMin) yMin = y; if (y > yMax) yMax = y;
    }
  }
  if (!Number.isFinite(xMin)) {
    // No data — render placeholder.
    const { open } = chartHeader(frame, `${title} (no data)`);
    return `${open}<text x="${frame.width / 2 - 60}" y="${frame.height / 2 - 30}" fill="${CHART_COLORS.textMid}" text-anchor="middle">no data</text>${chartFooter()}`;
  }
  // Pad y range.
  const yPad = (yMax - yMin) * 0.05 || 1;
  yMin = Math.min(0, yMin - yPad);
  yMax = yMax + yPad;

  const { open, plotW, plotH } = chartHeader(frame, title);
  const scaleX = linearScale(xMin, xMax, 0, plotW);
  const scaleY = linearScale(yMin, yMax, plotH, 0);
  const xTicks = niceTicks(xMin, xMax, 6);
  const yTicks = niceTicks(yMin, yMax, 5);

  let body = "";
  body += gridY(scaleY, plotW, yTicks);
  body += xAxis(scaleX, plotH, plotW, xTicks, xLabel, (v) => v.toFixed(2));
  body += yAxis(scaleY, plotH, yTicks, yLabel, (v) => v.toFixed(1));

  // Title.
  body += `<text x="${plotW / 2}" y="-12" font-size="13" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="700">${escapeXML(title)}</text>`;

  // Series.
  for (let i = 0; i < series.length; i++) {
    const s = series[i]!;
    const color = SERIES_PALETTE[i % SERIES_PALETTE.length]!;
    body += polyline(s.data, scaleX, scaleY, color, 2);
  }

  // Legend (top-right).
  if (series.length > 0) {
    const legendX = plotW - 130;
    const legendY = 5;
    body += `<g transform="translate(${legendX} ${legendY})">`;
    for (let i = 0; i < series.length; i++) {
      const color = SERIES_PALETTE[i % SERIES_PALETTE.length]!;
      const y = i * 16;
      body += `<line x1="0" y1="${y}" x2="20" y2="${y}" stroke="${color}" stroke-width="2"/>`;
      body += `<text x="26" y="${y + 4}" font-size="11" fill="${CHART_COLORS.text}">${escapeXML(series[i]!.name)}</text>`;
    }
    body += `</g>`;
  }

  return open + body + chartFooter();
}
