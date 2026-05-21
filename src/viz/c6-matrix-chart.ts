// ─────────────────────────────────────────────────────────────
// c6-matrix-chart.ts — N×N heatmap of pairwise C6 van-der-Waals
// coefficients between molecule labels. Useful for showing how
// C6 scales with molecular size and how DFT vs HF predictions
// compare for a set of molecules.
// ─────────────────────────────────────────────────────────────

import {
  chartHeader, chartFooter, escapeXML, fmtNum,
  CHART_COLORS, type ChartFrame,
} from "./chart-base.js";

export interface C6MatrixChartOpts {
  readonly frame?: ChartFrame;
  readonly title?: string;
  /** Color scale: linear (default) or log10. */
  readonly scale?: "linear" | "log";
}

/**
 * Render an N×N C6 matrix as a colored heatmap with labels.
 * `matrix[i][j]` is C6 between molecules labels[i] and labels[j]
 * in atomic units. Diagonal is self-dispersion.
 */
export function c6MatrixChart(
  labels: readonly string[],
  matrix: readonly (readonly number[])[],
  opts: C6MatrixChartOpts = {},
): string {
  const N = labels.length;
  const scale = opts.scale ?? "linear";
  const title = opts.title ?? "C₆ pairwise dispersion (a.u.)";

  const cellSize = 56;
  const labelPad = 90;
  const padding = 20;
  const width = labelPad + N * cellSize + padding;
  const height = labelPad + N * cellSize + padding + 40;
  const frame: ChartFrame = opts.frame ?? {
    width, height, margin: { top: 30, right: padding, bottom: 20, left: padding },
  };

  // Color scale: find min/max.
  let vMin = Infinity, vMax = -Infinity;
  for (const row of matrix) {
    for (const v of row) {
      if (Number.isFinite(v) && v > 0) {
        const sv = scale === "log" ? Math.log10(v) : v;
        if (sv < vMin) vMin = sv;
        if (sv > vMax) vMax = sv;
      }
    }
  }
  if (!Number.isFinite(vMin)) { vMin = 0; vMax = 1; }
  const colorFor = (v: number): string => {
    if (!Number.isFinite(v) || v <= 0) return CHART_COLORS.bg1;
    const sv = scale === "log" ? Math.log10(v) : v;
    const t = Math.max(0, Math.min(1, (sv - vMin) / (vMax - vMin || 1)));
    // Cyan → amber gradient.
    return interpColor("#06121f", CHART_COLORS.cyan, CHART_COLORS.amber, t);
  };

  const { open, plotW } = chartHeader(frame, title);
  let body = "";
  body += `<text x="${plotW / 2}" y="-12" font-size="13" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="700">${escapeXML(title)}</text>`;

  // Column labels (top).
  for (let j = 0; j < N; j++) {
    const x = labelPad + j * cellSize + cellSize / 2;
    body += `<text x="${x}" y="-2" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-weight="600">${escapeXML(labels[j] ?? "")}</text>`;
  }
  // Row labels (left).
  for (let i = 0; i < N; i++) {
    const y = labelPad + i * cellSize + cellSize / 2 + 4;
    body += `<text x="${labelPad - 8}" y="${y - labelPad}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="end" font-weight="600">${escapeXML(labels[i] ?? "")}</text>`;
  }
  // Adjust: shift the matrix body down by labelPad - top label area.
  body += `<g transform="translate(${labelPad} ${5})">`;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v = matrix[i]?.[j] ?? Number.NaN;
      const x = j * cellSize;
      const y = i * cellSize;
      body += `<rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" fill="${colorFor(v)}" stroke="${CHART_COLORS.bg1}" stroke-width="1"/>`;
      const text = Number.isFinite(v) ? fmtNum(v, 3) : "—";
      // Use white text on dark cells, dark text on light cells.
      const sv = scale === "log" && Number.isFinite(v) && v > 0 ? Math.log10(v) : v;
      const t = (sv - vMin) / (vMax - vMin || 1);
      const textColor = t > 0.5 ? CHART_COLORS.bg1 : CHART_COLORS.text;
      body += `<text x="${x + cellSize / 2}" y="${y + cellSize / 2 + 4}" font-size="11" fill="${textColor}" text-anchor="middle" font-family="ui-monospace, monospace" font-weight="600">${escapeXML(text)}</text>`;
    }
  }
  body += `</g>`;

  // Scale label.
  body += `<text x="${labelPad + N * cellSize / 2}" y="${labelPad + N * cellSize + 20}" font-size="10" fill="${CHART_COLORS.textFaded}" text-anchor="middle" font-style="italic">color: ${scale} · range ${fmtNum(scale === "log" ? Math.pow(10, vMin) : vMin)} – ${fmtNum(scale === "log" ? Math.pow(10, vMax) : vMax)} a.u.</text>`;

  return open + body + chartFooter();
}

/** Interpolate through three colors at t ∈ [0, 1]. */
function interpColor(c0: string, c1: string, c2: string, t: number): string {
  const [r0, g0, b0] = hexToRgb(c0);
  const [r1, g1, b1] = hexToRgb(c1);
  const [r2, g2, b2] = hexToRgb(c2);
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const u = t * 2;
    r = r0 + (r1 - r0) * u; g = g0 + (g1 - g0) * u; b = b0 + (b1 - b0) * u;
  } else {
    const u = (t - 0.5) * 2;
    r = r1 + (r2 - r1) * u; g = g1 + (g2 - g1) * u; b = b1 + (b2 - b1) * u;
  }
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1]!, 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}
