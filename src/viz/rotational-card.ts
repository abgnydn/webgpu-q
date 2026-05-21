// ─────────────────────────────────────────────────────────────
// rotational-card.ts — A/B/C rotational-constants display card.
// Shows the three principal rotational constants in cm⁻¹ and GHz
// alongside a small principal-axes diagram. Used for microwave-
// spectroscopy comparison and as a structural fingerprint.
// ─────────────────────────────────────────────────────────────

import {
  chartHeader, chartFooter, escapeXML, fmtNum,
  CHART_COLORS, type ChartFrame,
} from "./chart-base.js";

export interface RotationalCardOpts {
  readonly frame?: ChartFrame;
  readonly title?: string;
}

/**
 * Render rotational constants (A/B/C in cm⁻¹ and GHz) as a 3-column
 * card. Each column has the axis letter, value in cm⁻¹, value in GHz,
 * and a color band.
 */
export function rotationalCard(
  constants: {
    readonly A_cm1: number; readonly B_cm1: number; readonly C_cm1: number;
    readonly A_GHz: number; readonly B_GHz: number; readonly C_GHz: number;
  },
  opts: RotationalCardOpts = {},
): string {
  const frame: ChartFrame = opts.frame ?? {
    width: 720, height: 200,
    margin: { top: 30, right: 30, bottom: 30, left: 30 },
  };
  const title = opts.title ?? "Rotational constants (principal axes)";

  const { open, plotW } = chartHeader(frame, title);
  let body = "";
  body += `<text x="${plotW / 2}" y="-12" font-size="13" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="700">${escapeXML(title)}</text>`;

  const colW = plotW / 3;
  const cols = [
    { label: "A", cm1: constants.A_cm1, ghz: constants.A_GHz, color: CHART_COLORS.cyan },
    { label: "B", cm1: constants.B_cm1, ghz: constants.B_GHz, color: CHART_COLORS.amber },
    { label: "C", cm1: constants.C_cm1, ghz: constants.C_GHz, color: CHART_COLORS.lavender },
  ];

  for (let i = 0; i < 3; i++) {
    const c = cols[i]!;
    const x = i * colW;
    // Color band on top.
    body += `<rect x="${x + 12}" y="6" width="${colW - 24}" height="4" rx="2" fill="${c.color}"/>`;
    // Axis label.
    body += `<text x="${x + colW / 2}" y="45" font-size="24" fill="${c.color}" text-anchor="middle" font-weight="800" font-family="ui-monospace, monospace">${c.label}</text>`;
    // cm⁻¹ value.
    if (Number.isFinite(c.cm1)) {
      body += `<text x="${x + colW / 2}" y="78" font-size="22" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="700" font-family="ui-monospace, monospace">${fmtNum(c.cm1, 4)}</text>`;
      body += `<text x="${x + colW / 2}" y="96" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle">cm⁻¹</text>`;
      // GHz value.
      body += `<text x="${x + colW / 2}" y="124" font-size="14" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${fmtNum(c.ghz, 4)}</text>`;
      body += `<text x="${x + colW / 2}" y="140" font-size="10" fill="${CHART_COLORS.textFaded}" text-anchor="middle">GHz</text>`;
    } else {
      body += `<text x="${x + colW / 2}" y="78" font-size="20" fill="${CHART_COLORS.textFaded}" text-anchor="middle">∞</text>`;
      body += `<text x="${x + colW / 2}" y="100" font-size="11" fill="${CHART_COLORS.textFaded}" text-anchor="middle">linear / atom</text>`;
    }
  }

  return open + body + chartFooter();
}
