// ─────────────────────────────────────────────────────────────
// energy-decomp-chart.ts — stacked-bar visualization of the HF/UHF
// energy decomposition: one-electron + Coulomb + exchange + V_nn
// plus optional correlation + dispersion stacked on top.
//
// Layout: a single horizontal bar with each component segment
// labeled by value + (optional) % of total. Useful for "why does
// this molecule have this energy?" teaching plus method-comparison
// (HF vs DFT vs DFT+D2) side-by-side.
// ─────────────────────────────────────────────────────────────

import {
  chartHeader, chartFooter, escapeXML, fmtNum,
  CHART_COLORS, DEFAULT_FRAME, type ChartFrame,
} from "./chart-base.js";

export interface EnergyComponent {
  readonly label: string;
  readonly value: number;     // Hartree
  readonly color: string;
}

export interface EnergyDecompChartOpts {
  readonly frame?: ChartFrame;
  readonly title?: string;
  /** Total energy for the percentage-of-total annotation. If omitted,
   *  uses Σ components.  */
  readonly total?: number;
  /** Show signed values + units in segment labels. */
  readonly showValues?: boolean;
}

/**
 * Stacked bar of energy components. Positive and negative segments
 * are rendered separately (positive going right from a zero line,
 * negative going left).
 */
export function energyDecompChart(
  components: readonly EnergyComponent[],
  opts: EnergyDecompChartOpts = {},
): string {
  const frame = opts.frame ?? { ...DEFAULT_FRAME, height: 240 };
  const title = opts.title ?? "Energy decomposition";
  const showValues = opts.showValues ?? true;

  if (components.length === 0) {
    const { open } = chartHeader(frame, `${title} (empty)`);
    return open + `<text x="100" y="100" fill="${CHART_COLORS.textMid}">no components</text>` + chartFooter();
  }

  // Split positives/negatives.
  let totalPos = 0, totalNeg = 0;
  for (const c of components) {
    if (c.value >= 0) totalPos += c.value;
    else totalNeg += Math.abs(c.value);
  }
  const totalSpan = totalPos + totalNeg;
  if (totalSpan === 0) {
    const { open } = chartHeader(frame, `${title} (zero)`);
    return open + `<text x="100" y="100" fill="${CHART_COLORS.textMid}">all zero</text>` + chartFooter();
  }

  const { open, plotW, plotH } = chartHeader(frame, title);
  // Zero line position: proportional to negative fraction.
  const zeroX = (totalNeg / totalSpan) * plotW;
  const barHeight = 50;
  const barY = plotH / 2 - barHeight / 2;
  const scaleVal = (v: number) => Math.abs(v) / totalSpan * plotW;

  let body = "";
  body += `<text x="${plotW / 2}" y="-12" font-size="13" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="700">${escapeXML(title)}</text>`;

  // Negatives stacked left from zero line.
  let negCursor = zeroX;
  for (const c of components) {
    if (c.value >= 0) continue;
    const w = scaleVal(c.value);
    negCursor -= w;
    body += `<rect x="${negCursor}" y="${barY}" width="${w}" height="${barHeight}" fill="${c.color}" fill-opacity="0.85" stroke="${CHART_COLORS.bg1}" stroke-width="1"/>`;
    if (w > 50 && showValues) {
      body += `<text x="${negCursor + w / 2}" y="${barY + barHeight / 2 + 4}" font-size="11" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle" font-family="ui-monospace, monospace">${fmtNum(c.value)}</text>`;
    }
    body += `<text x="${negCursor + w / 2}" y="${barY - 6}" font-size="10" fill="${c.color}" text-anchor="middle" font-weight="600">${escapeXML(c.label)}</text>`;
  }

  // Positives stacked right from zero line.
  let posCursor = zeroX;
  for (const c of components) {
    if (c.value < 0) continue;
    const w = scaleVal(c.value);
    body += `<rect x="${posCursor}" y="${barY}" width="${w}" height="${barHeight}" fill="${c.color}" fill-opacity="0.85" stroke="${CHART_COLORS.bg1}" stroke-width="1"/>`;
    if (w > 50 && showValues) {
      body += `<text x="${posCursor + w / 2}" y="${barY + barHeight / 2 + 4}" font-size="11" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle" font-family="ui-monospace, monospace">${fmtNum(c.value)}</text>`;
    }
    body += `<text x="${posCursor + w / 2}" y="${barY - 6}" font-size="10" fill="${c.color}" text-anchor="middle" font-weight="600">${escapeXML(c.label)}</text>`;
    posCursor += w;
  }

  // Zero line.
  body += `<line x1="${zeroX}" y1="${barY - 25}" x2="${zeroX}" y2="${barY + barHeight + 25}" stroke="${CHART_COLORS.text}" stroke-dasharray="3,3" stroke-opacity="0.5"/>`;
  body += `<text x="${zeroX}" y="${barY + barHeight + 40}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle">0 Ha</text>`;

  // Total annotation.
  const total = opts.total ?? (totalPos - totalNeg);
  body += `<text x="${plotW / 2}" y="${plotH - 6}" font-size="12" fill="${CHART_COLORS.text}" text-anchor="middle" font-family="ui-monospace, monospace" font-weight="700">total: ${fmtNum(total, 5)} Ha</text>`;

  return open + body + chartFooter();
}

/** Convenience: build EnergyComponent array from a decomposition + opts. */
export function decompositionToComponents(
  decomp: {
    readonly oneElectron: number;
    readonly coulomb: number;
    readonly exchange: number;
    readonly nuclearRepulsion: number;
  },
  extras?: { readonly correlation?: number; readonly dispersion?: number },
): EnergyComponent[] {
  const out: EnergyComponent[] = [
    { label: "1-e (T + V_ne)", value: decomp.oneElectron, color: CHART_COLORS.cyan },
    { label: "Coulomb J", value: decomp.coulomb, color: CHART_COLORS.amber },
    { label: "Exchange K", value: decomp.exchange, color: CHART_COLORS.lavender },
    { label: "V_nn", value: decomp.nuclearRepulsion, color: CHART_COLORS.lime },
  ];
  if (extras?.correlation !== undefined) {
    out.push({ label: "E_corr", value: extras.correlation, color: CHART_COLORS.pink });
  }
  if (extras?.dispersion !== undefined) {
    out.push({ label: "D2", value: extras.dispersion, color: CHART_COLORS.yellow });
  }
  return out;
}
