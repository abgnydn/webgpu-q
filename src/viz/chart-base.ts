// ─────────────────────────────────────────────────────────────
// chart-base.ts — minimal SVG chart primitives shared by all
// chemistry visualization modules (alpha-omega, NOON, UV-vis,
// energy decomposition, C6 matrix, rotational constants, charges
// overlay, D2 overlay, orbital contours).
//
// Style: matches the webgpu-q dark theme.
//   bg:    #0b1224 → #020617 radial
//   text:  #e2e8f0 high · #94a3b8 mid · #64748b faded
//   accents: cyan #22d3ee · amber #ffb56e · lavender #c084fc ·
//            lime #34d399 · pink #f472b6 · yellow #fbbf24
//
// All chart functions return a complete SVG string (with xmlns,
// viewBox, the dark background). Result can be set as innerHTML on
// any container — no scripting, no external CSS required.
// ─────────────────────────────────────────────────────────────

export const CHART_COLORS = {
  bg0: "#0b1224",
  bg1: "#020617",
  text: "#e2e8f0",
  textMid: "#94a3b8",
  textFaded: "#64748b",
  cyan: "#22d3ee",
  amber: "#ffb56e",
  lavender: "#c084fc",
  lime: "#34d399",
  pink: "#f472b6",
  yellow: "#fbbf24",
  red: "#f87171",
} as const;

/** Series cycle for multi-line/bar charts. */
export const SERIES_PALETTE: readonly string[] = [
  CHART_COLORS.cyan,
  CHART_COLORS.amber,
  CHART_COLORS.lavender,
  CHART_COLORS.lime,
  CHART_COLORS.pink,
  CHART_COLORS.yellow,
];

export interface ChartFrame {
  readonly width: number;
  readonly height: number;
  readonly margin: { top: number; right: number; bottom: number; left: number };
}

/** Standard frame: 720×360 with comfortable margins. */
export const DEFAULT_FRAME: ChartFrame = {
  width: 720, height: 360,
  margin: { top: 30, right: 30, bottom: 50, left: 60 },
};

/** Build the SVG root + dark background + grid. Returns the open SVG
 *  string up through the plot-region group; caller appends marks + closes. */
export function chartHeader(frame: ChartFrame, ariaLabel: string): {
  open: string;
  plotX: number;
  plotY: number;
  plotW: number;
  plotH: number;
} {
  const { width, height, margin } = frame;
  const plotX = margin.left;
  const plotY = margin.top;
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const open = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" style="height:auto" role="img" aria-label="${escapeXML(ariaLabel)}">
    <defs>
      <linearGradient id="cbg" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="${CHART_COLORS.bg0}"/>
        <stop offset="100%" stop-color="${CHART_COLORS.bg1}"/>
      </linearGradient>
    </defs>
    <rect width="${width}" height="${height}" fill="url(#cbg)" rx="6"/>
    <g transform="translate(${plotX} ${plotY})">`;
  return { open, plotX, plotY, plotW, plotH };
}

export function chartFooter(): string {
  return `</g></svg>`;
}

/** Linear scale: maps [domainMin, domainMax] → [rangeMin, rangeMax]. */
export function linearScale(
  domainMin: number, domainMax: number,
  rangeMin: number, rangeMax: number,
): (x: number) => number {
  const dSpan = domainMax - domainMin || 1;
  const rSpan = rangeMax - rangeMin;
  return (x) => rangeMin + ((x - domainMin) / dSpan) * rSpan;
}

/** Pretty axis ticks for a given numeric range; ~5-8 ticks. */
export function niceTicks(min: number, max: number, count: number = 6): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return [min];
  }
  const span = max - min;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = mag * (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10);
  const tickMin = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let t = tickMin; t <= max + step * 0.001; t += step) {
    ticks.push(Number(t.toPrecision(10)));
  }
  return ticks;
}

/** X-axis with ticks + label. plotH = chart height (line position). */
export function xAxis(
  scaleX: (x: number) => number, plotH: number, plotW: number,
  ticks: number[], label: string, fmt: (v: number) => string = (v) => String(v),
): string {
  let s = `<line x1="0" y1="${plotH}" x2="${plotW}" y2="${plotH}" stroke="${CHART_COLORS.textFaded}" stroke-width="1"/>`;
  for (const t of ticks) {
    const x = scaleX(t);
    s += `<line x1="${x}" y1="${plotH}" x2="${x}" y2="${plotH + 5}" stroke="${CHART_COLORS.textFaded}"/>`;
    s += `<text x="${x}" y="${plotH + 20}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${escapeXML(fmt(t))}</text>`;
  }
  s += `<text x="${plotW / 2}" y="${plotH + 42}" font-size="12" fill="${CHART_COLORS.text}" text-anchor="middle" font-weight="600">${escapeXML(label)}</text>`;
  return s;
}

/** Y-axis with ticks + label. */
export function yAxis(
  scaleY: (y: number) => number, plotH: number,
  ticks: number[], label: string, fmt: (v: number) => string = (v) => String(v),
): string {
  let s = `<line x1="0" y1="0" x2="0" y2="${plotH}" stroke="${CHART_COLORS.textFaded}" stroke-width="1"/>`;
  for (const t of ticks) {
    const y = scaleY(t);
    s += `<line x1="-5" y1="${y}" x2="0" y2="${y}" stroke="${CHART_COLORS.textFaded}"/>`;
    s += `<text x="-10" y="${y + 4}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="end" font-family="ui-monospace, monospace">${escapeXML(fmt(t))}</text>`;
  }
  // Y-axis label rotated.
  s += `<text x="0" y="0" font-size="12" fill="${CHART_COLORS.text}" font-weight="600" transform="translate(-46 ${plotH / 2}) rotate(-90)" text-anchor="middle">${escapeXML(label)}</text>`;
  return s;
}

/** Grid lines at every tick (horizontal). */
export function gridY(
  scaleY: (y: number) => number, plotW: number, ticks: number[],
): string {
  let s = "";
  for (const t of ticks) {
    s += `<line x1="0" y1="${scaleY(t)}" x2="${plotW}" y2="${scaleY(t)}" stroke="${CHART_COLORS.textFaded}" stroke-opacity="0.15"/>`;
  }
  return s;
}

/** A polyline of (x, y) data points. */
export function polyline(
  data: ReadonlyArray<readonly [number, number]>,
  scaleX: (x: number) => number, scaleY: (y: number) => number,
  color: string, strokeWidth: number = 2,
): string {
  const pts = data.map(([x, y]) => `${scaleX(x).toFixed(2)},${scaleY(y).toFixed(2)}`).join(" ");
  return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-linecap="round"/>`;
}

export function escapeXML(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}

/** Format a number for display: 3 sig figs unless integer. */
export function fmtNum(x: number, sig: number = 3): string {
  if (!Number.isFinite(x)) return "—";
  if (Math.abs(x) < 1e-12) return "0";
  if (Number.isInteger(x) && Math.abs(x) < 10000) return String(x);
  return x.toPrecision(sig);
}
