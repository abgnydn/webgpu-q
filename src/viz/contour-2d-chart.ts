// ─────────────────────────────────────────────────────────────
// contour-2d-chart.ts — generic 2D scalar-field heatmap with
// optional contour lines and atom overlays. Drives orbital
// contour plots, density, and spin-density visualizations.
//
// Caller supplies a precomputed (nx × ny) grid of values plus the
// (xMin, xMax, yMin, yMax) bounds in whatever coordinate unit
// they want axes to display (e.g. Bohr or Å). The chart is
// agnostic — it just renders.
//
// Colormaps:
//   "sequential" — single-hue (cyan family), magnitude → bright.
//                  Good for densities ρ ≥ 0.
//   "divergent"  — red↔white↔blue, zero-centered.
//                  Good for orbital wavefunctions (signed) and
//                  spin density M = ρ_α − ρ_β.
// ─────────────────────────────────────────────────────────────

import {
  CHART_COLORS, chartHeader, chartFooter, escapeXML, fmtNum,
  type ChartFrame,
} from "./chart-base.js";

export interface Contour2DOpts {
  readonly title?: string;
  /** Field unit label, e.g. "ρ (e/Bohr³)" or "ψ_HOMO". */
  readonly fieldLabel?: string;
  /** Domain bounds in plot coordinates. */
  readonly xMin: number;
  readonly xMax: number;
  readonly yMin: number;
  readonly yMax: number;
  /** Grid shape (matches values.length === nx*ny, row-major). */
  readonly nx: number;
  readonly ny: number;
  /** Colormap. Default: "divergent". */
  readonly colormap?: "sequential" | "divergent";
  /** Explicit range override [vMin, vMax]. Otherwise auto. */
  readonly range?: readonly [number, number];
  /** Optional 2D-projected atom overlay. */
  readonly atoms?: ReadonlyArray<{
    readonly symbol: string;
    readonly x: number;
    readonly y: number;
  }>;
  /** Draw N contour lines (default 0 = no lines, just heatmap). */
  readonly contourCount?: number;
  readonly frame?: ChartFrame;
}

function divergentColor(t: number): string {
  // t in [-1, 1].  Red (positive) ↔ white ↔ blue (negative).
  const c = Math.max(-1, Math.min(1, t));
  if (c >= 0) {
    // white → red
    const r = 255;
    const g = Math.round(255 * (1 - c));
    const b = Math.round(255 * (1 - c));
    return `rgb(${r},${g},${b})`;
  } else {
    const r = Math.round(255 * (1 + c));
    const g = Math.round(255 * (1 + c));
    const b = 255;
    return `rgb(${r},${g},${b})`;
  }
}

function sequentialColor(t: number): string {
  // t in [0, 1]. dark → cyan → white.
  const u = Math.max(0, Math.min(1, t));
  // Two-stop gradient on cyan.
  const r = Math.round(2 + u * (180 - 2));
  const g = Math.round(8 + u * (240 - 8));
  const b = Math.round(36 + u * (250 - 36));
  return `rgb(${r},${g},${b})`;
}

export function contour2DChart(
  values: ReadonlyArray<number> | Float64Array | Float32Array,
  opts: Contour2DOpts,
): string {
  const frame: ChartFrame = opts.frame ?? {
    width: 460, height: 460,
    margin: { top: 50, right: 30, bottom: 50, left: 50 },
  };
  const title = opts.title ?? "Scalar field";
  const colormap = opts.colormap ?? "divergent";

  const { open, plotW, plotH } = chartHeader(frame, title);
  const { nx, ny, xMin, xMax, yMin, yMax } = opts;
  if (values.length !== nx * ny) {
    return open
      + `<text x="${plotW / 2}" y="${plotH / 2}" font-size="13" fill="${CHART_COLORS.red}" text-anchor="middle">grid mismatch: ${values.length} vs ${nx}·${ny}</text>`
      + chartFooter();
  }

  // Compute auto range if not provided.
  let vMin = Infinity, vMax = -Infinity;
  for (let k = 0; k < values.length; k++) {
    const v = values[k]!;
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
  }
  if (opts.range) { vMin = opts.range[0]; vMax = opts.range[1]; }
  let cMin = vMin, cMax = vMax;
  if (colormap === "divergent") {
    const absMax = Math.max(Math.abs(vMin), Math.abs(vMax), 1e-12);
    cMin = -absMax; cMax = absMax;
  }
  const cSpan = cMax - cMin || 1;

  const cellW = plotW / nx;
  const cellH = plotH / ny;

  let body = `<text x="${plotW / 2}" y="-28" font-size="14" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle">${escapeXML(title)}</text>`;
  if (opts.fieldLabel) {
    body += `<text x="${plotW / 2}" y="-10" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${escapeXML(opts.fieldLabel)}   ·   range [${fmtNum(vMin, 3)}, ${fmtNum(vMax, 3)}]</text>`;
  }

  // Render cells. y grows downward in SVG, so flip when drawing.
  // values index = ix + iy*nx with iy=0 at yMin.
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const v = values[ix + iy * nx]!;
      const t = colormap === "divergent"
        ? (v / Math.max(Math.abs(cMin), Math.abs(cMax), 1e-12))
        : (v - cMin) / cSpan;
      const color = colormap === "divergent" ? divergentColor(t) : sequentialColor(t);
      const x = ix * cellW;
      const y = plotH - (iy + 1) * cellH;
      body += `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${(cellW + 0.5).toFixed(2)}" height="${(cellH + 0.5).toFixed(2)}" fill="${color}"/>`;
    }
  }

  // Contour lines (very rough — just iso-contours via marching across rows).
  const contourCount = opts.contourCount ?? 0;
  if (contourCount > 0) {
    for (let k = 1; k <= contourCount; k++) {
      const level = cMin + (k / (contourCount + 1)) * cSpan;
      body += contourLine(values, nx, ny, cellW, cellH, plotH, level);
    }
  }

  // Atom overlay (project into plot box).
  if (opts.atoms) {
    const xSpan = xMax - xMin || 1;
    const ySpan = yMax - yMin || 1;
    for (const a of opts.atoms) {
      const sx = ((a.x - xMin) / xSpan) * plotW;
      const sy = plotH - ((a.y - yMin) / ySpan) * plotH;
      body += `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="6" fill="#0b1224" stroke="#e2e8f0" stroke-width="1.5"/>`;
      body += `<text x="${sx.toFixed(1)}" y="${(sy + 4).toFixed(1)}" font-size="9" font-weight="700" fill="#e2e8f0" text-anchor="middle" font-family="ui-monospace, monospace">${escapeXML(a.symbol)}</text>`;
    }
  }

  // Box outline.
  body += `<rect x="0" y="0" width="${plotW}" height="${plotH}" fill="none" stroke="${CHART_COLORS.textFaded}" stroke-width="1"/>`;

  // Axes tick labels (just min/mid/max).
  const xMid = (xMin + xMax) / 2;
  const yMid = (yMin + yMax) / 2;
  body += `<text x="0" y="${plotH + 18}" font-size="10" fill="${CHART_COLORS.textMid}" font-family="ui-monospace, monospace">${fmtNum(xMin, 3)}</text>`;
  body += `<text x="${plotW / 2}" y="${plotH + 18}" font-size="10" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${fmtNum(xMid, 3)}</text>`;
  body += `<text x="${plotW}" y="${plotH + 18}" font-size="10" fill="${CHART_COLORS.textMid}" text-anchor="end" font-family="ui-monospace, monospace">${fmtNum(xMax, 3)}</text>`;
  body += `<text x="-8" y="${plotH}" font-size="10" fill="${CHART_COLORS.textMid}" text-anchor="end" font-family="ui-monospace, monospace">${fmtNum(yMin, 3)}</text>`;
  body += `<text x="-8" y="${plotH / 2 + 4}" font-size="10" fill="${CHART_COLORS.textMid}" text-anchor="end" font-family="ui-monospace, monospace">${fmtNum(yMid, 3)}</text>`;
  body += `<text x="-8" y="8" font-size="10" fill="${CHART_COLORS.textMid}" text-anchor="end" font-family="ui-monospace, monospace">${fmtNum(yMax, 3)}</text>`;
  body += `<text x="${plotW / 2}" y="${plotH + 35}" font-size="11" font-weight="600" fill="${CHART_COLORS.text}" text-anchor="middle">x</text>`;

  return open + body + chartFooter();
}

/** Crude contour line via cell-boundary linear interpolation (no marching squares). */
function contourLine(
  values: ReadonlyArray<number> | Float64Array | Float32Array,
  nx: number, ny: number, cellW: number, cellH: number, plotH: number,
  level: number,
): string {
  let s = "";
  for (let iy = 0; iy < ny - 1; iy++) {
    for (let ix = 0; ix < nx - 1; ix++) {
      const v00 = values[ix + iy * nx]!;
      const v10 = values[(ix + 1) + iy * nx]!;
      const v01 = values[ix + (iy + 1) * nx]!;
      // Check whether `level` is crossed by the bottom edge (v00→v10) or right edge (v00→v01).
      const x0 = ix * cellW, y0 = plotH - iy * cellH;
      if ((v00 - level) * (v10 - level) < 0) {
        const t = (level - v00) / (v10 - v00);
        const px = x0 + t * cellW;
        s += `<circle cx="${px.toFixed(1)}" cy="${y0.toFixed(1)}" r="0.7" fill="#020617" opacity="0.7"/>`;
      }
      if ((v00 - level) * (v01 - level) < 0) {
        const t = (level - v00) / (v01 - v00);
        const py = y0 - t * cellH;
        s += `<circle cx="${x0.toFixed(1)}" cy="${py.toFixed(1)}" r="0.7" fill="#020617" opacity="0.7"/>`;
      }
    }
  }
  return s;
}
