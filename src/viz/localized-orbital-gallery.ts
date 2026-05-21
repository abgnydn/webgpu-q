// ─────────────────────────────────────────────────────────────
// localized-orbital-gallery.ts — composite SVG laying out
// multiple 2D-slice orbital plots in a responsive grid. Consumer
// supplies precomputed slices (e.g. from Foster-Boys or
// Pipek-Mezey output projected onto the molecular plane).
//
// Pure layout — delegates each panel to contour2DChart.
// ─────────────────────────────────────────────────────────────

import { contour2DChart, type Contour2DOpts } from "./contour-2d-chart.js";
import { CHART_COLORS, escapeXML, fmtNum } from "./chart-base.js";

export interface OrbitalSlice {
  /** Display title (e.g. "LMO 1 · σ(O-H)"). */
  readonly title: string;
  /** Flat Float-like grid of length nx·ny. */
  readonly values: ReadonlyArray<number> | Float64Array | Float32Array;
  readonly nx: number;
  readonly ny: number;
  readonly xMin: number;
  readonly xMax: number;
  readonly yMin: number;
  readonly yMax: number;
  /** Optional one-line caption: e.g. centroid + spread. */
  readonly caption?: string;
}

export interface LocalizedOrbitalGalleryOpts {
  readonly title?: string;
  /** Atoms projected to (x, y) for the overlay on every panel. */
  readonly atoms?: ReadonlyArray<{
    readonly symbol: string;
    readonly x: number;
    readonly y: number;
  }>;
  /** Panels per row. Default 3. */
  readonly columns?: number;
  /** Per-panel dimensions (px). Default 280×280. */
  readonly panelWidth?: number;
  readonly panelHeight?: number;
}

export function localizedOrbitalGallery(
  slices: readonly OrbitalSlice[],
  opts: LocalizedOrbitalGalleryOpts = {},
): string {
  const title = opts.title ?? "Localized molecular orbitals";
  const columns = opts.columns ?? 3;
  const panelW = opts.panelWidth ?? 280;
  const panelH = opts.panelHeight ?? 280;
  const headerH = 60;
  const rowCount = Math.max(1, Math.ceil(slices.length / columns));
  const totalW = columns * panelW;
  const totalH = headerH + rowCount * panelH + 20;

  let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${totalW} ${totalH}" width="100%" height="auto" role="img" aria-label="${escapeXML(title)}">`;
  svg += `<defs><linearGradient id="lobg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${CHART_COLORS.bg0}"/><stop offset="100%" stop-color="${CHART_COLORS.bg1}"/></linearGradient></defs>`;
  svg += `<rect width="${totalW}" height="${totalH}" fill="url(#lobg)" rx="6"/>`;
  svg += `<text x="${totalW / 2}" y="32" font-size="18" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle">${escapeXML(title)}</text>`;
  svg += `<text x="${totalW / 2}" y="52" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">${slices.length} localized orbitals</text>`;

  if (slices.length === 0) {
    svg += `<text x="${totalW / 2}" y="${totalH / 2}" font-size="14" fill="${CHART_COLORS.textFaded}" text-anchor="middle">no orbitals</text></svg>`;
    return svg;
  }

  for (let i = 0; i < slices.length; i++) {
    const slice = slices[i]!;
    const col = i % columns;
    const row = Math.floor(i / columns);
    const x = col * panelW;
    const y = headerH + row * panelH;
    const innerOpts: Contour2DOpts = {
      title: slice.title,
      nx: slice.nx, ny: slice.ny,
      xMin: slice.xMin, xMax: slice.xMax,
      yMin: slice.yMin, yMax: slice.yMax,
      colormap: "divergent",
      ...(opts.atoms ? { atoms: opts.atoms } : {}),
      frame: {
        width: panelW, height: panelH,
        margin: { top: 30, right: 16, bottom: 32, left: 30 },
      },
      fieldLabel: slice.caption ?? "ψ_LMO",
    };
    // contour2DChart returns a full <svg> — we need to embed it as a nested <svg> via x/y offset.
    const inner = contour2DChart(slice.values, innerOpts);
    // Strip outer <svg xmlns…> and replace with <g transform="…">; safe because we know the structure.
    const innerInner = inner.replace(/^<svg[^>]*>/, "").replace(/<\/svg>$/, "");
    svg += `<g transform="translate(${x} ${y})">${innerInner}</g>`;
  }

  return svg + `</svg>`;
}

/** Helper: build a simple caption string for an LMO panel. */
export function lmoCaption(centroid: readonly [number, number, number], spread: number): string {
  return `r̄ = (${fmtNum(centroid[0], 2)}, ${fmtNum(centroid[1], 2)}, ${fmtNum(centroid[2], 2)})   σ = ${fmtNum(spread, 3)}`;
}
