// ─────────────────────────────────────────────────────────────
// dispersion-d2-chart.ts — pairwise Grimme-D2 contributions
// rendered as a horizontal bar chart sorted by |E_ij|.
//
// Input shape matches DispersionD2Result.pairs from
// src/chemistry/dispersion-d2.ts. Each bar shows one (A, B) atom
// pair with its attractive contribution (mHa) and bond distance.
// ─────────────────────────────────────────────────────────────

import {
  CHART_COLORS, chartHeader, chartFooter,
  escapeXML, fmtNum, type ChartFrame,
} from "./chart-base.js";

export interface DispersionPair {
  readonly indexA: number;
  readonly indexB: number;
  readonly distanceBohr: number;
  readonly contributionHartree: number;
}

export interface DispersionD2ChartOpts {
  readonly title?: string;
  readonly atomLabels?: readonly string[];
  readonly maxBars?: number;
  readonly totalHartree?: number;
  readonly frame?: ChartFrame;
}

export function dispersionD2Chart(
  pairs: readonly DispersionPair[],
  opts: DispersionD2ChartOpts = {},
): string {
  const frame: ChartFrame = opts.frame ?? {
    width: 720, height: 380,
    margin: { top: 50, right: 30, bottom: 50, left: 120 },
  };
  const title = opts.title ?? "Grimme D2 — pairwise dispersion";
  const maxBars = opts.maxBars ?? 12;
  const labels = opts.atomLabels;

  // Sort pairs by magnitude descending, take top-K.
  const sorted = pairs.slice().sort(
    (a, b) => Math.abs(b.contributionHartree) - Math.abs(a.contributionHartree),
  ).slice(0, maxBars);

  const { open, plotW, plotH } = chartHeader(frame, title);
  if (sorted.length === 0) {
    return open
      + `<text x="${plotW / 2}" y="${plotH / 2}" font-size="14" fill="${CHART_COLORS.textFaded}" text-anchor="middle">no D2 pairs</text>`
      + chartFooter();
  }

  // Plot title above the area.
  let body = `<text x="${plotW / 2}" y="-28" font-size="14" font-weight="700" fill="${CHART_COLORS.text}" text-anchor="middle">${escapeXML(title)}</text>`;
  if (opts.totalHartree !== undefined) {
    const mHa = opts.totalHartree * 1000;
    body += `<text x="${plotW / 2}" y="-10" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle" font-family="ui-monospace, monospace">total: ${fmtNum(mHa, 4)} mHa</text>`;
  }

  // Domain: max |contribution| in mHa.
  const maxMHa = sorted.reduce(
    (m, p) => Math.max(m, Math.abs(p.contributionHartree) * 1000), 0,
  ) || 1;

  const rowH = plotH / sorted.length;
  const barH = Math.min(rowH * 0.6, 22);
  const barOff = (rowH - barH) / 2;

  for (let i = 0; i < sorted.length; i++) {
    const p = sorted[i]!;
    const mHa = p.contributionHartree * 1000;
    const w = (Math.abs(mHa) / maxMHa) * plotW;
    const yTop = i * rowH + barOff;
    const lbl = labels
      ? `${labels[p.indexA] ?? p.indexA}-${labels[p.indexB] ?? p.indexB}`
      : `${p.indexA}-${p.indexB}`;
    const R_A = (p.distanceBohr * 0.529177).toFixed(2); // Å
    body += `<text x="-10" y="${yTop + barH / 2 + 4}" font-size="11" fill="${CHART_COLORS.text}" font-family="ui-monospace, monospace" text-anchor="end">${escapeXML(lbl)}</text>`;
    body += `<rect x="0" y="${yTop}" width="${w}" height="${barH}" fill="${CHART_COLORS.lavender}" fill-opacity="0.85" rx="2"/>`;
    body += `<text x="${w + 6}" y="${yTop + barH / 2 + 4}" font-size="10" fill="${CHART_COLORS.textMid}" font-family="ui-monospace, monospace">${fmtNum(mHa, 3)} mHa · ${R_A} Å</text>`;
  }

  // X-axis legend.
  body += `<line x1="0" y1="${plotH}" x2="${plotW}" y2="${plotH}" stroke="${CHART_COLORS.textFaded}"/>`;
  body += `<text x="${plotW / 2}" y="${plotH + 30}" font-size="11" fill="${CHART_COLORS.textMid}" text-anchor="middle">|E_disp(A,B)|  (mHa, attractive)</text>`;

  return open + body + chartFooter();
}
