// ─────────────────────────────────────────────────────────────
// order-svg.ts — render a sweep of (h, m_z, m_x, S) datapoints
// as a compact SVG line chart. Two series:
//   • orange: |m_z| (Z-magnetization, the symmetry-broken order
//     parameter — drops to 0 at criticality going right)
//   • cyan:   bipartite mid-cut entropy S (peaks at criticality)
//
// A vertical dashed marker on the current h slider is drawn so
// you can see where you sit on the curve in real time.
// ─────────────────────────────────────────────────────────────

import type { SweepPoint } from "./order-sweep.js";

const SVG_NS = "http://www.w3.org/2000/svg";

interface OrderSvgOpts {
  root: SVGSVGElement;
  points: readonly SweepPoint[];
  cursorH?: number;
}

const PAD_L = 32;
const PAD_R = 12;
const PAD_T = 18;
const PAD_B = 24;

export function renderOrderSweep({ root, points, cursorH }: OrderSvgOpts): void {
  while (root.firstChild) root.removeChild(root.firstChild);
  const w = root.clientWidth || 280;
  const h = root.clientHeight || 130;
  if (points.length < 2) return;

  const xMin = points[0]!.h;
  const xMax = points[points.length - 1]!.h;
  const innerW = w - PAD_L - PAD_R;
  const innerH = h - PAD_T - PAD_B;

  function xPx(hVal: number): number {
    return PAD_L + ((hVal - xMin) / (xMax - xMin)) * innerW;
  }
  function yPxNorm(v01: number): number {
    return PAD_T + (1 - v01) * innerH;
  }

  // Magnetization is in [0, 1]. Entropy is in [0, ln(2) · ⌊N/2⌋] roughly;
  // normalize by max observed for plotting.
  const maxS = points.reduce((a, p) => Math.max(a, p.maxEntropy), 0) || 1;

  // Axes.
  const axis = document.createElementNS(SVG_NS, "g");
  axis.setAttribute("stroke", "#3a4054");
  axis.setAttribute("fill", "none");
  axis.innerHTML = `
    <line x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${PAD_T + innerH}" />
    <line x1="${PAD_L}" y1="${PAD_T + innerH}" x2="${PAD_L + innerW}" y2="${PAD_T + innerH}" />
  `;
  root.appendChild(axis);

  // x-axis label
  const xLab = document.createElementNS(SVG_NS, "text");
  xLab.setAttribute("x", (PAD_L + innerW / 2).toFixed(1));
  xLab.setAttribute("y", (h - 6).toFixed(1));
  xLab.setAttribute("text-anchor", "middle");
  xLab.setAttribute("font-size", "10");
  xLab.setAttribute("fill", "#7a8294");
  xLab.setAttribute("font-family", "ui-monospace, monospace");
  xLab.textContent = "h / J";
  root.appendChild(xLab);

  // Critical-point vertical (h = 1 in TFIM with J=1) — annotate the textbook QPT.
  if (xMin <= 1 && xMax >= 1) {
    const crit = document.createElementNS(SVG_NS, "line");
    crit.setAttribute("x1", xPx(1).toFixed(1));
    crit.setAttribute("y1", PAD_T.toString());
    crit.setAttribute("x2", xPx(1).toFixed(1));
    crit.setAttribute("y2", (PAD_T + innerH).toString());
    crit.setAttribute("stroke", "#ffd9a8");
    crit.setAttribute("stroke-width", "1");
    crit.setAttribute("stroke-dasharray", "3,3");
    crit.setAttribute("opacity", "0.5");
    root.appendChild(crit);
    const critLab = document.createElementNS(SVG_NS, "text");
    critLab.setAttribute("x", xPx(1).toFixed(1));
    critLab.setAttribute("y", (PAD_T + 10).toFixed(1));
    critLab.setAttribute("text-anchor", "middle");
    critLab.setAttribute("font-size", "9");
    critLab.setAttribute("fill", "#ffd9a8");
    critLab.setAttribute("opacity", "0.7");
    critLab.setAttribute("font-family", "ui-monospace, monospace");
    critLab.textContent = "h_c";
    root.appendChild(critLab);
  }

  // |m_z| series — orange.
  const mzPath = document.createElementNS(SVG_NS, "path");
  let d = "";
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    const x = xPx(p.h).toFixed(1);
    const y = yPxNorm(p.mZ).toFixed(1);
    d += (i === 0 ? "M " : " L ") + x + " " + y;
  }
  mzPath.setAttribute("d", d);
  mzPath.setAttribute("stroke", "#ffb56e");
  mzPath.setAttribute("stroke-width", "1.8");
  mzPath.setAttribute("fill", "none");
  root.appendChild(mzPath);

  // Entropy series — cyan, normalized.
  const sPath = document.createElementNS(SVG_NS, "path");
  let dS = "";
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    const x = xPx(p.h).toFixed(1);
    const y = yPxNorm(p.maxEntropy / maxS).toFixed(1);
    dS += (i === 0 ? "M " : " L ") + x + " " + y;
  }
  sPath.setAttribute("d", dS);
  sPath.setAttribute("stroke", "#6ea8ff");
  sPath.setAttribute("stroke-width", "1.5");
  sPath.setAttribute("fill", "none");
  sPath.setAttribute("opacity", "0.85");
  root.appendChild(sPath);

  // Legend.
  const legend = document.createElementNS(SVG_NS, "g");
  legend.innerHTML = `
    <rect x="${PAD_L + 6}" y="${PAD_T + 4}" width="9" height="9" fill="#ffb56e"/>
    <text x="${PAD_L + 18}" y="${PAD_T + 12}" font-size="9" fill="#d8dde6"
          font-family="ui-monospace, monospace">|m_z|</text>
    <rect x="${PAD_L + 60}" y="${PAD_T + 4}" width="9" height="9" fill="#6ea8ff"/>
    <text x="${PAD_L + 72}" y="${PAD_T + 12}" font-size="9" fill="#d8dde6"
          font-family="ui-monospace, monospace">S/S_max</text>
  `;
  root.appendChild(legend);

  // Cursor mark.
  if (typeof cursorH === "number" && cursorH >= xMin && cursorH <= xMax) {
    const cursor = document.createElementNS(SVG_NS, "line");
    cursor.setAttribute("x1", xPx(cursorH).toFixed(1));
    cursor.setAttribute("y1", PAD_T.toString());
    cursor.setAttribute("x2", xPx(cursorH).toFixed(1));
    cursor.setAttribute("y2", (PAD_T + innerH).toString());
    cursor.setAttribute("stroke", "#ffffff");
    cursor.setAttribute("stroke-width", "1.2");
    cursor.setAttribute("opacity", "0.7");
    root.appendChild(cursor);
  }
}
