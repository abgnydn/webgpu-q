// ─────────────────────────────────────────────────────────────
// lightcone-svg.ts — render a sites×time heatmap of bond entropy
// as an inline SVG. Reads an MPSRecording and lays it out with
// time growing rightward and bonds stacked vertically.
//
// Visual: dark when entropy is low, hot orange/yellow when high.
// In a quench dynamics demo the entropy front spreads outward at
// ≤ Lieb-Robinson velocity v_LR — visible as a triangular "light
// cone" emanating from the perturbation source.
// ─────────────────────────────────────────────────────────────

import type { MPSRecording } from "./mps-recorder.js";

const SVG_NS = "http://www.w3.org/2000/svg";

interface LightconeOpts {
  root: SVGSVGElement;
  recording: MPSRecording;
  /** Optional time-cursor (snapshot index) drawn as a vertical line. */
  cursorT?: number;
}

const PAD_L = 32;
const PAD_R = 12;
const PAD_T = 22;
const PAD_B = 24;

export function renderLightcone({ root, recording, cursorT }: LightconeOpts): void {
  while (root.firstChild) root.removeChild(root.firstChild);
  const w = root.clientWidth || 800;
  const h = root.clientHeight || 200;
  const T = recording.snapshots.length;
  const NB = recording.nQubits - 1;
  if (T < 2 || NB < 1) return;

  const innerW = w - PAD_L - PAD_R;
  const innerH = h - PAD_T - PAD_B;
  const cellW = innerW / T;
  const cellH = innerH / NB;
  const maxS = Math.max(0.01, recording.maxEntropy);

  // Render the heatmap as one <rect> per cell. NB·T can be ~2000;
  // cheap to flush on every snapshot scrub.
  for (let t = 0; t < T; t++) {
    const snap = recording.snapshots[t]!;
    for (let b = 0; b < NB; b++) {
      const s = snap.entropies[b] ?? 0;
      const v = Math.min(1, s / maxS);
      // Hot ramp (dark blue → magenta → orange → yellow).
      const r = Math.round(60 + 195 * v);
      const g = Math.round(20 + 200 * Math.pow(v, 1.4));
      const bb = Math.round(120 - 100 * v);

      const cell = document.createElementNS(SVG_NS, "rect");
      cell.setAttribute("x", (PAD_L + t * cellW).toFixed(2));
      cell.setAttribute("y", (PAD_T + b * cellH).toFixed(2));
      cell.setAttribute("width", (cellW + 0.5).toFixed(2));
      cell.setAttribute("height", (cellH + 0.5).toFixed(2));
      cell.setAttribute("fill", `rgb(${r},${g},${bb})`);
      cell.setAttribute("stroke", "none");
      root.appendChild(cell);
    }
  }

  // Axis labels.
  const axis = document.createElementNS(SVG_NS, "g");
  axis.setAttribute("font-family", "ui-monospace, monospace");
  axis.setAttribute("fill", "#9aa3b3");
  axis.setAttribute("font-size", "10");
  axis.innerHTML = `
    <text x="${(PAD_L + innerW / 2).toFixed(1)}" y="${(h - 6).toFixed(1)}" text-anchor="middle">time t →</text>
    <text x="${(PAD_L - 8).toFixed(1)}" y="${(PAD_T + innerH / 2).toFixed(1)}" text-anchor="end" transform="rotate(-90 ${PAD_L - 8} ${PAD_T + innerH / 2})">bond ↑</text>
    <text x="${PAD_L}" y="${(PAD_T - 6).toFixed(1)}" text-anchor="start" font-size="10" fill="#7a8294">${recording.nQubits} sites · ${T - 1} steps · max S=${recording.maxEntropy.toFixed(2)} bits</text>
  `;
  root.appendChild(axis);

  // Cursor at current snapshot.
  if (typeof cursorT === "number" && cursorT >= 0 && cursorT < T) {
    const x = PAD_L + (cursorT + 0.5) * cellW;
    const cur = document.createElementNS(SVG_NS, "line");
    cur.setAttribute("x1", x.toFixed(1));
    cur.setAttribute("y1", PAD_T.toString());
    cur.setAttribute("x2", x.toFixed(1));
    cur.setAttribute("y2", (PAD_T + innerH).toString());
    cur.setAttribute("stroke", "#ffffff");
    cur.setAttribute("stroke-width", "1.4");
    cur.setAttribute("opacity", "0.85");
    root.appendChild(cur);
  }
}
