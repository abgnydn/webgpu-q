// ─────────────────────────────────────────────────────────────
// trajectory-svg.ts — render a quantum trajectory as a layered
// SVG: sites×time heatmap of ⟨σ_z⟩ on top, measurement events
// overlaid as marks, mid-cut entanglement entropy plotted as a
// thin line on the bottom strip.
//
// The picture in one figure:
//   • blue/red rows = ⟨σ_z⟩ at each site over time (magnetization
//     pattern under unitary evolution + measurement collapse)
//   • white circle = measurement happened, |0⟩ outcome
//   • orange circle = measurement happened, |1⟩ outcome
//   • cyan curve along the bottom = entanglement entropy S(t)
//
// At γ = 0: smooth fluctuating heatmap, S(t) climbs ballistically.
// At γ ≫ 1: measurements pepper the heatmap, S(t) stays low.
// At γ ≈ γ_c: scattered measurements, S(t) climbs slowly /
//             logarithmically — the MIPT signature.
// ─────────────────────────────────────────────────────────────

import type { TrajectoryRecording } from "../manybody/trajectory.js";

const SVG_NS = "http://www.w3.org/2000/svg";

interface TrajOpts {
  root: SVGSVGElement;
  recording: TrajectoryRecording;
  cursorT?: number;
}

const PAD_L = 36;
const PAD_R = 12;
const PAD_T = 22;
const PAD_B = 56;       // leaves room for the entropy strip
const ENTROPY_H = 36;   // height of the entropy strip below

export function renderTrajectory({ root, recording, cursorT }: TrajOpts): void {
  while (root.firstChild) root.removeChild(root.firstChild);
  const w = root.clientWidth || 800;
  const h = root.clientHeight || 200;
  const T = recording.history.length;
  const N = recording.nQubits;
  if (T < 2 || N < 1) return;

  const innerW = w - PAD_L - PAD_R;
  const innerH = h - PAD_T - PAD_B;
  const cellW = innerW / T;
  const cellH = innerH / N;

  // ── Heatmap of ⟨σ_z⟩ ─────────────────────────────
  for (let t = 0; t < T; t++) {
    const step = recording.history[t]!;
    for (let q = 0; q < N; q++) {
      const sz = step.sz[q] ?? 0;
      const v = Math.max(-1, Math.min(1, sz));
      // Diverging color: -1 (red-ish) → 0 (dark) → +1 (blue-ish).
      let r: number, g: number, b: number;
      if (v >= 0) {
        // 0 to +1 — blue ramp.
        const t01 = v;
        r = Math.round(40 + 60 * t01);
        g = Math.round(80 + 100 * t01);
        b = Math.round(110 + 140 * t01);
      } else {
        // 0 to -1 — warm/red ramp.
        const t01 = -v;
        r = Math.round(60 + 200 * t01);
        g = Math.round(60 + 50 * t01);
        b = Math.round(80 + 30 * t01);
      }
      const cell = document.createElementNS(SVG_NS, "rect");
      cell.setAttribute("x", (PAD_L + t * cellW).toFixed(2));
      cell.setAttribute("y", (PAD_T + q * cellH).toFixed(2));
      cell.setAttribute("width", (cellW + 0.4).toFixed(2));
      cell.setAttribute("height", (cellH + 0.4).toFixed(2));
      cell.setAttribute("fill", `rgb(${r},${g},${b})`);
      root.appendChild(cell);
    }
  }

  // ── Measurement marks ─────────────────────────────
  for (let t = 0; t < T; t++) {
    const step = recording.history[t]!;
    for (const m of step.measurements) {
      const cx = PAD_L + (t + 0.5) * cellW;
      const cy = PAD_T + (m.site + 0.5) * cellH;
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", cx.toFixed(1));
      dot.setAttribute("cy", cy.toFixed(1));
      dot.setAttribute("r", String(Math.min(3, cellH / 3)));
      dot.setAttribute("fill", m.outcome === 1 ? "#ffffff" : "#ffb56e");
      dot.setAttribute("stroke", "#0b0d12");
      dot.setAttribute("stroke-width", "0.7");
      root.appendChild(dot);
    }
  }

  // ── Entropy line (mid-cut) along the bottom strip ─────────
  const stripY = PAD_T + innerH + 8;
  const sMax = Math.max(0.01, recording.maxEntropy);
  const stripBg = document.createElementNS(SVG_NS, "rect");
  stripBg.setAttribute("x", PAD_L.toString());
  stripBg.setAttribute("y", stripY.toString());
  stripBg.setAttribute("width", innerW.toString());
  stripBg.setAttribute("height", ENTROPY_H.toString());
  stripBg.setAttribute("fill", "#11141d");
  stripBg.setAttribute("stroke", "#1b2030");
  root.appendChild(stripBg);

  let d = "";
  for (let t = 0; t < T; t++) {
    const e = recording.history[t]!.midcutEntropy;
    const x = (PAD_L + t * cellW + cellW / 2).toFixed(1);
    const y = (stripY + ENTROPY_H - (e / sMax) * (ENTROPY_H - 4) - 2).toFixed(1);
    d += (t === 0 ? "M " : " L ") + x + " " + y;
  }
  const line = document.createElementNS(SVG_NS, "path");
  line.setAttribute("d", d);
  line.setAttribute("stroke", "#6ea8ff");
  line.setAttribute("stroke-width", "1.6");
  line.setAttribute("fill", "none");
  root.appendChild(line);

  // ── Axis / labels ─────────────────────────────────
  const axis = document.createElementNS(SVG_NS, "g");
  axis.setAttribute("font-family", "ui-monospace, monospace");
  axis.setAttribute("fill", "#9aa3b3");
  axis.setAttribute("font-size", "10");
  axis.innerHTML = `
    <text x="${(PAD_L + innerW / 2).toFixed(1)}" y="${(stripY + ENTROPY_H + 14).toFixed(1)}" text-anchor="middle">time t →   ·   bottom: mid-cut entropy S(t)</text>
    <text x="${PAD_L - 6}" y="${(PAD_T + innerH / 2).toFixed(1)}" text-anchor="end" transform="rotate(-90 ${PAD_L - 6} ${PAD_T + innerH / 2})">site q ↑</text>
    <text x="${PAD_L}" y="${(PAD_T - 6).toFixed(1)}" text-anchor="start" font-size="10" fill="#7a8294">N=${N} · γ=${recording.gamma.toFixed(2)} · ${T - 1} steps · max S=${recording.maxEntropy.toFixed(2)} bits · ⊙ |0⟩  ⦿ |1⟩</text>
  `;
  root.appendChild(axis);

  // ── Cursor at current snapshot ────────────────────
  if (typeof cursorT === "number" && cursorT >= 0 && cursorT < T) {
    const x = PAD_L + (cursorT + 0.5) * cellW;
    const cur = document.createElementNS(SVG_NS, "line");
    cur.setAttribute("x1", x.toFixed(1));
    cur.setAttribute("y1", PAD_T.toString());
    cur.setAttribute("x2", x.toFixed(1));
    cur.setAttribute("y2", (stripY + ENTROPY_H).toString());
    cur.setAttribute("stroke", "#ffffff");
    cur.setAttribute("stroke-width", "1.4");
    cur.setAttribute("opacity", "0.85");
    root.appendChild(cur);
  }
}
