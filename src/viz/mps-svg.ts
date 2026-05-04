// ─────────────────────────────────────────────────────────────
// mps-svg.ts — render the MPS bond network as inline SVG, with
// optional per-site magnetization arrows when the snapshot
// includes expectation values.
//
// Layout: N qubits as nodes on a horizontal line. Each node is
// a circle; if expectations are present we draw an arrow inside
// pointing in the (⟨σ_x⟩, ⟨σ_z⟩) plane (color encodes |m| =
// √(⟨σ_z⟩² + ⟨σ_x⟩²)). Bonds are colored arcs whose thickness
// ∝ Schmidt entropy at that cut.
// ─────────────────────────────────────────────────────────────

import type { MPSSnapshot } from "./mps-recorder.js";

interface SVGOpts {
  root: SVGSVGElement;
  nQubits: number;
  maxEntropy: number;
  snapshot: MPSSnapshot;
}

const PADDING_X = 30;
const NODE_R = 12;
const ARROW_LEN = 10;
const MIN_THICKNESS = 1.5;
const MAX_THICKNESS = 14;
const SVG_NS = "http://www.w3.org/2000/svg";

export function renderMPSNetwork({ root, nQubits, maxEntropy, snapshot }: SVGOpts): void {
  const w = root.clientWidth || 800;
  const h = root.clientHeight || 140;
  while (root.firstChild) root.removeChild(root.firstChild);

  const defs = document.createElementNS(SVG_NS, "defs");
  defs.innerHTML = `
    <filter id="bond-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="4" markerHeight="4" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
    </marker>
  `;
  root.appendChild(defs);

  if (nQubits < 2) return;
  const ySite = h / 2;
  const xs: number[] = [];
  const usableW = w - 2 * PADDING_X;
  for (let i = 0; i < nQubits; i++) {
    xs.push(PADDING_X + (usableW * i) / (nQubits - 1));
  }

  // Bonds first (under the nodes).
  for (let i = 0; i < nQubits - 1; i++) {
    const e = snapshot.entropies[i] ?? 0;
    const t = maxEntropy > 0 ? Math.min(1, e / maxEntropy) : 0;
    const thickness = MIN_THICKNESS + (MAX_THICKNESS - MIN_THICKNESS) * Math.pow(t, 0.6);
    const r = Math.round(60 + 195 * t);
    const g = Math.round(120 + 60 * (1 - t));
    const b = Math.round(255 - 200 * t);

    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", xs[i]!.toString());
    line.setAttribute("y1", ySite.toString());
    line.setAttribute("x2", xs[i + 1]!.toString());
    line.setAttribute("y2", ySite.toString());
    line.setAttribute("stroke", `rgb(${r},${g},${b})`);
    line.setAttribute("stroke-width", thickness.toFixed(2));
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("opacity", (0.45 + 0.55 * t).toFixed(2));
    line.setAttribute("filter", "url(#bond-glow)");
    root.appendChild(line);

    const chi = snapshot.bondDims[i] ?? 1;
    const label = document.createElementNS(SVG_NS, "text");
    label.setAttribute("x", ((xs[i]! + xs[i + 1]!) / 2).toFixed(1));
    label.setAttribute("y", (ySite + 32).toFixed(1));
    label.setAttribute("font-size", "10");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", t > 0.3 ? "#ffd9a8" : "#7a8294");
    label.setAttribute("font-family", "ui-monospace, Menlo, monospace");
    label.textContent = `χ${chi}`;
    root.appendChild(label);
  }

  const exp = snapshot.expectations;

  // Nodes on top.
  for (let i = 0; i < nQubits; i++) {
    const cx = xs[i]!;
    const cy = ySite;
    let fill = "#0e1320";
    let stroke = "#6ea8ff";

    if (exp) {
      // Map magnitude to fill brightness; large m_z → blue/red, large m_x → cyan/orange.
      const mz = exp.sz[i] ?? 0;
      const mx = exp.sx[i] ?? 0;
      const mag = Math.min(1, Math.sqrt(mz * mz + mx * mx));
      // Color: ferro (high |m_z|) → orange-warm; para (high |m_x|) → cyan;
      // mixed/critical → pale.
      const warm = Math.abs(mz) / Math.max(1e-6, Math.abs(mz) + Math.abs(mx));
      const r = Math.round(70 + 180 * mag * warm);
      const g = Math.round(110 + 60 * mag);
      const b = Math.round(180 - 100 * mag * warm);
      fill = `rgb(${r},${g},${b})`;
      stroke = `rgb(${Math.min(255, r + 40)},${Math.min(255, g + 40)},${Math.min(255, b + 40)})`;
    }

    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", cx.toString());
    c.setAttribute("cy", cy.toString());
    c.setAttribute("r", NODE_R.toString());
    c.setAttribute("fill", fill);
    c.setAttribute("stroke", stroke);
    c.setAttribute("stroke-width", "1.6");
    root.appendChild(c);

    if (exp) {
      const mz = exp.sz[i] ?? 0;
      const mx = exp.sx[i] ?? 0;
      // SVG's y-axis is inverted: positive m_z (up) → arrow head at smaller y.
      const head = document.createElementNS(SVG_NS, "line");
      head.setAttribute("x1", cx.toString());
      head.setAttribute("y1", cy.toString());
      head.setAttribute("x2", (cx + mx * ARROW_LEN).toFixed(2));
      head.setAttribute("y2", (cy - mz * ARROW_LEN).toFixed(2));
      head.setAttribute("stroke", "#ffffff");
      head.setAttribute("stroke-width", "2");
      head.setAttribute("stroke-linecap", "round");
      head.setAttribute("opacity", "0.95");
      head.setAttribute("marker-end", "url(#arrow)");
      head.setAttribute("color", "#ffffff");
      root.appendChild(head);
    }

    const lab = document.createElementNS(SVG_NS, "text");
    lab.setAttribute("x", cx.toString());
    lab.setAttribute("y", (cy - NODE_R - 6).toFixed(1));
    lab.setAttribute("font-size", "10");
    lab.setAttribute("text-anchor", "middle");
    lab.setAttribute("fill", "#7a8294");
    lab.setAttribute("font-family", "ui-monospace, Menlo, monospace");
    lab.textContent = `q${i}`;
    root.appendChild(lab);
  }

  const caption = document.createElementNS(SVG_NS, "text");
  caption.setAttribute("x", (w / 2).toString());
  caption.setAttribute("y", (h - 8).toString());
  caption.setAttribute("font-size", "11");
  caption.setAttribute("text-anchor", "middle");
  caption.setAttribute("fill", "#9aa3b3");
  caption.setAttribute("font-family", "ui-monospace, Menlo, monospace");
  caption.textContent = snapshot.gateLabel;
  root.appendChild(caption);
}
