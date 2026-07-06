// ─────────────────────────────────────────────────────────────
// learn/main.ts — "Why is water special?" — the grows-with-you
// education wedge. One real water molecule, four depths (Feel →
// See → Know → Prove), live Hartree–Fock recomputed on drag.
//
// The engine (quickReport → real RHF/STO-3G) is the invariant;
// depth only re-skins overlays + side panel + copy — it never
// recomputes. Dragging an atom recomputes (the fast HF is the
// product). CPU/WASM path: no GPU, no backend, no install.
//
// The SCF runs in a Web Worker (compute-worker.ts) so a slow
// basis (cc-pVDZ ≈ 1 s) can never jank the drag. Latest-wins:
// at most one request is in flight; newer geometry overwrites
// whatever was waiting; stale replies are dropped by seq.
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "../chemistry/atoms.js";
import type { MolecularReport } from "../chemistry/molecular-report.js";
import type { ComputeReply } from "./compute-worker.js";

type Depth = "feel" | "see" | "know" | "prove";
interface P2 { x: number; y: number } // Ångström, molecule frame (O at origin)
interface AtomXY { sym: "O" | "H"; x: number; y: number }

const SVGNS = "http://www.w3.org/2000/svg";
const SCALE = 95;            // viewBox units per Ångström
const OX = 320, OY = 165;    // O sits here in the 640×420 viewBox
const R0 = 0.958;            // default O–H bond length (Å)
const A0 = 104.5;            // default H–O–H angle (deg)
const COL = { O: "#ff6b6b", H: "#dfe6f5", bond: "#39415a", plus: "#ff8a8a", minus: "#6ea8ff", dip: "#22d3ee" };

const scene = document.getElementById("scene") as unknown as SVGSVGElement;
const side = document.getElementById("side") as HTMLElement;
const hintEl = document.getElementById("hint") as HTMLElement;
const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>(".depth button"));

// ── State ────────────────────────────────────────────────────
const state: {
  depth: Depth;
  atoms: AtomXY[];
  report: MolecularReport | null;
  /** basis that produced `report` — differs from `basis` while a fast preview is on screen */
  reportBasis: BasisName | null;
  basis: BasisName;
  error: string | null;
} = {
  depth: "feel",
  atoms: geometryFromAngle(A0, R0),
  report: null,
  reportBasis: null,
  basis: "sto-3g",
  error: null,
};

function geometryFromAngle(angleDeg: number, r: number): AtomXY[] {
  const h = (angleDeg / 2) * Math.PI / 180;
  return [
    { sym: "O", x: 0, y: 0 },
    { sym: "H", x: -r * Math.sin(h), y: -r * Math.cos(h) },
    { sym: "H", x: r * Math.sin(h), y: -r * Math.cos(h) },
  ];
}

// ── Live compute (Web Worker, latest-wins, fault-tolerant) ───
// The SCF never runs on this thread. At most ONE request is in
// flight in the worker; while it runs, newer geometry overwrites
// the single `queued` slot (so the worker never builds a queue and
// always computes the freshest geometry next). Replies carry the
// request's seq; anything but the in-flight seq is stale → dropped.
// On SCF failure the last good report stays rendered + a warning.
//
// Basis-aware drag strategy: "live" requests (drag / slider /
// animation in progress) run STO-3G when the selected basis is
// cc-pVDZ, so feedback stays ~ms-fast; "final" requests (release)
// run the selected basis. While a preview-basis result is on
// screen, renderSide says so honestly (know/prove depths).
const worker = new Worker(new URL("./compute-worker.ts", import.meta.url), { type: "module" });
let computeSeq = 0;
let inFlight: { seq: number; basis: BasisName } | null = null;
let queued: { atoms: Atom[]; basis: BasisName } | null = null;
let finalTimer: number | null = null;

function snapshotAtoms(): Atom[] {
  return state.atoms.map((a) => ({ symbol: a.sym, pos: [a.x, a.y, 0] as const }));
}

function post(atoms: Atom[], basis: BasisName): void {
  computeSeq += 1;
  inFlight = { seq: computeSeq, basis };
  worker.postMessage({ atoms, basis, seq: computeSeq });
}

/** live = in-progress feedback (drag / slider / animation frame);
 *  final = settled geometry, always in the SELECTED basis. */
function requestCompute(kind: "live" | "final"): void {
  const basis: BasisName = kind === "live" && state.basis === "cc-pvdz" ? "sto-3g" : state.basis;
  if (kind === "final" && finalTimer !== null) { clearTimeout(finalTimer); finalTimer = null; }
  if (kind === "live" && basis !== state.basis) scheduleFinal();  // safety net: converge to the selected basis even if a release event is missed
  const atoms = snapshotAtoms();
  if (inFlight) { queued = { atoms, basis }; return; }   // latest wins — overwrite whatever waited
  post(atoms, basis);
}

function scheduleFinal(): void {
  if (finalTimer !== null) clearTimeout(finalTimer);
  finalTimer = window.setTimeout(() => { finalTimer = null; requestCompute("final"); }, 400);
}

worker.addEventListener("message", (ev: MessageEvent<ComputeReply>) => {
  const reply = ev.data;
  if (!inFlight || reply.seq !== inFlight.seq) return;   // stale — a newer request owns the screen
  const basis = inFlight.basis;
  inFlight = null;
  if (reply.report) {
    state.report = reply.report;
    state.reportBasis = basis;
    state.error = null;
  } else {
    state.error = reply.error;   // keep last good report on screen
  }
  if (queued) { const q = queued; queued = null; post(q.atoms, q.basis); }
  render();
});

// ── Geometry helpers ─────────────────────────────────────────
const toPx = (p: P2): P2 => ({ x: OX + p.x * SCALE, y: OY - p.y * SCALE });
const sub = (a: P2, b: P2): P2 => ({ x: a.x - b.x, y: a.y - b.y });
const len = (a: P2): number => Math.hypot(a.x, a.y);
function angleDeg(): number {
  const [, h1, h2] = state.atoms as [AtomXY, AtomXY, AtomXY];
  const a = sub(h1, { x: 0, y: 0 }), b = sub(h2, { x: 0, y: 0 });
  const c = (a.x * b.x + a.y * b.y) / (len(a) * len(b) || 1);
  return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
}
function bondLen(): number {
  const [, h1] = state.atoms as [AtomXY, AtomXY, AtomXY];
  return len(h1);
}

// ── SVG element factory ──────────────────────────────────────
function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, txt?: string): SVGElementTagNameMap[K] {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (txt !== undefined) n.textContent = txt;
  return n;
}

// ── Render: the scene (molecule + depth overlays) ────────────
function renderScene(): void {
  while (scene.firstChild) scene.removeChild(scene.firstChild);
  const rep = state.report;
  const depth = state.depth;
  const px = state.atoms.map(toPx);
  const O = px[0]!, H1 = px[1]!, H2 = px[2]!;

  // Bonds
  for (const H of [H1, H2]) scene.appendChild(el("line", { x1: O.x, y1: O.y, x2: H.x, y2: H.y, stroke: "#39415a", "stroke-width": 8, "stroke-linecap": "round" }));

  // Feel: soft glow halos
  if (depth === "feel") {
    scene.appendChild(el("circle", { cx: O.x, cy: O.y, r: 46, fill: "#6ea8ff", opacity: 0.16 }));
    for (const H of [H1, H2]) scene.appendChild(el("circle", { cx: H.x, cy: H.y, r: 34, fill: "#ff8a8a", opacity: 0.16 }));
  }

  // Dipole arrow (See/Know/Prove) — canonical: from H-midpoint toward the − end (O), length ∝ |μ|.
  if (rep && depth !== "feel") {
    const hmid: P2 = { x: (H1.x + H2.x) / 2, y: (H1.y + H2.y) / 2 };
    const dir = sub(O, hmid); const d = len(dir) || 1;
    const mag = Math.max(0, Math.min(150, rep.dipoleMagnitudeDebye * 55));
    const tip: P2 = { x: hmid.x + dir.x / d * mag, y: hmid.y + dir.y / d * mag };
    if (mag > 4) {
      scene.appendChild(el("line", { x1: hmid.x, y1: hmid.y, x2: tip.x, y2: tip.y, stroke: COL.dip, "stroke-width": 4 }));
      const ang = Math.atan2(tip.y - hmid.y, tip.x - hmid.x);
      const ah = 11;
      const p1 = `${tip.x},${tip.y}`;
      const p2 = `${tip.x - ah * Math.cos(ang - 0.4)},${tip.y - ah * Math.sin(ang - 0.4)}`;
      const p3 = `${tip.x - ah * Math.cos(ang + 0.4)},${tip.y - ah * Math.sin(ang + 0.4)}`;
      scene.appendChild(el("polygon", { points: `${p1} ${p2} ${p3}`, fill: COL.dip }));
      scene.appendChild(el("text", { x: tip.x + 8, y: tip.y, fill: COL.dip, "font-size": 13, "font-weight": 600 }, "dipole"));
    }
  }

  // Atoms (O static, H draggable)
  state.atoms.forEach((a, i) => {
    const p = px[i]!;
    const r = a.sym === "O" ? 26 : 19;
    const c = el("circle", { cx: p.x, cy: p.y, r, fill: a.sym === "O" ? COL.O : "#c9d4ea", stroke: "#0b0d12", "stroke-width": 2 });
    if (a.sym === "H") { c.setAttribute("cursor", "grab"); c.setAttribute("data-i", String(i)); c.classList.add("draggable"); }
    scene.appendChild(c);
    scene.appendChild(el("text", { x: p.x, y: p.y + 5, "text-anchor": "middle", fill: a.sym === "O" ? "#2a0d0d" : "#1a2233", "font-size": 15, "font-weight": 700 }, a.sym));

    // Charge badges (See+): sign from real Mulliken charge
    if (rep && depth !== "feel") {
      const q = rep.mullikenCharges[i]!;
      const sign = q < 0 ? "−" : "+";
      const badge = q < 0 ? COL.minus : COL.plus;
      scene.appendChild(el("text", { x: p.x + (a.sym === "O" ? 30 : 22), y: p.y - (a.sym === "O" ? 20 : 14), fill: badge, "font-size": 16, "font-weight": 800 }, sign));
      if (depth === "know" || depth === "prove") {
        scene.appendChild(el("text", { x: p.x + (a.sym === "O" ? 30 : 22), y: p.y - (a.sym === "O" ? 4 : 0), fill: "#8b93a7", "font-size": 11 }, q.toFixed(2)));
      }
    }
  });

  // Angle arc label (See+)
  if (depth !== "feel") {
    scene.appendChild(el("text", { x: O.x + 34, y: O.y + 30, fill: "#96a0b5", "font-size": 12 }, `${angleDeg().toFixed(1)}°`));
  }
}

// ── Render: the side panel (this is where "level" lives) ─────
function esc(s: string): string { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]!)); }

function renderSide(): void {
  const rep = state.report;
  const depth = state.depth;
  const parts: string[] = [];

  if (depth === "feel") {
    parts.push(card("Why it matters",
      `<p>Water has a tiny <b style="color:var(--plus)">+</b> side and a tiny
       <b style="color:var(--minus)">−</b> side. That's why it sticks to itself,
       climbs up a straw, and makes rain drops round.</p>
       <p style="color:var(--dim)">Grab a white atom and move it — the molecule
       springs back to its comfy shape.</p>`));
  }

  if (depth === "see") {
    parts.push(card("Why it matters",
      `<p>The <b>bent</b> shape is the whole trick: the two <b style="color:var(--plus)">+</b>
       pulls don't line up to cancel, so the molecule ends up <b>polar</b> — one side
       plus, one side minus.</p>`));
    parts.push(angleControl());
  }

  if (depth === "know" && rep) {
    parts.push(card("Why it matters",
      `<p>Bent geometry + the two lone pairs on oxygen give a net dipole along the
       bisector. Straighten it and the dipole cancels by symmetry.</p>`));
    parts.push(card("The real numbers",
      kv("Dipole μ", `${rep.dipoleMagnitudeDebye.toFixed(3)} D`) +
      kv("H–O–H angle", `${angleDeg().toFixed(1)}°`) +
      kv("O charge", rep.mullikenCharges[0]!.toFixed(3)) +
      kv("H charge", `${rep.mullikenCharges[1]!.toFixed(3)} ×2`) +
      kv("HOMO–LUMO", `${rep.homoLumoGapEv.toFixed(1)} eV`) +
      kv("Total energy", `${rep.totalEnergy.toFixed(4)} Ha`) +
      previewNote()));
    parts.push(angleControl());
  }

  if (depth === "prove" && rep) {
    parts.push(card("The calculation",
      `<div class="controls" style="margin-bottom:8px">Method <b>RHF</b> · basis
       <select id="basisSel">
         <option value="sto-3g"${state.basis === "sto-3g" ? " selected" : ""}>STO-3G</option>
         <option value="cc-pvdz"${state.basis === "cc-pvdz" ? " selected" : ""}>cc-pVDZ</option>
       </select></div>` +
      kv("SCF energy", `${rep.scfEnergy.toFixed(6)} Ha`) +
      kv("Dipole μ", `${rep.dipoleMagnitudeDebye.toFixed(4)} D`) +
      kv("μ vector (a.u.)", `[${rep.dipole.map((d) => d.toFixed(3)).join(", ")}]`) +
      kv("Mulliken O / H", `${rep.mullikenCharges[0]!.toFixed(3)} / ${rep.mullikenCharges[1]!.toFixed(3)}`) +
      kv("HOMO–LUMO", `${rep.homoLumoGapEv.toFixed(3)} eV`) +
      kv("Reference", rep.multireferenceSeverity) +
      previewNote()));
    parts.push(card("Reproduce this exact run",
      `<p style="margin:0 0 6px;color:var(--dim);font-size:12px">Same URL → same numbers, forever.</p>
       <code>${esc(location.href)}</code>`));
  }

  parts.push(tutorCard());
  if (state.error) parts.push(`<div class="card"><span class="warn">⚠ that geometry didn't converge — showing the last good result. Nudge it back.</span></div>`);

  side.innerHTML = parts.join("");
  wireSide();
}

const BASIS_LABEL: Partial<Record<BasisName, string>> = { "sto-3g": "STO-3G", "cc-pvdz": "cc-pVDZ" };

/** Honesty note (know/prove): while a drag shows fast preview-basis numbers,
 *  say so. Disappears as soon as the final selected-basis result lands. */
function previewNote(): string {
  if (!state.report || state.reportBasis === null || state.reportBasis === state.basis) return "";
  const from = BASIS_LABEL[state.reportBasis] ?? state.reportBasis;
  const to = BASIS_LABEL[state.basis] ?? state.basis;
  return `<p class="preview-note" style="margin:6px 0 0;color:var(--dim);font-size:11px">previewing in ${esc(from)}… ${esc(to)} lands on release</p>`;
}

function card(title: string, body: string): string { return `<div class="card"><h2>${title}</h2>${body}</div>`; }
function kv(k: string, v: string): string { return `<div class="kv"><span>${k}</span><b class="num">${esc(v)}</b></div>`; }
function angleControl(): string {
  return `<div class="card"><h2>Bend it</h2>
    <div class="controls">
      <input type="range" id="angle" min="70" max="180" step="0.5" value="${angleDeg().toFixed(1)}" />
      <button class="act" id="straighten">Straighten →</button>
    </div>
    <p style="margin:8px 0 0;color:var(--dim);font-size:12px">Slide to 180° — watch the dipole arrow vanish.</p></div>`;
}

// ── Tutor (scripted preview — reads the live numbers) ────────
const TUTOR: Record<Depth, { q: string; a: (r: MolecularReport | null) => string }[]> = {
  feel: [
    { q: "why is it bent?", a: () => "Oxygen holds two hidden pairs of electrons that push the two H's down into a V — like two invisible balloons taking up room. That bend is what makes water special!" },
    { q: "why does water stick?", a: () => "Because it's a tiny magnet: a + side and a − side. The + of one water grabs the − of the next, so they all hold hands." },
  ],
  see: [
    { q: "what's a dipole?", a: (r) => `A dipole is a split of charge — a + end and a − end. The arrow points to the − end (the oxygen). Right now it's ${r ? (r.dipoleMagnitudeDebye > 0.3 ? "clearly there" : "almost gone") : "…"}.` },
    { q: "why does straightening kill it?", a: () => "When the two O–H pulls point exactly opposite (180°), they cancel — like two people pulling a rope equally. No net pull = no dipole." },
  ],
  know: [
    { q: "why 1.73 vs 1.85 D (experiment)?", a: (r) => `You're seeing ${r ? r.dipoleMagnitudeDebye.toFixed(2) : "~1.7"} D at STO-3G — a minimal basis underestimates polarity. Switch to cc-pVDZ in Prove and it moves toward experiment.` },
    { q: "what are lone pairs?", a: () => "Oxygen has 8 electrons; 4 go into O–H bonds, the other 4 sit as two 'lone pairs.' They occupy space and force the bent geometry — the reason the dipole survives." },
  ],
  prove: [
    { q: "what basis should I trust?", a: (r) => `STO-3G is a teaching basis (fast, qualitative). cc-pVDZ is the smallest basis you'd cite. Current μ = ${r ? r.dipoleMagnitudeDebye.toFixed(3) : "…"} D.` },
    { q: "is this really Hartree–Fock?", a: (r) => `Yes — a full RHF SCF ran in your tab just now. E = ${r ? r.scfEnergy.toFixed(6) : "…"} Ha, converged with DIIS. No server involved.` },
  ],
};

function tutorCard(): string {
  const chips = TUTOR[state.depth].map((t, i) => `<button class="chip" data-tq="${i}">${esc(t.q)}</button>`).join("");
  return `<div class="card tutor"><h2>Ask why <span style="opacity:.6;font-weight:400;text-transform:none;letter-spacing:0">· tutor (preview)</span></h2>
    <div class="chips">${chips}</div>
    <div class="answer" id="answer"><span class="tag">tap a question — the tutor answers at this depth, using the live numbers</span></div></div>`;
}

// ── Wiring ───────────────────────────────────────────────────
function wireSide(): void {
  side.querySelectorAll<HTMLButtonElement>("[data-tq]").forEach((b) => {
    b.addEventListener("click", () => {
      const t = TUTOR[state.depth][Number(b.dataset.tq)]!;
      const ans = side.querySelector("#answer");
      if (ans) ans.textContent = t.a(state.report);
    });
  });
  const angle = side.querySelector<HTMLInputElement>("#angle");
  if (angle) {
    angle.addEventListener("input", () => { state.atoms = geometryFromAngle(Number(angle.value), bondLen()); requestCompute("live"); syncURL(); render(); });
    // Slider release → the selected basis. (`change` fires on release;
    // the debounced final in requestCompute covers any missed release.)
    angle.addEventListener("change", () => requestCompute("final"));
  }
  const straighten = side.querySelector<HTMLButtonElement>("#straighten");
  if (straighten) straighten.addEventListener("click", animateStraighten);
  const basisSel = side.querySelector<HTMLSelectElement>("#basisSel");
  if (basisSel) basisSel.addEventListener("change", () => { state.basis = basisSel.value as BasisName; requestCompute("final"); syncURL(); render(); });
}

function animateTo(target: AtomXY[], dur: number): void {
  const start = state.atoms.map((a) => ({ ...a }));
  const t0 = performance.now();
  const step = (now: number): void => {
    const k = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    state.atoms = start.map((a, i) => ({ sym: a.sym, x: a.x + (target[i]!.x - a.x) * e, y: a.y + (target[i]!.y - a.y) * e }));
    requestCompute("live"); render();
    if (k < 1) requestAnimationFrame(step); else { requestCompute("final"); syncURL(); render(); }
  };
  requestAnimationFrame(step);
}

function animateStraighten(): void {
  animateTo(geometryFromAngle(180, bondLen()), 650);
}

// ── Dragging (pointer + touch) ───────────────────────────────
let dragging = -1;
function pointToA(evt: PointerEvent): P2 {
  const ctm = scene.getScreenCTM();
  const pt = scene.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
  const v = ctm ? pt.matrixTransform(ctm.inverse()) : { x: evt.clientX, y: evt.clientY };
  return { x: (v.x - OX) / SCALE, y: -(v.y - OY) / SCALE };
}
scene.addEventListener("pointerdown", (e) => {
  const t = e.target as SVGElement;
  if (!t.classList.contains("draggable")) return;
  dragging = Number(t.getAttribute("data-i"));
  try { scene.setPointerCapture(e.pointerId); } catch { /* synthetic/stale pointer — drag still works, just uncaptured */ }
  t.setAttribute("cursor", "grabbing");
});
scene.addEventListener("pointermove", (e) => {
  if (dragging < 0) return;
  const a = pointToA(e);
  // clamp bond length so the SCF stays sane
  const d = Math.hypot(a.x, a.y) || 1e-6;
  const r = Math.max(0.5, Math.min(1.6, d));
  state.atoms[dragging] = { sym: "H", x: a.x / d * r, y: a.y / d * r };
  requestCompute("live"); render();   // geometry renders at 60fps; numbers land when the worker replies
});
scene.addEventListener("pointerup", () => {
  if (dragging < 0) return;
  dragging = -1;
  // Feel = play mode: the molecule springs back to its comfy (equilibrium)
  // shape on release, as the copy promises — the point at that depth is
  // "molecules have a preferred shape." Deeper levels are lab mode: the
  // geometry stays where you put it so stretched states can be studied.
  if (state.depth === "feel") { animateTo(geometryFromAngle(A0, R0), 550); return; }
  requestCompute("final"); syncURL(); render();
});

// ── Depth tabs ───────────────────────────────────────────────
tabs.forEach((tab) => tab.addEventListener("click", () => setDepth(tab.dataset.depth as Depth)));
function setDepth(d: Depth): void {
  state.depth = d;
  tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.depth === d)));
  hintEl.textContent = {
    feel: "👆 drag a white atom — it springs back to its comfy shape",
    see: "slide the angle to 180° → watch the dipole arrow vanish",
    know: "drag or bend it → watch every number track the geometry",
    prove: "change the basis, or copy the URL to reproduce this exact run",
  }[d];
  syncURL(); render();   // depth switch: re-skin only, no recompute
}

// ── URL state (reproducibility-as-a-URL) ─────────────────────
function syncURL(): void {
  const g = state.atoms.slice(1).map((a) => `${a.x.toFixed(3)},${a.y.toFixed(3)}`).join(";");
  const p = new URLSearchParams({ d: state.depth, b: state.basis, g });
  history.replaceState(null, "", `?${p.toString()}`);
}
function restoreURL(): void {
  const p = new URLSearchParams(location.search);
  const d = p.get("d") as Depth | null;
  if (d && ["feel", "see", "know", "prove"].includes(d)) state.depth = d;
  const b = p.get("b"); if (b === "sto-3g" || b === "cc-pvdz") state.basis = b;
  const g = p.get("g");
  if (g) {
    const hs = g.split(";").map((s) => s.split(",").map(Number));
    if (hs.length === 2 && hs.every((h) => h.length === 2 && h.every(Number.isFinite))) {
      state.atoms = [{ sym: "O", x: 0, y: 0 }, { sym: "H", x: hs[0]![0]!, y: hs[0]![1]! }, { sym: "H", x: hs[1]![0]!, y: hs[1]![1]! }];
    }
  }
}

// ── Boot ─────────────────────────────────────────────────────
function render(): void { renderScene(); renderSide(); }
restoreURL();
tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.depth === state.depth)));
requestCompute("final");   // async: page renders immediately, numbers land when the worker replies
setDepth(state.depth);
