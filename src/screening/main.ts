// /screening.html entry — a LIVE molecular screen that runs real Hartree–Fock
// in the browser tab and ranks an isoelectronic aza-chain library by HOMO–LUMO
// gap (the property that sets color / reactivity / stability).
//
// Nothing here is canned: every gap is produced by `runChemEnergyTile` — the
// same kernel the swarm distributes, the same SCF that's cross-checked against
// PySCF/FCI in CI. The page just narrates the screen as it streams in:
//   • each molecule drawn as a skeletal structure (N highlighted, azo N=N glowing)
//   • a relative red-shift color ramp (honest: HF/STO-3G overestimates absolute λ,
//     so we color by RANK within the library, where the trend is the robust part)
//   • a self-sorting leaderboard, an azo-enrichment meter, a live compute log
//   • honest flagging of anomalously small gaps as likely RHF instabilities.
//
// Geometry generator is the same `azaChain` used by the e2e screening specs.

import { runChemEnergyTile, type ChemEnergyResult } from "../swarm/chemistry-kernel.js";
import type { Atom } from "../chemistry/atoms.js";

// ─────────────────────────────────────────────────────────────────────────
// Geometry — all-trans conjugated 6-atom chain; carbons at positions in `nSet`
// become nitrogen. Terminal N keeps 1 H, interior N keeps 0 (electron-neutral
// C↔N swap → every isomer is the same closed-shell species). First K entries
// are the backbone; H atoms are appended after.
const K = 6;
const norm = (v: number[]): number[] => { const n = Math.hypot(v[0]!, v[1]!, v[2]!); return [v[0]! / n, v[1]! / n, v[2]! / n]; };

function azaChain(nSet: ReadonlySet<number>, k = K): Atom[] {
  const dDouble = 1.34, dSingle = 1.46, dXH = 1.05, aHalf = (28 * Math.PI) / 180;
  const P: [number, number, number][] = [[0, 0, 0]];
  for (let i = 0; i < k - 1; i++) {
    const theta = i % 2 === 0 ? aHalf : -aHalf;
    const d = i % 2 === 0 ? dDouble : dSingle;
    const p = P[i]!;
    P.push([p[0] + d * Math.cos(theta), p[1] + d * Math.sin(theta), 0]);
  }
  const atoms: Atom[] = P.map((p, i) => ({ symbol: nSet.has(i) ? "N" : "C", pos: p }));
  for (let i = 0; i < k; i++) {
    const ci = P[i]!, isN = nSet.has(i), terminal = i === 0 || i === k - 1;
    const nH = terminal ? (isN ? 1 : 2) : (isN ? 0 : 1);
    if (nH === 0) continue;
    if (terminal) {
      const nb = i === 0 ? P[1]! : P[k - 2]!;
      const u = norm([nb[0] - ci[0], nb[1] - ci[1], 0]);
      for (const s of nH === 2 ? [1, -1] : [1]) {
        const phi = (s * 120 * Math.PI) / 180;
        atoms.push({ symbol: "H", pos: [ci[0] + dXH * (u[0]! * Math.cos(phi) - u[1]! * Math.sin(phi)), ci[1] + dXH * (u[0]! * Math.sin(phi) + u[1]! * Math.cos(phi)), 0] });
      }
    } else {
      const a = P[i - 1]!, b = P[i + 1]!;
      const ua = norm([a[0] - ci[0], a[1] - ci[1], 0]), ub = norm([b[0] - ci[0], b[1] - ci[1], 0]);
      const bis = norm([ua[0]! + ub[0]!, ua[1]! + ub[1]!, 0]);
      atoms.push({ symbol: "H", pos: [ci[0] - dXH * bis[0]!, ci[1] - dXH * bis[1]!, 0] });
    }
  }
  return atoms;
}

const popcount = (m: number): number => { let c = 0, x = m; while (x) { c += x & 1; x >>= 1; } return c; };
const labelOf = (nAt: number[]): string => (nAt.length ? nAt.map((i) => i + 1).join(",") + "-aza" : "all-C (hexatriene)");
// Azo = an N=N double bond: backbone double bonds sit on even bond indices.
const hasAzo = (s: ReadonlySet<number>): boolean => { for (let i = 0; i < K - 1; i += 2) if (s.has(i) && s.has(i + 1)) return true; return false; };

interface Cand { id: string; label: string; nSet: Set<number>; azo: boolean; atoms: Atom[]; }
function candFromMask(m: number): Cand {
  const nSet = new Set<number>();
  for (let b = 0; b < K; b++) if (m & (1 << b)) nSet.add(b);
  return { id: `m${m}`, label: labelOf([...nSet]), nSet, azo: hasAzo(nSet), atoms: azaChain(nSet) };
}

const CURATED_MASKS = [0, 0b1, 0b10, 0b100, 0b110, 0b1100, 0b11000, 0b1001, 0b100001, 0b1010, 0b101, 0b10010];
const curatedLib = (): Cand[] => CURATED_MASKS.map(candFromMask);
const fullLib = (): Cand[] => Array.from({ length: 1 << K }, (_, m) => m).filter((m) => popcount(m) <= 3).map(candFromMask);

// ─────────────────────────────────────────────────────────────────────────
// Skeletal structure → SVG. Draws the backbone (alternating double/single
// bonds), labels N atoms, glows the azo N=N bond. H atoms omitted (skeletal
// convention). Carbons are implicit vertices.
function structSVG(c: Cand): string {
  const bb = c.atoms.slice(0, K);
  const xs = bb.map((a) => a.pos[0]), ys = bb.map((a) => a.pos[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const W = 170, H = 70, pad = 16;
  const sx = (maxX - minX) > 1e-6 ? (W - 2 * pad) / (maxX - minX) : 1;
  const sy = (maxY - minY) > 1e-6 ? (H - 2 * pad) / (maxY - minY) : 1;
  const s = Math.min(sx, sy);
  const px = (x: number): number => pad + (x - minX) * s + (W - 2 * pad - (maxX - minX) * s) / 2;
  const py = (y: number): number => H - (pad + (y - minY) * s + (H - 2 * pad - (maxY - minY) * s) / 2);
  const pts = bb.map((a) => [px(a.pos[0]), py(a.pos[1])] as [number, number]);

  let bonds = "";
  for (let i = 0; i < K - 1; i++) {
    const [x1, y1] = pts[i]!, [x2, y2] = pts[i + 1]!;
    const dbl = i % 2 === 0;
    const azoBond = dbl && c.nSet.has(i) && c.nSet.has(i + 1);
    const col = azoBond ? "#f472b6" : "#7e8aa6";
    const wdt = azoBond ? 2.4 : 1.6;
    if (dbl) {
      // offset the two parallel lines perpendicular to the bond
      const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1, ox = (-dy / L) * 2.1, oy = (dx / L) * 2.1;
      bonds += `<line x1="${(x1 + ox).toFixed(1)}" y1="${(y1 + oy).toFixed(1)}" x2="${(x2 + ox).toFixed(1)}" y2="${(y2 + oy).toFixed(1)}" stroke="${col}" stroke-width="${wdt}"/>`;
      bonds += `<line x1="${(x1 - ox).toFixed(1)}" y1="${(y1 - oy).toFixed(1)}" x2="${(x2 - ox).toFixed(1)}" y2="${(y2 - oy).toFixed(1)}" stroke="${col}" stroke-width="${wdt}"/>`;
      if (azoBond) bonds += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="#f472b6" stroke-width="6" opacity="0.18"/>`;
    } else {
      bonds += `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${col}" stroke-width="${wdt}"/>`;
    }
  }
  let nodes = "";
  for (let i = 0; i < K; i++) {
    const [x, y] = pts[i]!;
    if (c.nSet.has(i)) {
      nodes += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="7.5" fill="#0f1422" stroke="#6ea8ff" stroke-width="1.6"/>`;
      nodes += `<text x="${x.toFixed(1)}" y="${(y + 3.2).toFixed(1)}" text-anchor="middle" font-size="9" font-weight="700" fill="#9cc2ff" font-family="ui-monospace,monospace">N</text>`;
    } else {
      nodes += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2" fill="#8190ad"/>`;
    }
  }
  return `<svg class="struct" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${bonds}${nodes}</svg>`;
}

// Relative red-shift color: smallest gap (t=0) → red, largest (t=1) → violet.
function rampColor(t: number): string {
  const h = Math.max(0, Math.min(1, t)) * 285; // 0=red … 285=violet
  return `hsl(${h.toFixed(0)} 78% 58%)`;
}

// ─────────────────────────────────────────────────────────────────────────
interface Row extends Cand { gap: number; res: ChemEnergyResult | null; suspect: boolean; done: boolean; }

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const grid = $("grid"), spectrum = $("spectrum"), ticker = $<HTMLPreElement>("ticker");
const runBtn = $<HTMLButtonElement>("run"), hint = $("hint");
const kDone = $("kDone"), kGap = $("kGap"), kEnrich = $("kEnrich"), kTime = $("kTime");
const enrichTop = $("enrichTop"), enrichAll = $("enrichAll"), enrichTopTxt = $("enrichTopTxt"), enrichAllTxt = $("enrichAllTxt"), enrichVerdict = $("enrichVerdict");
const callout = $("callout"), calloutText = $("calloutText");

let libMode: "curated" | "full" = "curated";
$("libSel").querySelectorAll<HTMLButtonElement>("button").forEach((b) => {
  b.addEventListener("click", () => {
    if (runBtn.disabled) return;
    $("libSel").querySelectorAll("button").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    libMode = b.dataset.lib as "curated" | "full";
    hint.textContent = libMode === "full" ? "~3–6 min · 42 isomers · STO-3G · all on your machine" : "~30–60 s · 12 real SCF runs · STO-3G · everything computed on your machine";
  });
});

function tick(html: string): void {
  const atBottom = ticker.scrollHeight - ticker.scrollTop - ticker.clientHeight < 30;
  ticker.insertAdjacentHTML("beforeend", "\n" + html);
  if (atBottom) ticker.scrollTop = ticker.scrollHeight;
}
const yieldFrame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));
const median = (xs: number[]): number => { if (!xs.length) return NaN; const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2; };

function renderCard(r: Row): void {
  let el = document.getElementById(r.id);
  if (!el) {
    el = document.createElement("div");
    el.id = r.id;
    grid.appendChild(el);
  }
  const azoBadge = r.azo ? `<span class="badge azo">azo</span>` : "";
  const suspectBadge = r.suspect ? ` <span class="badge suspect">⚠ suspect</span>` : "";
  el.className = `mol${r.azo ? " azo" : ""}${r.done ? " done" : ""}`;
  if (!r.done) {
    el.innerHTML = `<div class="rank">·</div><div class="name">${r.label} ${azoBadge}</div>${structSVG(r)}<div class="gap" style="color:var(--faint)">computing…</div><div class="meta">SCF running</div>`;
    return;
  }
  if (r.res && !Number.isFinite(r.gap)) {
    el.innerHTML = `<div class="rank">—</div><div class="name">${r.label}</div>${structSVG(r)}<div class="gap" style="color:var(--err)">no gap</div><div class="meta">${r.res.converged ? "no virtual orbital" : "did not converge"}</div>`;
    return;
  }
  const res = r.res!;
  el.innerHTML =
    `<div class="rank" id="${r.id}-rank">·</div>` +
    `<div class="name">${r.label} ${azoBadge}${suspectBadge}</div>` +
    structSVG(r) +
    `<div class="gap" id="${r.id}-gap"><span class="swatch" id="${r.id}-sw"></span>${r.gap.toFixed(2)} <span style="font-size:11px;color:var(--dim)">eV</span></div>` +
    `<div class="meta">HF · ${res.iter} iter · ${res.engine} · ${res.durationMs.toFixed(0)} ms</div>`;
}

function recolorAndRank(rows: Row[]): void {
  const finished = rows.filter((r) => r.done && Number.isFinite(r.gap));
  if (!finished.length) return;
  const gaps = finished.map((r) => r.gap);
  const gMin = Math.min(...gaps), gMax = Math.max(...gaps), span = gMax - gMin || 1;
  const sorted = [...finished].sort((a, b) => a.gap - b.gap);
  sorted.forEach((r, i) => {
    const t = (r.gap - gMin) / span;
    const col = rampColor(t);
    const card = document.getElementById(r.id);
    if (card) card.style.order = String(i);
    const rankEl = document.getElementById(`${r.id}-rank`); if (rankEl) rankEl.textContent = `#${i + 1}`;
    const sw = document.getElementById(`${r.id}-sw`); if (sw) sw.style.background = col;
    // spectrum chip
    let chip = document.getElementById(`${r.id}-chip`);
    if (!chip) {
      chip = document.createElement("div");
      chip.id = `${r.id}-chip`;
      chip.className = `chip${r.azo ? " azo" : ""}${r.suspect ? " suspect" : ""}`;
      chip.title = `${r.label}: ${r.gap.toFixed(2)} eV`;
      spectrum.appendChild(chip);
    }
    chip.className = `chip${r.azo ? " azo" : ""}${r.suspect ? " suspect" : ""}`;
    chip.style.left = `${(6 + t * 88).toFixed(1)}%`;
    chip.style.background = col;
  });
  kGap.textContent = `${gMin.toFixed(1)} – ${gMax.toFixed(1)}`;
}

function updateEnrichment(rows: Row[]): void {
  const fin = rows.filter((r) => r.done && Number.isFinite(r.gap)).sort((a, b) => a.gap - b.gap);
  if (fin.length < 3) return;
  const TOP = Math.min(15, Math.max(3, Math.round(fin.length * 0.35)));
  const top = fin.slice(0, TOP);
  const azoTop = top.filter((r) => r.azo).length / top.length;
  const azoAll = fin.filter((r) => r.azo).length / fin.length;
  enrichTop.style.width = `${(azoTop * 100).toFixed(0)}%`;
  enrichAll.style.width = `${(azoAll * 100).toFixed(0)}%`;
  enrichTopTxt.textContent = `${(azoTop * 100).toFixed(0)}% (top ${TOP})`;
  enrichAllTxt.textContent = `${(azoAll * 100).toFixed(0)}%`;
  const enr = azoAll > 0 ? azoTop / azoAll : 0;
  kEnrich.textContent = enr > 0 ? `${enr.toFixed(1)}×` : "—";
  enrichVerdict.innerHTML = enr > 1.05
    ? `Azo is <b style="color:var(--azo)">${enr.toFixed(1)}× enriched</b> at the top — the descriptor tracks the real dye motif, not noise.`
    : `Azo is not enriched here — with this few molecules the signal is thin; run the full sweep.`;
}

function flagArtifacts(rows: Row[]): void {
  const fin = rows.filter((r) => r.done && Number.isFinite(r.gap));
  if (fin.length < 4) return;
  const med = median(fin.map((r) => r.gap));
  for (const r of fin) r.suspect = r.gap < 0.55 * med; // anomalously small gap → likely RHF instability
  const suspects = fin.filter((r) => r.suspect).sort((a, b) => a.gap - b.gap);
  for (const r of fin) renderCard(r);
  recolorAndRank(rows);
  if (suspects.length) {
    callout.style.display = "block";
    const names = suspects.slice(0, 3).map((r) => `<b>${r.label}</b> (${r.gap.toFixed(1)} eV)`).join(", ");
    calloutText.innerHTML =
      `The smallest-gap hits — ${names} — sit far below the pack (under 55% of the median). ` +
      `For minimal-basis Hartree–Fock that's the classic fingerprint of an <b>RHF instability</b> ` +
      `(near-diradical character — the restricted wavefunction breaking down), <i>not</i> a real lead. ` +
      `A trustworthy screen surfaces both the leads <i>and</i> the traps — you'd re-check these with ` +
      `UHF / a multireference method before believing them. That skepticism is the feature, not a caveat.`;
  }
}

async function run(): Promise<void> {
  runBtn.disabled = true;
  callout.style.display = "none";
  grid.innerHTML = "";
  spectrum.querySelectorAll(".chip").forEach((c) => c.remove());
  ticker.textContent = `starting screen — ${libMode} library · STO-3G · real HF SCF per molecule`;

  const lib = libMode === "full" ? fullLib() : curatedLib();
  const rows: Row[] = lib.map((c) => ({ ...c, gap: NaN, res: null, suspect: false, done: false }));
  kDone.innerHTML = `0<span style="font-size:15px;color:var(--dim)">/${rows.length}</span>`;
  kEnrich.textContent = kTime.textContent = "—"; kGap.textContent = "—";
  rows.forEach(renderCard); // show all cards immediately (queued state)
  await yieldFrame();

  const t0 = performance.now();
  let done = 0;
  for (const r of rows) {
    const card = document.getElementById(r.id);
    if (card) card.classList.add("computing");
    await yieldFrame(); // let the "computing" sweep paint before the SCF blocks the thread
    try {
      const res = await runChemEnergyTile({ label: r.label, atoms: r.atoms, basis: "sto-3g", method: "hf" });
      r.res = res; r.gap = res.homoLumoGapEv; r.done = true;
      if (Number.isFinite(r.gap)) {
        tick(`<b>✓</b> ${r.label.padEnd(20).slice(0, 20)} gap ${r.gap.toFixed(2)} eV · ${res.iter} iter · ${res.engine} · ${res.durationMs.toFixed(0)} ms${r.azo ? ` <span class="a">[azo]</span>` : ""}`);
      } else {
        tick(`<span class="w">~</span> ${r.label} — no usable gap (${res.converged ? "no virtual" : "not converged"})`);
      }
    } catch (e) {
      r.done = true; r.gap = NaN;
      tick(`<span class="w">✗</span> ${r.label} — ${(e as Error).message.slice(0, 50)}`);
    }
    if (card) card.classList.remove("computing");
    renderCard(r);
    recolorAndRank(rows);
    updateEnrichment(rows);
    done++;
    kDone.innerHTML = `${done}<span style="font-size:15px;color:var(--dim)">/${rows.length}</span>`;
    kTime.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
    await yieldFrame();
  }

  flagArtifacts(rows);
  updateEnrichment(rows);
  const elapsed = (performance.now() - t0) / 1000;
  kTime.textContent = `${elapsed.toFixed(1)} s`;
  const fin = rows.filter((r) => r.done && Number.isFinite(r.gap)).sort((a, b) => a.gap - b.gap);
  tick(`\n<b>done</b> — ${fin.length}/${rows.length} converged in ${elapsed.toFixed(1)} s. winner (smallest gap): ${fin[0]?.label ?? "—"} ${fin[0] ? fin[0].gap.toFixed(2) + " eV" : ""}`);
  runBtn.disabled = false;
  runBtn.textContent = "▶ Run again";
}

runBtn.addEventListener("click", () => { void run(); });
