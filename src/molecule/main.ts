// ─────────────────────────────────────────────────────────────
// molecule/main.ts — drives the molecular SI report page.
//
// Pipeline (each step renders into its own card on completion):
//   1. Geometry optimization.
//   2. SCF + dipole + Mulliken + Mayer.
//   3. Vibrational analysis (frequencies + IR + Raman).
//   4. UV-vis (TDA singlet + triplet, oscillator strengths).
//   5. Polarizability + first hyperpolarizability β.
//   6. Ideal-gas thermochemistry at 298.15 K, 1 atm.
//   7. Ionization potential (Koopmans + ΔSCF).
// ─────────────────────────────────────────────────────────────

import type { Atom, BasisName } from "../chemistry/atoms.js";
import { moleculeToShellsNuclei } from "../chemistry/atoms.js";
import { computeMolecularIntegrals, type MolecularIntegrals } from "../chemistry/cg-molecular.js";
import { runRHFSCF } from "../chemistry/hf-scf.js";
import { runRKSDFT } from "../chemistry/dft/rks-scf.js";
import { runUHFSCF } from "../chemistry/uhf-scf.js";
import { optimizeGeometry } from "../chemistry/geometry.js";
import {
  dipoleMoment, dipoleMagnitude, mullikenCharges,
  bondOrders, mayerValences, AU_TO_DEBYE,
} from "../chemistry/properties.js";
import { ramanSpectrum } from "../chemistry/raman.js";
import { polarizabilityFiniteField, AU_TO_ANGSTROM3 } from "../chemistry/polarizability.js";
import { hyperpolarizabilityFiniteField } from "../chemistry/hyperpolarizability.js";
import { runTDA } from "../chemistry/tda-dft.js";
import {
  thermochemistry, HA_TO_KCAL_PER_MOL, HA_PER_K_TO_CAL_PER_MOL_K,
} from "../chemistry/thermochemistry.js";
import { ionizationPotential, HA_TO_EV } from "../chemistry/redox.js";
import type { FunctionalKind } from "../chemistry/dft/functional.js";
import { moleculeGraph2D, moleculeStructure } from "../viz/molecular-graph-2d.js";
import { mullikenChart } from "../viz/mulliken-chart.js";
import { uvVisChart } from "../viz/uv-vis-chart.js";
import {
  basisCoverageChart, basisSummary, type AtomBasis,
} from "../viz/basis-coverage-chart.js";

type Method = "hf" | FunctionalKind;

// ── Molecule library ───────────────────────────────────────────
interface MoleculeDef {
  readonly name: string;
  readonly symbol: string;
  readonly atoms: readonly Atom[];
  readonly basis: BasisName;
  readonly symmetryNumber: number;
  readonly linear: boolean;
  readonly closedShell: boolean;
  readonly nAlpha?: number;
  readonly nBeta?: number;
  readonly references?: {
    readonly bondLenAng?: { i: number; j: number; expt: number; label: string }[];
    readonly bondAngleDeg?: { i: number; j: number; k: number; expt: number; label: string }[];
    readonly dipoleD?: number;
    readonly ipEv?: number;
  };
}

const MOLECULES: Record<string, MoleculeDef> = {
  h2o: {
    name: "Water", symbol: "H₂O",
    atoms: (() => {
      const half = (104.52 / 2) * Math.PI / 180;
      const x = 0.9572 * Math.sin(half), z = 0.9572 * Math.cos(half);
      return [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ x, 0, z] },
        { symbol: "H", pos: [-x, 0, z] },
      ] as Atom[];
    })(),
    basis: "sto-3g", symmetryNumber: 2, linear: false, closedShell: true,
    references: {
      bondLenAng:  [{ i: 0, j: 1, expt: 0.9572, label: "O–H" }],
      bondAngleDeg:[{ i: 1, j: 0, k: 2, expt: 104.52, label: "∠HOH" }],
      dipoleD: 1.85, ipEv: 12.62,
    },
  },
  h2: {
    name: "Hydrogen", symbol: "H₂",
    atoms: [
      { symbol: "H", pos: [0, 0, -0.37] },
      { symbol: "H", pos: [0, 0,  0.37] },
    ],
    basis: "sto-3g", symmetryNumber: 2, linear: true, closedShell: true,
    references: {
      bondLenAng: [{ i: 0, j: 1, expt: 0.7414, label: "H–H" }],
      dipoleD: 0, ipEv: 15.43,
    },
  },
  lih: {
    name: "Lithium hydride", symbol: "LiH",
    atoms: [
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.5959] },
    ],
    basis: "sto-3g", symmetryNumber: 1, linear: true, closedShell: true,
    references: {
      bondLenAng: [{ i: 0, j: 1, expt: 1.5959, label: "Li–H" }],
      dipoleD: 5.88, ipEv: 7.85,
    },
  },
  beh2: {
    name: "Beryllium hydride", symbol: "BeH₂",
    atoms: [
      { symbol: "H",  pos: [0, 0, -1.291] },
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0,  1.291] },
    ],
    basis: "sto-3g", symmetryNumber: 2, linear: true, closedShell: true,
    references: {
      bondLenAng: [{ i: 0, j: 1, expt: 1.291, label: "Be–H" }],
      dipoleD: 0,
    },
  },
  li: {
    name: "Lithium atom", symbol: "Li (²S)",
    atoms: [{ symbol: "Li", pos: [0, 0, 0] }],
    basis: "sto-3g", symmetryNumber: 1, linear: false, closedShell: false,
    nAlpha: 2, nBeta: 1, references: { ipEv: 5.39 },
  },
  be: {
    name: "Beryllium atom", symbol: "Be (¹S)",
    atoms: [{ symbol: "Be", pos: [0, 0, 0] }],
    basis: "sto-3g", symmetryNumber: 1, linear: false, closedShell: true,
    references: { ipEv: 9.32 },
  },
  h: {
    name: "Hydrogen atom", symbol: "H (²S)",
    atoms: [{ symbol: "H", pos: [0, 0, 0] }],
    basis: "sto-3g", symmetryNumber: 1, linear: false, closedShell: false,
    nAlpha: 1, nBeta: 0, references: { ipEv: 13.61 },
  },
};

// ── DOM helpers ─────────────────────────────────────────────────
const $ = <T extends HTMLElement = HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const moleculeSel = $<HTMLSelectElement>("moleculeSel");
const methodSel   = $<HTMLSelectElement>("methodSel");
const runBtn      = $<HTMLButtonElement>("runBtn");
const progressEl  = $("progress");
const errEl       = $("err");
const resultsEl   = $("results");

// ── URL-as-citation contract ───────────────────────────────────
// ?molecule=h2o&method=b3lyp5&autorun=1
// Selectors initialized from URL on page load; URL kept in sync as
// dropdowns change so the address bar IS the citation.
function syncFromURL(): void {
  const params = new URLSearchParams(window.location.search);
  const m = params.get("molecule");
  const meth = params.get("method");
  if (m && MOLECULES[m]) moleculeSel.value = m;
  if (meth && Array.from(methodSel.options).some((o) => o.value === meth)) {
    methodSel.value = meth;
  }
}

function syncToURL(): void {
  const params = new URLSearchParams(window.location.search);
  params.set("molecule", moleculeSel.value);
  params.set("method", methodSel.value);
  const url = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState({}, "", url);
}

syncFromURL();
moleculeSel.addEventListener("change", syncToURL);
methodSel.addEventListener("change", syncToURL);

// ── Pipeline ───────────────────────────────────────────────────
runBtn.addEventListener("click", () => {
  syncToURL();
  void runPipeline().catch((e) =>
    showError(e instanceof Error ? e.message : String(e)),
  );
});

// Autorun on load if ?autorun=1 is present.
if (new URLSearchParams(window.location.search).get("autorun") === "1") {
  // Defer until after DOM + module init.
  queueMicrotask(() => runBtn.click());
}

// "Copy citation link" — builds the canonical &autorun=1 URL so the
// recipient sees the exact calculation re-run on open.
const citeBtn = document.getElementById("citeBtn") as HTMLButtonElement | null;
if (citeBtn) {
  citeBtn.addEventListener("click", async () => {
    const params = new URLSearchParams();
    params.set("molecule", moleculeSel.value);
    params.set("method", methodSel.value);
    params.set("autorun", "1");
    const url = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
    try {
      await navigator.clipboard.writeText(url);
      const orig = citeBtn.textContent;
      citeBtn.textContent = "Copied ✓";
      window.setTimeout(() => { citeBtn.textContent = orig; }, 1500);
    } catch {
      window.prompt("Copy this citation URL:", url);
    }
  });
}

interface Ctx {
  optAtoms: readonly Atom[];
  energy: number;
  integrals: MolecularIntegrals;
  shellAtomIdx: readonly number[];
  nElectrons: number;
  P: Float64Array;
  C_MO: Float64Array;
  orbitalEnergies: Float64Array;
  nOccupied: number;
  frequenciesCmInv?: Float64Array;
}

async function runPipeline(): Promise<void> {
  resetUI();
  runBtn.disabled = true; runBtn.textContent = "Running…";

  const mol = MOLECULES[moleculeSel.value]!;
  const method = methodSel.value as Method;
  if (!mol.closedShell && method !== "hf") {
    showError("Open-shell parents only support method='HF' for now (UHF). Pick HF and re-run.");
    runBtn.disabled = false; runBtn.textContent = "Run";
    return;
  }

  // Pre-create cards in display order.
  const cardStructure = ensureCard("structure", "Structure + basis");
  const cardEnergy   = ensureCard("energy",   "Energy + geometry");
  const cardSpectra  = ensureCard("spectra",  "IR + Raman");
  const cardUVvis    = ensureCard("uvvis",    "UV-vis (TDA)");
  const cardField    = ensureCard("field",    "Dipole · polarizability · hyperpolarizability");
  const cardCharges  = ensureCard("charges",  "Charges + bond orders");
  const cardThermo   = ensureCard("thermo",   "Thermochemistry (298.15 K, 1 atm)");
  const cardIP       = ensureCard("ip",       "Ionization potential");

  const ctx = {} as Ctx;

  await step("geom", "Optimizing geometry…", async () => {
    if (!mol.closedShell || mol.atoms.length < 2) {
      ctx.optAtoms = mol.atoms;          // single atoms have no geometry
      return;
    }
    const opt = optimizeGeometry(mol.atoms, { method, basis: mol.basis });
    ctx.optAtoms = opt.atoms;
    ctx.energy = opt.energy;
  });

  await step("scf", "Running SCF + dipole + charges + bonds…", async () => {
    const { shells, nuclei, nElectrons, shellAtomIdx } =
      moleculeToShellsNuclei(ctx.optAtoms, mol.basis);
    const integrals = computeMolecularIntegrals(shells, nuclei);
    ctx.integrals = integrals;
    ctx.shellAtomIdx = shellAtomIdx;
    ctx.nElectrons = nElectrons;
    if (mol.closedShell) {
      if (method === "hf") {
        const hf = runRHFSCF(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
        });
        ctx.energy = hf.energy; ctx.P = hf.D; ctx.C_MO = hf.C_MO;
        ctx.orbitalEnergies = hf.orbitalEnergies; ctx.nOccupied = hf.nOccupied;
      } else {
        const symbols = ctx.optAtoms.map((a) => a.symbol);
        const dft = runRKSDFT(integrals, nElectrons, symbols, {
          functional: method, energyTol: 1e-10, residualTol: 1e-7, maxIter: 200,
        });
        ctx.energy = dft.energy; ctx.P = dft.D; ctx.C_MO = dft.C_MO;
        ctx.orbitalEnergies = dft.orbitalEnergies; ctx.nOccupied = dft.nOccupied;
      }
    } else {
      const uhf = runUHFSCF(integrals, mol.nAlpha!, mol.nBeta!, {
        useDIIS: true, energyTol: 1e-10, densityTol: 1e-8, maxIter: 200,
      });
      ctx.energy = uhf.energy;
      ctx.P = new Float64Array(integrals.n * integrals.n);
      for (let k = 0; k < integrals.n * integrals.n; k++) {
        ctx.P[k] = uhf.D_alpha[k]! + uhf.D_beta[k]!;
      }
      ctx.C_MO = uhf.C_alpha; ctx.orbitalEnergies = uhf.orbitalEnergiesAlpha;
      ctx.nOccupied = mol.nAlpha!;
    }
    renderStructureCard(cardStructure, mol, ctx);
    renderEnergyCard(cardEnergy, mol, ctx, method);
    renderChargesCard(cardCharges, mol, ctx);
    renderFieldCardDipole(cardField, ctx);
  });

  await step("vib", "Vibrational frequencies + IR + Raman…", async () => {
    if (ctx.optAtoms.length < 2 || !mol.closedShell) {
      renderSkipped(cardSpectra,
        "Skipped — vibrations need ≥2 atoms and a closed-shell ground state. UHF gradients are a planned follow-up.");
      return;
    }
    const spec = ramanSpectrum(ctx.optAtoms, { method, linear: mol.linear });
    ctx.frequenciesCmInv = spec.frequenciesCmInv;
    renderSpectraCard(cardSpectra, spec, mol);
  });

  await step("uvvis", "UV-vis (TDA singlet + triplet)…", async () => {
    if (ctx.nElectrons < 2) {
      renderSkipped(cardUVvis, "Skipped — TDA needs ≥2 electrons.");
      return;
    }
    try {
      const tdaOpts = {
        method, nucleiSymbols: ctx.optAtoms.map((a) => a.symbol),
        nRoots: Math.min(5, ctx.nOccupied * (ctx.integrals.n - ctx.nOccupied)),
      } as const;
      const ks = {
        nOccupied: ctx.nOccupied, n: ctx.integrals.n,
        orbitalEnergies: ctx.orbitalEnergies, C_MO: ctx.C_MO, D: ctx.P,
      };
      const sing = runTDA(ctx.integrals, ks, { ...tdaOpts, spin: "singlet" });
      let trip: typeof sing | null = null;
      try { trip = runTDA(ctx.integrals, ks, { ...tdaOpts, spin: "triplet" }); }
      catch { /* triplet+GGA may throw; that's fine */ }
      renderUVvisCard(cardUVvis, sing, trip);
    } catch (e) {
      renderSkipped(cardUVvis,
        `Skipped — ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  await step("pol", "Polarizability + hyperpolarizability…", async () => {
    if (!mol.closedShell || ctx.optAtoms.length < 1) {
      renderFieldCardSkipPol(cardField,
        "α/β skipped — finite-field uses the closed-shell SCF path internally; UHF support is a planned follow-up.");
      return;
    }
    const pol = polarizabilityFiniteField(ctx.optAtoms, { method });
    const hyp = hyperpolarizabilityFiniteField(ctx.optAtoms, { method });
    renderFieldCardPolHyp(cardField, pol, hyp);
  });

  await step("thermo", "Thermochemistry…", async () => {
    if (!ctx.frequenciesCmInv) {
      renderSkipped(cardThermo, "Skipped — needs vibrational frequencies (≥2 atoms).");
      return;
    }
    const thermo = thermochemistry(ctx.optAtoms, ctx.frequenciesCmInv, {
      temperatureK: 298.15, pressureAtm: 1.0,
      symmetryNumber: mol.symmetryNumber, linear: mol.linear,
    });
    renderThermoCard(cardThermo, thermo);
  });

  await step("ip", "Ionization potential…", async () => {
    if (!mol.closedShell) {
      renderSkipped(cardIP,
        "Skipped — IP needs a closed-shell parent (the cation is N−1 e⁻ doublet, computed via UHF).");
      return;
    }
    if (ctx.nElectrons < 2) {
      renderSkipped(cardIP, "Skipped — IP undefined for 1-electron parents.");
      return;
    }
    try {
      const ip = ionizationPotential(ctx.optAtoms, { basis: mol.basis });
      renderIPCard(cardIP, ip, mol);
    } catch (e) {
      renderSkipped(cardIP,
        `Failed — ${e instanceof Error ? e.message : String(e)}`);
    }
  });

  runBtn.disabled = false; runBtn.textContent = "Run";
}

async function step(key: string, label: string, fn: () => Promise<void> | void): Promise<void> {
  setProgress(key, "run", label);
  try {
    await new Promise((r) => requestAnimationFrame(() => r(null)));
    await fn();
    setProgress(key, "done", label);
  } catch (e) {
    setProgress(key, "fail", `${label} — failed`);
    throw e;
  }
}

// ── UI helpers ──────────────────────────────────────────────────
function resetUI(): void {
  errEl.style.display = "none"; errEl.textContent = "";
  resultsEl.innerHTML = "";
  progressEl.classList.add("active");
  progressEl.innerHTML = "";
}
function showError(msg: string): void {
  errEl.style.display = "block"; errEl.textContent = msg;
}
function setProgress(key: string, state: "run" | "done" | "fail", label: string): void {
  let line = progressEl.querySelector<HTMLDivElement>(`[data-step="${key}"]`);
  if (!line) {
    line = document.createElement("div");
    line.dataset.step = key;
    progressEl.appendChild(line);
  }
  line.className = `line ${state}`;
  const sym = state === "done" ? "✓" : state === "fail" ? "✗" : "…";
  line.textContent = `${sym}  ${label}`;
}
function ensureCard(id: string, title: string): HTMLDivElement {
  const wrapper = document.createElement("div");
  wrapper.id = `card-${id}`;
  wrapper.className = "card";
  wrapper.innerHTML = `<h2>${title}</h2><div class="body"><p class="h-sub">computing…</p></div>`;
  resultsEl.appendChild(wrapper);
  return wrapper.querySelector<HTMLDivElement>(".body")!;
}
function renderSkipped(body: HTMLElement, message: string): void {
  body.innerHTML = `<p class="h-sub">${message}</p>`;
}

// ── Renderers ───────────────────────────────────────────────────
function renderEnergyCard(
  body: HTMLElement, mol: MoleculeDef, ctx: Ctx, method: Method,
): void {
  const html: string[] = [];
  html.push(`<table class="kv">
    <tr><td class="k">Molecule</td><td class="v">${mol.symbol} (${mol.name})</td></tr>
    <tr><td class="k">Method / basis</td><td class="v">${method.toUpperCase()} / ${mol.basis}</td></tr>
    <tr><td class="k">N basis functions</td><td class="v">${ctx.integrals.n}</td></tr>
    <tr><td class="k">N electrons</td><td class="v">${ctx.nElectrons}</td></tr>
    <tr><td class="k">Total energy</td><td class="v">${ctx.energy.toFixed(8)} Ha</td></tr>
  </table>`);
  // Bond lengths + angles vs experiment if any.
  if (mol.references?.bondLenAng?.length) {
    html.push(`<p class="h-sub" style="margin-top:14px">Optimized geometry vs experiment:</p>`);
    html.push(`<table class="t"><thead><tr><th>bond</th><th class="num">computed (Å)</th><th class="num">expt (Å)</th><th class="num">Δ</th></tr></thead><tbody>`);
    for (const b of mol.references.bondLenAng) {
      const r = bondLength(ctx.optAtoms, b.i, b.j);
      html.push(`<tr><td>${b.label}</td><td class="num">${r.toFixed(4)}</td><td class="num">${b.expt.toFixed(4)}</td><td class="num">${(r - b.expt).toFixed(4)}</td></tr>`);
    }
    html.push(`</tbody></table>`);
  }
  if (mol.references?.bondAngleDeg?.length) {
    html.push(`<table class="t" style="margin-top:8px"><thead><tr><th>angle</th><th class="num">computed (°)</th><th class="num">expt (°)</th><th class="num">Δ</th></tr></thead><tbody>`);
    for (const a of mol.references.bondAngleDeg) {
      const θ = bondAngleDeg(ctx.optAtoms, a.i, a.j, a.k);
      html.push(`<tr><td>${a.label}</td><td class="num">${θ.toFixed(2)}</td><td class="num">${a.expt.toFixed(2)}</td><td class="num">${(θ - a.expt).toFixed(2)}</td></tr>`);
    }
    html.push(`</tbody></table>`);
  }
  body.innerHTML = html.join("");
}

function renderFieldCardDipole(body: HTMLElement, ctx: Ctx): void {
  const mu  = dipoleMoment(ctx.integrals, ctx.P);
  const mag = dipoleMagnitude(mu);
  body.innerHTML = `
    <div id="field-dipole">
      <table class="kv">
        <tr><td class="k">μ_x · μ_y · μ_z (a.u.)</td><td class="v">${mu[0].toFixed(4)} · ${mu[1].toFixed(4)} · ${mu[2].toFixed(4)}</td></tr>
        <tr><td class="k">|μ| (a.u.)</td><td class="v">${mag.toFixed(4)}</td></tr>
        <tr><td class="k">|μ| (Debye)</td><td class="v">${(mag * AU_TO_DEBYE).toFixed(4)}</td></tr>
      </table>
    </div>
    <div id="field-pol"><p class="h-sub" style="margin-top:14px">computing polarizability + hyperpolarizability…</p></div>
  `;
}
function renderFieldCardSkipPol(body: HTMLElement, msg: string): void {
  const el = body.querySelector("#field-pol");
  if (el) el.innerHTML = `<p class="h-sub">${msg}</p>`;
}
function renderFieldCardPolHyp(
  body: HTMLElement,
  pol: ReturnType<typeof polarizabilityFiniteField>,
  hyp: ReturnType<typeof hyperpolarizabilityFiniteField>,
): void {
  const el = body.querySelector("#field-pol");
  if (!el) return;
  const m = pol.tensor;
  const matrix = `<div class="matrix3">
    <span class="v">${m[0]!.toFixed(3)}</span><span class="v">${m[1]!.toFixed(3)}</span><span class="v">${m[2]!.toFixed(3)}</span>
    <span class="v">${m[3]!.toFixed(3)}</span><span class="v">${m[4]!.toFixed(3)}</span><span class="v">${m[5]!.toFixed(3)}</span>
    <span class="v">${m[6]!.toFixed(3)}</span><span class="v">${m[7]!.toFixed(3)}</span><span class="v">${m[8]!.toFixed(3)}</span>
  </div>`;
  el.innerHTML = `
    <p class="h-sub" style="margin-top:14px">α tensor (a.u.):</p>${matrix}
    <table class="kv" style="margin-top:6px">
      <tr><td class="k">α_iso</td><td class="v">${pol.isotropic.toFixed(4)} a.u. = ${(pol.isotropic * AU_TO_ANGSTROM3).toFixed(4)} Å³</td></tr>
      <tr><td class="k">Δα (anisotropy)</td><td class="v">${pol.anisotropy.toFixed(4)} a.u.</td></tr>
      <tr><td class="k">|β_vec|</td><td class="v">${hyp.vectorInvariant.toFixed(4)} a.u.</td></tr>
      <tr><td class="k">β_xxx · β_yyy · β_zzz</td><td class="v">${hyp.diagonals.map((v) => v.toFixed(3)).join(" · ")} a.u.</td></tr>
    </table>
  `;
}

function renderChargesCard(body: HTMLElement, mol: MoleculeDef, ctx: Ctx): void {
  const charges = mullikenCharges(ctx.integrals, ctx.P, ctx.shellAtomIdx);
  const orders  = bondOrders(ctx.integrals, ctx.P, ctx.shellAtomIdx);
  const valences = mayerValences(ctx.integrals, ctx.P, ctx.shellAtomIdx);
  const nA = ctx.optAtoms.length;
  const html: string[] = [];
  html.push(`<p class="h-sub">Mulliken charges + Mayer bond orders. Charges sum to molecular charge by trace invariance.</p>`);
  // SVG visualization row: atom diagram with charge overlay + per-atom bar chart.
  const atomLabels = ctx.optAtoms.map((a, i) => `${a.symbol}${nA > 1 ? i + 1 : ""}`);
  const chargesArr = Array.from(charges);
  if (nA >= 1) {
    html.push(`<div class="grid2" style="margin-top:8px">`);
    html.push(`<div>${moleculeGraph2D(ctx.optAtoms.map((a) => ({ symbol: a.symbol, pos: a.pos })), {
      charges: chargesArr, atomLabels, title: `${mol.symbol} · Mulliken overlay`,
    })}</div>`);
    html.push(`<div>${mullikenChart(chargesArr, { atomLabels, title: "Per-atom q" })}</div>`);
    html.push(`</div>`);
  }
  html.push(`<table class="t" style="margin-top:14px"><thead><tr><th>atom</th><th>symbol</th><th class="num">q (e)</th><th class="num">Mayer valence</th></tr></thead><tbody>`);
  for (let i = 0; i < nA; i++) {
    html.push(`<tr><td>${i + 1}</td><td>${mol.atoms[i]!.symbol}</td><td class="num">${charges[i]!.toFixed(4)}</td><td class="num">${valences[i]!.toFixed(3)}</td></tr>`);
  }
  html.push(`</tbody></table>`);
  if (nA >= 2) {
    html.push(`<p class="h-sub" style="margin-top:14px">Bond-order matrix:</p><table class="t"><tbody>`);
    for (let i = 0; i < nA; i++) {
      const cells: string[] = [`<td>${mol.atoms[i]!.symbol}</td>`];
      for (let j = 0; j < nA; j++) cells.push(`<td class="num">${orders[i * nA + j]!.toFixed(3)}</td>`);
      html.push(`<tr>${cells.join("")}</tr>`);
    }
    html.push(`</tbody></table>`);
  }
  body.innerHTML = html.join("");
}

function renderStructureCard(body: HTMLElement, mol: MoleculeDef, ctx: Ctx): void {
  // Aggregate per-atom shell counts from the shell list. Each shell has L; a
  // spherical-harmonic shell contributes 2L+1 basis functions.
  const nA = ctx.optAtoms.length;
  const perAtomCounts: number[][] = Array.from({ length: nA }, () => [0, 0, 0, 0, 0]);
  for (let s = 0; s < ctx.integrals.shells.length; s++) {
    const shell = ctx.integrals.shells[s]!;
    const atomIdx = ctx.shellAtomIdx[s] ?? 0;
    const L = shell.angular[0] + shell.angular[1] + shell.angular[2];
    if (L >= 0 && L < 5) {
      perAtomCounts[atomIdx]![L] = (perAtomCounts[atomIdx]![L] ?? 0) + 1;
    }
  }
  const atomBases: AtomBasis[] = ctx.optAtoms.map((a, i) => ({
    symbol: a.symbol, counts: perAtomCounts[i] ?? [],
  }));
  const html: string[] = [];
  html.push(`<p class="h-sub">2D projection of the optimized geometry and the per-atom basis-set composition. Bonds detected by a 1.8 Å covalent cutoff.</p>`);
  html.push(`<div class="grid2" style="margin-top:8px">`);
  html.push(`<div>${moleculeStructure(ctx.optAtoms.map((a) => ({ symbol: a.symbol, pos: a.pos })), {
    title: `${mol.symbol} · ${mol.name}`,
  })}</div>`);
  html.push(`<div>${basisCoverageChart(atomBases, {
    title: `${mol.basis.toUpperCase()} coverage`,
    subtitle: `${ctx.integrals.n} basis functions · ${ctx.nElectrons} electrons`,
  })}</div>`);
  html.push(`</div>`);
  // Per-atom basis summary as a one-line caption.
  const perAtomSummary = atomBases
    .map((a, i) => `${a.symbol}${nA > 1 ? i + 1 : ""}: ${basisSummary(a.counts)}`)
    .join("   ·   ");
  html.push(`<p class="h-sub" style="margin-top:8px;font-family:ui-monospace,monospace">${perAtomSummary}</p>`);
  body.innerHTML = html.join("");
}

function renderSpectraCard(
  body: HTMLElement, spec: ReturnType<typeof ramanSpectrum>, mol: MoleculeDef,
): void {
  const html: string[] = [];
  html.push(`<p class="h-sub">${spec.frequenciesCmInv.length} vibrational modes (after projecting out ${mol.linear ? "5 (3T+2R, linear)" : "6 (3T+3R)"} zero modes). ${spec.nScfRuns} SCF runs total in ${spec.seconds.toFixed(1)} s.</p>`);
  html.push(`<table class="t"><thead><tr><th>mode</th><th class="num">freq (cm⁻¹)</th><th class="num">IR (km/mol)</th><th class="num">Raman (Å⁴/amu)</th></tr></thead><tbody>`);
  for (let k = 0; k < spec.frequenciesCmInv.length; k++) {
    html.push(`<tr><td>${k + 1}</td><td class="num">${spec.frequenciesCmInv[k]!.toFixed(1)}</td><td class="num">${spec.irIntensitiesKmPerMol[k]!.toFixed(2)}</td><td class="num">${spec.ramanActivitiesA4PerAmu[k]!.toFixed(2)}</td></tr>`);
  }
  html.push(`</tbody></table>`);
  html.push(`<div class="grid2" style="margin-top:18px">
    <div><p class="h-sub">IR spectrum (km/mol)</p><canvas id="canvas-ir" class="spec"></canvas></div>
    <div><p class="h-sub">Raman spectrum (Å⁴/amu)</p><canvas id="canvas-raman" class="spec"></canvas></div>
  </div>`);
  body.innerHTML = html.join("");
  drawSpectrum(document.getElementById("canvas-ir") as HTMLCanvasElement,
    spec.frequenciesCmInv, spec.irIntensitiesKmPerMol, "#6ea8ff");
  drawSpectrum(document.getElementById("canvas-raman") as HTMLCanvasElement,
    spec.frequenciesCmInv, spec.ramanActivitiesA4PerAmu, "#b0ffd0");
}

function renderUVvisCard(
  body: HTMLElement,
  sing: ReturnType<typeof runTDA>,
  trip: ReturnType<typeof runTDA> | null,
): void {
  const html: string[] = [];
  html.push(`<p class="h-sub">TDA excitation energies. Triplet oscillator strengths are zero from a closed-shell ground state (ΔS=1, dipole-forbidden).</p>`);
  html.push(`<div class="grid2"><div><p class="h-sub">Singlet</p><table class="t"><thead><tr><th>n</th><th class="num">ω (Ha)</th><th class="num">ω (eV)</th><th class="num">f</th></tr></thead><tbody>`);
  for (let r = 0; r < sing.singletEnergies.length; r++) {
    html.push(`<tr><td>S${r + 1}</td><td class="num">${sing.singletEnergies[r]!.toFixed(4)}</td><td class="num">${(sing.singletEnergies[r]! * HA_TO_EV).toFixed(3)}</td><td class="num">${sing.oscillatorStrengths[r]!.toFixed(4)}</td></tr>`);
  }
  html.push(`</tbody></table></div>`);
  if (trip) {
    html.push(`<div><p class="h-sub">Triplet</p><table class="t"><thead><tr><th>n</th><th class="num">ω (Ha)</th><th class="num">ω (eV)</th></tr></thead><tbody>`);
    for (let r = 0; r < trip.singletEnergies.length; r++) {
      html.push(`<tr><td>T${r + 1}</td><td class="num">${trip.singletEnergies[r]!.toFixed(4)}</td><td class="num">${(trip.singletEnergies[r]! * HA_TO_EV).toFixed(3)}</td></tr>`);
    }
    html.push(`</tbody></table></div>`);
  } else {
    html.push(`<div><p class="h-sub">Triplet</p><p class="h-sub">unavailable for this method × spin combo</p></div>`);
  }
  html.push(`</div>`);
  html.push(`<p class="h-sub" style="margin-top:18px">UV-vis spectrum — sticks (canvas) and Gaussian-broadened (SVG, σ ≈ 0.3 eV).</p><canvas id="canvas-uvvis" class="spec"></canvas>`);
  // Gaussian-broadened SVG version next to it.
  const energiesEv = Array.from(sing.singletEnergies).map((x) => x * HA_TO_EV);
  const strengths = Array.from(sing.oscillatorStrengths);
  const peaks = energiesEv.map((e, i) => ({ energy: e, absorption: strengths[i] ?? 0 }));
  const broadened = gaussianBroaden(energiesEv, strengths, 0.30, 80);
  html.push(`<div style="margin-top:8px">${uvVisChart(broadened, peaks, {
    title: "Singlet absorption (TDA)",
  })}</div>`);
  body.innerHTML = html.join("");
  const omegaEv = new Float64Array(energiesEv);
  drawSpectrum(document.getElementById("canvas-uvvis") as HTMLCanvasElement,
    omegaEv, sing.oscillatorStrengths, "#ffb56e", "eV");
}

function gaussianBroaden(
  energies: readonly number[], strengths: readonly number[],
  sigma: number, nGrid: number,
): Array<{ energy: number; absorption: number }> {
  if (energies.length === 0) return [];
  const eMin = Math.max(0, Math.min(...energies) - 3 * sigma);
  const eMax = Math.max(...energies) + 3 * sigma;
  const step = (eMax - eMin) / Math.max(nGrid - 1, 1);
  const norm = 1 / (sigma * Math.sqrt(2 * Math.PI));
  const out: Array<{ energy: number; absorption: number }> = [];
  for (let i = 0; i < nGrid; i++) {
    const e = eMin + i * step;
    let s = 0;
    for (let k = 0; k < energies.length; k++) {
      const dx = (e - energies[k]!) / sigma;
      s += (strengths[k] ?? 0) * norm * Math.exp(-0.5 * dx * dx);
    }
    out.push({ energy: e, absorption: s });
  }
  return out;
}

function renderThermoCard(
  body: HTMLElement, thermo: ReturnType<typeof thermochemistry>,
): void {
  body.innerHTML = `<table class="kv">
    <tr><td class="k">ZPE</td><td class="v">${(thermo.zpe * HA_TO_KCAL_PER_MOL).toFixed(3)} kcal/mol</td></tr>
    <tr><td class="k">Thermal U</td><td class="v">${(thermo.thermalU * HA_TO_KCAL_PER_MOL).toFixed(3)} kcal/mol</td></tr>
    <tr><td class="k">Thermal H</td><td class="v">${(thermo.thermalH * HA_TO_KCAL_PER_MOL).toFixed(3)} kcal/mol</td></tr>
    <tr><td class="k">T·S</td><td class="v">${(thermo.temperatureK * thermo.entropy * HA_TO_KCAL_PER_MOL).toFixed(3)} kcal/mol</td></tr>
    <tr><td class="k">G correction</td><td class="v">${(thermo.gibbsCorrection * HA_TO_KCAL_PER_MOL).toFixed(3)} kcal/mol</td></tr>
    <tr><td class="k">Total entropy</td><td class="v">${(thermo.entropy * HA_PER_K_TO_CAL_PER_MOL_K).toFixed(3)} cal/(mol·K)</td></tr>
  </table>
  <p class="h-sub" style="margin-top:14px">Per-component entropy (cal/(mol·K)):</p>
  <table class="kv">
    <tr><td class="k">S_translational (Sackur-Tetrode)</td><td class="v">${(thermo.breakdown.translationalS * HA_PER_K_TO_CAL_PER_MOL_K).toFixed(3)}</td></tr>
    <tr><td class="k">S_rotational (rigid rotor)</td><td class="v">${(thermo.breakdown.rotationalS * HA_PER_K_TO_CAL_PER_MOL_K).toFixed(3)}</td></tr>
    <tr><td class="k">S_vibrational (harmonic)</td><td class="v">${(thermo.breakdown.vibrationalS * HA_PER_K_TO_CAL_PER_MOL_K).toFixed(3)}</td></tr>
    <tr><td class="k">S_electronic</td><td class="v">${(thermo.breakdown.electronicS * HA_PER_K_TO_CAL_PER_MOL_K).toFixed(3)}</td></tr>
  </table>`;
}

function renderIPCard(
  body: HTMLElement, ip: ReturnType<typeof ionizationPotential>, mol: MoleculeDef,
): void {
  const refIp = mol.references?.ipEv;
  body.innerHTML = `<table class="kv">
    <tr><td class="k">Koopmans IP (= −ε_HOMO)</td><td class="v">${ip.koopmansEv.toFixed(3)} eV<span class="badge">${ip.koopmansHa.toFixed(5)} Ha</span></td></tr>
    <tr><td class="k">ΔSCF IP (= E_cation − E_neutral)</td><td class="v">${ip.deltaScfEv.toFixed(3)} eV<span class="badge">${ip.deltaScfHa.toFixed(5)} Ha</span></td></tr>
    <tr><td class="k">Cation ⟨S²⟩</td><td class="v">${ip.cationS2.toFixed(4)} (exact 0.75 = doublet)</td></tr>
    ${refIp !== undefined ? `<tr><td class="k">Experimental IP</td><td class="v">${refIp.toFixed(2)} eV</td></tr>` : ""}
  </table>`;
}

// ── Geometry helpers ────────────────────────────────────────────
function bondLength(atoms: readonly Atom[], i: number, j: number): number {
  const a = atoms[i]!.pos, b = atoms[j]!.pos;
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
function bondAngleDeg(atoms: readonly Atom[], i: number, j: number, k: number): number {
  const a = atoms[i]!.pos, b = atoms[j]!.pos, c = atoms[k]!.pos;
  const v1: [number, number, number] = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const v2: [number, number, number] = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
  const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  const n1  = Math.hypot(...v1), n2 = Math.hypot(...v2);
  return Math.acos(dot / (n1 * n2)) * 180 / Math.PI;
}

// ── Spectrum drawer ─────────────────────────────────────────────
function drawSpectrum(
  canvas: HTMLCanvasElement,
  freqs: Float64Array, intensities: Float64Array,
  color: string, xLabel: string = "cm⁻¹",
): void {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr); ctx.clearRect(0, 0, cssW, cssH);
  if (freqs.length === 0) return;
  const padL = 44, padR = 12, padT = 8, padB = 28;
  const plotW = cssW - padL - padR, plotH = cssH - padT - padB;
  const xMin = 0, xMax = Math.max(...freqs) * 1.1 || 1;
  const yMax = Math.max(...intensities) * 1.05 || 1;
  ctx.strokeStyle = "#2a2f40"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.fillStyle = color;
  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i]!, v = intensities[i]!;
    const x = padL + ((f - xMin) / (xMax - xMin)) * plotW;
    const h = Math.max(0, (v / yMax) * plotH);
    ctx.fillRect(x - 1.5, padT + plotH - h, 3, h);
    ctx.beginPath(); ctx.arc(x, padT + plotH - h, 3, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.fillStyle = "#99a0b5";
  ctx.font = "11px ui-monospace, SFMono-Regular, monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const t = i / 4, x = padL + t * plotW, val = xMin + t * (xMax - xMin);
    ctx.fillText(val.toFixed(0), x, padT + plotH + 6);
  }
  ctx.textAlign = "right";
  ctx.fillText(xLabel, padL + plotW, padT + plotH + 6);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.fillText(yMax.toFixed(2), padL - 6, padT + 6);
  ctx.fillText("0", padL - 6, padT + plotH);
}
