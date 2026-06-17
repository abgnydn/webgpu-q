// Landing hero centerpiece: a REAL Hartree–Fock SCF that runs in the visitor's
// browser, visualized live — the per-iteration energy converging to the ground
// state, plus the molecule's real MO energy levels around the HOMO–LUMO gap.
// The chemistry stack is lazy-imported on first use; it auto-runs once on load.

import { initScfViz } from "./scfviz.js";
import { initField } from "./field.js";
import { initReveal, initScrollRail } from "./reveal.js";

const fieldEl = document.getElementById("field");
if (fieldEl instanceof HTMLCanvasElement) initField(fieldEl);
initReveal();
initScrollRail();

type Atom = { symbol: string; pos: readonly [number, number, number] };

const MOLS: Record<string, { name: string; atoms: Atom[] }> = {
  benzene: { name: "benzene", atoms: [
    { symbol: "C", pos: [1.39, 0, 0] }, { symbol: "C", pos: [0.695, 1.2036, 0] },
    { symbol: "C", pos: [-0.695, 1.2036, 0] }, { symbol: "C", pos: [-1.39, 0, 0] },
    { symbol: "C", pos: [-0.695, -1.2036, 0] }, { symbol: "C", pos: [0.695, -1.2036, 0] },
    { symbol: "H", pos: [2.48, 0, 0] }, { symbol: "H", pos: [1.24, 2.1477, 0] },
    { symbol: "H", pos: [-1.24, 2.1477, 0] }, { symbol: "H", pos: [-2.48, 0, 0] },
    { symbol: "H", pos: [-1.24, -2.1477, 0] }, { symbol: "H", pos: [1.24, -2.1477, 0] },
  ] },
  h2o: { name: "H₂O", atoms: [
    { symbol: "O", pos: [0, 0, 0.1173] },
    { symbol: "H", pos: [0, 0.7572, -0.4692] },
    { symbol: "H", pos: [0, -0.7572, -0.4692] },
  ] },
  ch4: { name: "CH₄", atoms: [
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "H", pos: [0.6276, 0.6276, 0.6276] },
    { symbol: "H", pos: [-0.6276, -0.6276, 0.6276] },
    { symbol: "H", pos: [0.6276, -0.6276, -0.6276] },
    { symbol: "H", pos: [-0.6276, 0.6276, -0.6276] },
  ] },
};

const plotEl = document.getElementById("scfPlot");
const ladderEl = document.getElementById("scfLadder");
const viz = plotEl instanceof HTMLCanvasElement && ladderEl instanceof HTMLCanvasElement
  ? initScfViz(plotEl, ladderEl) : null;

const out = document.getElementById("liveOut");
const run = document.getElementById("liveRun") as HTMLButtonElement | null;
const eng = document.getElementById("liveEng");
const molWrap = document.getElementById("liveMol");
let sel = "benzene";
let busy = false;

const show = (html: string): void => { if (out) out.innerHTML = html; };
const raf = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));

molWrap?.querySelectorAll<HTMLButtonElement>(".molbtn").forEach((b) => {
  b.addEventListener("click", () => {
    if (busy) return;
    sel = b.dataset.mol ?? "benzene";
    molWrap.querySelectorAll(".molbtn").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    void runHF();
  });
});

async function runHF(): Promise<void> {
  if (busy) return;
  busy = true;
  const mol = MOLS[sel] ?? MOLS.benzene!;
  if (run) run.disabled = true;
  show(`<span class="dim">loading the chemistry engine…</span>`);
  try {
    const [{ moleculeToShellsNuclei }, { runRHFAuto }] = await Promise.all([
      import("../chemistry/atoms.js"),
      import("../chemistry/rhf-auto.js"),
    ]);
    show(`<span class="dim">${mol.name} · STO-3G · building integrals + running SCF…</span>`);
    await raf();
    const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(mol.atoms as never, "sto-3g");
    const n = shells.length;
    const t0 = performance.now();
    const { hf, provenance } = await runRHFAuto(shells, nuclei, nElectrons, {
      force: n >= 20 ? "df" : undefined,
      hf: { useDIIS: true, energyTol: 1e-8, densityTol: 1e-6, maxIter: 100 },
    });
    const ms = performance.now() - t0;
    if (eng) eng.textContent = provenance.engine;
    show(
      `<span class="dim">${mol.name} · STO-3G · Hartree–Fock</span>\n` +
      `${hf.converged ? `<span class="ok">✓ converged</span>` : `<span class="e">did not converge</span>`} in ${hf.iter} SCF iterations\n` +
      `E = <span class="e">${hf.energy.toFixed(6)} Ha</span>\n` +
      `<span class="dim">${n} basis functions · ${ms.toFixed(0)} ms · engine: ${provenance.engine} · cross-checked vs PySCF in CI</span>`,
    );
    try { viz?.show(hf.history, hf.orbitalEnergies, hf.nOccupied); } catch { /* viz is best-effort */ }
  } catch (err) {
    show(`<span class="e">couldn't run here:</span> <span class="dim">${(err as Error).message}</span>`);
  } finally {
    busy = false;
    if (run) { run.disabled = false; run.textContent = "▶ Run again"; }
  }
}

run?.addEventListener("click", () => { void runHF(); });
window.addEventListener("load", () => { setTimeout(() => { void runHF(); }, 700); });
