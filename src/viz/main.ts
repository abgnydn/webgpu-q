// ─────────────────────────────────────────────────────────────
// main.ts — viz entry point. Three synchronized panels:
//
//   Left   — H₂ electron density ρ(r) at current bond length R.
//   Right  — Conditional pair density ρ(r₂ | r₀); the cursor is r₀.
//   Bottom — MPS bond network: an N-qubit chain running a brick-wall
//            circuit; bond thickness ∝ Schmidt entropy at that cut.
//
// A time slider scrubs through pre-recorded snapshots of the
// brick-wall circuit so you can play, pause, rewind. R, r₀, and
// the time slider are all independent.
// ─────────────────────────────────────────────────────────────

import { densityGridFromCoeffs } from "./density.js";
import { pairDensityGridFromCoeffs } from "./pair-density.js";
import { fciState } from "./h2-fci-state.js";
import { Scene } from "./scene.js";
import { recordBrickwall, recordTFIMGroundState, recordHeisenbergGroundState, recordTFIMQuench, type MPSRecording } from "./mps-recorder.js";
import { renderMPSNetwork } from "./mps-svg.js";
import { renderLightcone } from "./lightcone-svg.js";
import { sweepTFIMOrderParameter, type SweepPoint } from "./order-sweep.js";
import { renderOrderSweep } from "./order-svg.js";
import { recordTrajectory, type TrajectoryRecording } from "../manybody/trajectory.js";
import { tfim } from "../manybody/hamiltonian.js";
import { renderTrajectory } from "./trajectory-svg.js";

const GRID_SIZE = 48;
const MAX_SPLATS = GRID_SIZE * GRID_SIZE * GRID_SIZE;

function $<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} not in DOM`);
  return el as T;
}

async function getDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error("WebGPU not available in this browser");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no GPU adapter");
  return adapter.requestDevice({});
}

async function boot(): Promise<void> {
  const canvasL = $<HTMLCanvasElement>("cL");
  const canvasR = $<HTMLCanvasElement>("cR");
  const netSvgEl = document.getElementById("netSvg");
  if (!(netSvgEl instanceof SVGSVGElement)) throw new Error("#netSvg missing or wrong type");
  const netSvg = netSvgEl;
  const statusL = $("statusL");
  const statusR = $("statusR");
  const statusNet = $("statusNet");
  const bond = $<HTMLInputElement>("bond");
  const bondLabel = $("bondLabel");
  const r0x = $<HTMLInputElement>("r0x");
  const r0y = $<HTMLInputElement>("r0y");
  const r0z = $<HTMLInputElement>("r0z");
  const r0xLabel = $("r0x-label");
  const r0yLabel = $("r0y-label");
  const r0zLabel = $("r0z-label");
  const nq = $<HTMLInputElement>("nq");
  const depth = $<HTMLInputElement>("depth");
  const time = $<HTMLInputElement>("time");
  const nLabel = $("nLabel");
  const dLabel = $("dLabel");
  const tLabel = $("tLabel");
  const rerunBtn = $<HTMLButtonElement>("rerun");
  const playBtn = $<HTMLButtonElement>("play");
  const resetBtn = $<HTMLButtonElement>("reset");
  const modelSel = $<HTMLSelectElement>("model");
  const hField = $<HTMLInputElement>("hField");
  const hLabel = $("hLabel");
  const depthRow = $("depthRow");
  const hRow = $("hRow");
  const orderTitle = $("orderTitle");
  const orderRow = $("orderRow");
  const orderStatus = $("orderStatus");
  const sweepBtn = $<HTMLButtonElement>("sweepBtn");
  const gammaRow = $("gammaRow");
  const gamma = $<HTMLInputElement>("gamma");
  const gammaLabel = $("gammaLabel");
  const orderSvgEl = document.getElementById("orderSvg");
  if (!(orderSvgEl instanceof SVGSVGElement)) throw new Error("#orderSvg missing");
  const orderSvg = orderSvgEl;

  function resize(canvas: HTMLCanvasElement): void {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  }
  function resizeAll(): void { resize(canvasL); resize(canvasR); }
  window.addEventListener("resize", resizeAll);
  resizeAll();

  // A WebGPU failure must NOT abort boot(): the MPS bond-network panel below is
  // pure CPU (mps-recorder + inline SVG) and used to be left permanently on
  // "init…" because this path returned early. Degrade the 3D panels only.
  let sceneL: Scene | null = null;
  let sceneR: Scene | null = null;
  try {
    const device = await getDevice();
    sceneL = Scene.fromDevice({ device, canvas: canvasL, maxSplats: MAX_SPLATS });
    sceneR = Scene.fromDevice({ device, canvas: canvasR, maxSplats: MAX_SPLATS });
    sceneR.tint = [1.6, 1.05, 0.55];
  } catch (err) {
    const msg = `WebGPU unavailable — 3D density panels disabled: ${(err as Error).message}`;
    statusL.textContent = msg;
    statusL.style.color = "#ff6e6e";
    statusR.textContent = "3D panel needs WebGPU. The MPS bond-network panel below still works.";
    statusR.style.color = "#ff6e6e";
  }

  // ── 3D-cloud panel state ────────────────────────────────
  function R(): number { return parseFloat(bond.value); }
  function r0(): [number, number, number] {
    return [parseFloat(r0x.value), parseFloat(r0y.value), parseFloat(r0z.value)];
  }
  let cG = 1, cU = 0, fciEnergy = 0;
  function refreshFCI(): void {
    const s = fciState(R());
    cG = s.cG; cU = s.cU; fciEnergy = s.energy;
  }
  function rebuildLeft(): void {
    const t0 = performance.now();
    if (!sceneL) return;
    const grid = densityGridFromCoeffs(R(), cG, cU, GRID_SIZE);
    sceneL.updateGrid(grid);
    statusL.textContent =
      `${grid.gridSize}³ · max ρ ${grid.maxDensity.toFixed(3)} · ` +
      `c_g=${cG.toFixed(3)} c_u=${cU.toFixed(3)} · E_FCI=${fciEnergy.toFixed(4)} Ha · ${(performance.now() - t0).toFixed(1)} ms`;
    bondLabel.textContent = `${R().toFixed(3)} Å`;
  }
  function rebuildRight(): void {
    const t0 = performance.now();
    if (!sceneR) return;
    const grid = pairDensityGridFromCoeffs(R(), cG, cU, r0(), GRID_SIZE);
    sceneR.updateGrid(grid);
    statusR.textContent =
      `${grid.gridSize}³ · ρ(r₀) ${grid.cursorRho.toFixed(3)} · max ρ_cond ${grid.maxDensity.toFixed(3)} · ${(performance.now() - t0).toFixed(1)} ms`;
    const [x, y, z] = r0();
    r0xLabel.textContent = `${x.toFixed(2)} Bohr`;
    r0yLabel.textContent = `${y.toFixed(2)} Bohr`;
    r0zLabel.textContent = `${z.toFixed(2)} Bohr`;
  }

  refreshFCI();
  rebuildLeft();
  rebuildRight();

  bond.addEventListener("input", () => { refreshFCI(); rebuildLeft(); rebuildRight(); });
  for (const inp of [r0x, r0y, r0z]) inp.addEventListener("input", rebuildRight);

  // ── MPS bond-network panel state ────────────────────────
  let recording: MPSRecording | null = null;
  let trajectory: TrajectoryRecording | null = null;

  function applyModelUI(): void {
    const m = modelSel.value;
    depthRow.style.display = m === "brickwall" ? "" : "none";
    hRow.style.display = (m === "tfim" || m === "quench" || m === "trajectory") ? "" : "none";
    gammaRow.style.display = m === "trajectory" ? "" : "none";
    const showSweep = m === "tfim" ? "" : "none";
    orderTitle.style.display = showSweep;
    orderRow.style.display = showSweep;
  }

  function rerunRecording(): void {
    const N = parseInt(nq.value, 10);
    const m = modelSel.value;
    const t0 = performance.now();
    let label: string;

    if (m === "tfim") {
      const h = parseFloat(hField.value);
      hLabel.textContent = h.toFixed(2);
      recording = recordTFIMGroundState(N, h, { chiMax: 32, steps: 100, tau: 0.05 });
      label = `TFIM N=${N} h=${h.toFixed(2)}`;
    } else if (m === "heisenberg") {
      recording = recordHeisenbergGroundState(N, { chiMax: 32, steps: 100, tau: 0.05 });
      label = `Heisenberg N=${N}`;
    } else if (m === "quench") {
      const h = parseFloat(hField.value);
      hLabel.textContent = h.toFixed(2);
      recording = recordTFIMQuench(N, h, {
        chiMax: 64,
        steps: 80,
        tau: 0.1,
        initialState: "all-up",
      });
      label = `TFIM quench N=${N} h=${h.toFixed(2)}`;
    } else if (m === "trajectory") {
      const h = parseFloat(hField.value);
      const g = parseFloat(gamma.value);
      hLabel.textContent = h.toFixed(2);
      gammaLabel.textContent = g.toFixed(2);
      trajectory = recordTrajectory({
        hamiltonian: tfim(N, 1, h),
        gamma: g,
        steps: 80,
        tau: 0.1,
        chiMax: 32,
        seed: 0xCAFE5E0,
        initialState: "x-pol",
      });
      recording = null;
      label = `monitored trajectory N=${N} h=${h.toFixed(2)} γ=${g.toFixed(2)}`;
    } else {
      const D = parseInt(depth.value, 10);
      dLabel.textContent = D.toString();
      recording = recordBrickwall({ nQubits: N, depth: D, chiMax: 64, seed: 0xBA1CA70 });
      label = `brick-wall N=${N} D=${D}`;
    }

    nLabel.textContent = N.toString();
    if (m === "trajectory" && trajectory) {
      time.max = (trajectory.history.length - 1).toString();
      time.value = "0";
      drawNet();
      let totalMeas = 0;
      for (const s of trajectory.history) totalMeas += s.measurements.length;
      statusNet.textContent =
        `${label} · ${trajectory.history.length - 1} steps · ${totalMeas} measurements · max S=${trajectory.maxEntropy.toFixed(2)} bits · build ${(performance.now() - t0).toFixed(0)} ms`;
    } else if (recording) {
      time.max = (recording.snapshots.length - 1).toString();
      time.value = "0";
      drawNet();
      statusNet.textContent =
        `${label} · ${recording.snapshots.length - 1} steps · max χ=${recording.maxBond} · max S=${recording.maxEntropy.toFixed(2)} bits · build ${(performance.now() - t0).toFixed(0)} ms`;
    }
  }
  function drawNet(): void {
    const m = modelSel.value;
    const t = parseInt(time.value, 10);
    if (m === "trajectory" && trajectory) {
      renderTrajectory({ root: netSvg, recording: trajectory, cursorT: t });
      tLabel.textContent = `${t} / ${trajectory.history.length - 1}`;
      return;
    }
    if (!recording) return;
    if (m === "quench") {
      renderLightcone({ root: netSvg, recording, cursorT: t });
    } else {
      const snap = recording.snapshots[Math.min(t, recording.snapshots.length - 1)]!;
      renderMPSNetwork({
        root: netSvg,
        nQubits: recording.nQubits,
        maxEntropy: Math.max(recording.maxEntropy, 0.01),
        snapshot: snap,
      });
    }
    tLabel.textContent = `${t} / ${recording.snapshots.length - 1}`;
  }
  // Order-parameter sweep: cached set of points; redo on demand or N change.
  let orderPoints: SweepPoint[] = [];
  function runSweep(): void {
    const N = parseInt(nq.value, 10);
    orderStatus.textContent = `running TFIM h-sweep at N=${N}…`;
    // Yield to layout so the "running…" label paints before we hog the thread.
    setTimeout(() => {
      const t0 = performance.now();
      orderPoints = sweepTFIMOrderParameter({
        N,
        hMin: 0,
        hMax: 2,
        steps: 25,
        chiMax: 16,
        imagTimeSteps: 60,
        tau: 0.05,
      });
      const dt = performance.now() - t0;
      drawOrder();
      orderStatus.textContent =
        `${orderPoints.length} points · N=${N} · ${dt.toFixed(0)} ms · |m_z| crashes near h=1 (the QPT)`;
    }, 30);
  }
  function drawOrder(): void {
    if (modelSel.value !== "tfim" || orderPoints.length === 0) return;
    renderOrderSweep({ root: orderSvg, points: orderPoints, cursorH: parseFloat(hField.value) });
  }
  sweepBtn.addEventListener("click", runSweep);
  hField.addEventListener("input", drawOrder);

  applyModelUI();
  rerunRecording();

  nq.addEventListener("input", rerunRecording);
  depth.addEventListener("input", rerunRecording);
  hField.addEventListener("input", () => { rerunRecording(); drawOrder(); });
  gamma.addEventListener("input", () => {
    if (modelSel.value === "trajectory") rerunRecording();
  });
  modelSel.addEventListener("change", () => {
    applyModelUI(); rerunRecording();
    if (modelSel.value === "tfim" && orderPoints.length === 0) runSweep();
  });
  time.addEventListener("input", drawNet);
  rerunBtn.addEventListener("click", rerunRecording);
  resetBtn.addEventListener("click", () => { time.value = "0"; drawNet(); });

  // Play: step time forward at ~30 fps until end.
  let playing = false;
  let lastStep = 0;
  function play(): void {
    if (!recording) return;
    if (parseInt(time.value, 10) >= recording.snapshots.length - 1) {
      time.value = "0";
    }
    playing = true;
    playBtn.textContent = "⏸ pause";
  }
  function pause(): void {
    playing = false;
    playBtn.textContent = "▶ play";
  }
  playBtn.addEventListener("click", () => { (playing ? pause : play)(); });

  function tickPlay(now: number): void {
    if (playing && recording && now - lastStep > 80) {
      const t = parseInt(time.value, 10);
      const next = t + 1;
      if (next >= recording.snapshots.length) {
        pause();
      } else {
        time.value = next.toString();
        drawNet();
      }
      lastStep = now;
    }
  }

  // ── Independent orbit cameras per pane ──────────────────
  function attachOrbit(canvas: HTMLCanvasElement, scene: Scene): void {
    let dragging = false;
    let lx = 0, ly = 0;
    canvas.addEventListener("pointerdown", (e) => {
      dragging = true; lx = e.clientX; ly = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx; const dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      scene.orbitBy(-dx * 0.01, -dy * 0.01);
    });
    canvas.addEventListener("pointerup", (e) => {
      dragging = false; canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      scene.zoomBy(Math.exp(e.deltaY * 0.001));
    }, { passive: false });
  }
  if (sceneL) attachOrbit(canvasL, sceneL);
  if (sceneR) attachOrbit(canvasR, sceneR);

  // ── Render loop ────────────────────────────────────────
  function tick(now: number): void {
    tickPlay(now);
    sceneL?.render();
    sceneR?.render();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

boot().catch((err: unknown) => {
  console.error(err);
  // Surface it: previously any boot failure left the page silently half-rendered.
  const status = document.getElementById("statusL");
  if (status) {
    status.textContent = `Failed to start: ${(err as Error)?.message ?? String(err)}`;
    status.style.color = "#ff6e6e";
  }
});
