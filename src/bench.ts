// ─────────────────────────────────────────────────────────────
// Benchmark harness: GPU vs CPU reference.
//
// Sweeps qubit count N ∈ [2, 24] and measures gates/sec for a
// random brick-wall circuit. Reports wall-clock per gate,
// effective bandwidth (statevector bytes touched per gate), and
// cross-checks norm.
//
// Runs in the browser. Call `runBench()` from the console or
// a button to populate a <table id="bench">.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "./quantum.js";
import { CpuCircuit } from "./cpu-reference.js";
import { randomCircuit } from "./circuits.js";

export interface BenchRow {
  nQubits: number;
  nStates: number;
  gatesApplied: number;
  gpuMs: number;
  cpuMs: number;
  gpuGatesPerSec: number;
  cpuGatesPerSec: number;
  gpuGBps: number;          // approx bytes touched / sec (statevector bytes × 2 r/w)
  normGpu: number;
}

/** How many layers of brickwall per N. Smaller N → more layers to fight noise. */
function layersFor(n: number): number {
  if (n < 10) return 50;
  if (n < 16) return 20;
  if (n < 20) return 8;
  return 3;
}

async function oneRun(device: GPUDevice, n: number): Promise<BenchRow> {
  const layers = layersFor(n);

  // GPU warmup (compile/upload first).
  {
    const warm = new QuantumCircuit({ device, nQubits: n });
    randomCircuit(warm, 1, 1);
    await warm.probabilities();
    warm.dispose();
  }

  const qc = new QuantumCircuit({ device, nQubits: n });
  const t0 = performance.now();
  randomCircuit(qc, layers, 0xBEEF);
  // force completion with readback
  const pGpu = await qc.probabilities();
  const t1 = performance.now();
  const gpuMs = t1 - t0;

  // count gates applied: ~ n single-q + n/2 two-q per layer
  const singles = layers * n;
  const twos = layers * Math.floor(n / 2);
  const gatesApplied = singles + twos;

  // CPU reference — only for small N to keep the page responsive.
  let cpuMs = Number.NaN;
  if (n <= 18) {
    const cpu = new CpuCircuit(n);
    const c0 = performance.now();
    randomCircuit(cpu, layers, 0xBEEF);
    cpu.probabilities();
    const c1 = performance.now();
    cpuMs = c1 - c0;
  }

  let normGpu = 0;
  for (const p of pGpu) normGpu += p;

  const nStates = 1 << n;
  // each gate touches every amplitude (read + write, 8 B each for vec2<f32>)
  // single-qubit: reads 2 and writes 2 per pair, nStates amplitudes total
  // approximation: 2 × 8 B × nStates per gate (R + W)
  const bytesPerGate = 2 * 8 * nStates;
  const gpuGBps = (bytesPerGate * gatesApplied) / (gpuMs / 1000) / 1e9;

  qc.dispose();

  return {
    nQubits: n,
    nStates,
    gatesApplied,
    gpuMs,
    cpuMs,
    gpuGatesPerSec: gatesApplied / (gpuMs / 1000),
    cpuGatesPerSec: gatesApplied / (cpuMs / 1000),
    gpuGBps,
    normGpu,
  };
}

export async function runBench(
  qubitSweep: number[] = [4, 8, 12, 16, 20, 22, 24],
  sink?: (row: BenchRow) => void,
): Promise<BenchRow[]> {
  const device = await initGPU();
  const out: BenchRow[] = [];
  for (const n of qubitSweep) {
    const row = await oneRun(device, n);
    out.push(row);
    sink?.(row);
  }
  return out;
}

// If this module is loaded directly as the page's entry, wire it
// to a <table id="bench"> automatically.
if (typeof document !== "undefined" && document.getElementById("bench")) {
  const table = document.getElementById("bench") as HTMLTableElement;
  table.innerHTML = `<thead><tr>
    <th align="right">N</th>
    <th align="right">states</th>
    <th align="right">gates</th>
    <th align="right">GPU ms</th>
    <th align="right">CPU ms</th>
    <th align="right">GPU gates/s</th>
    <th align="right">GPU GB/s</th>
    <th align="right">speedup</th>
    <th align="right">‖ψ‖²</th>
  </tr></thead><tbody></tbody>`;
  const tbody = table.querySelector("tbody")!;

  const fmt = (x: number, d = 2): string =>
    Number.isFinite(x) ? x.toFixed(d) : "—";

  runBench(undefined, (r) => {
    const speedup = Number.isFinite(r.cpuMs) ? r.cpuMs / r.gpuMs : Number.NaN;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td align="right">${r.nQubits}</td>
      <td align="right">${r.nStates.toLocaleString()}</td>
      <td align="right">${r.gatesApplied}</td>
      <td align="right">${fmt(r.gpuMs, 1)}</td>
      <td align="right">${fmt(r.cpuMs, 1)}</td>
      <td align="right">${(r.gpuGatesPerSec / 1000).toFixed(1)}k</td>
      <td align="right">${fmt(r.gpuGBps)}</td>
      <td align="right">${Number.isFinite(speedup) ? speedup.toFixed(1) + "×" : "—"}</td>
      <td align="right">${r.normGpu.toFixed(5)}</td>`;
    tbody.appendChild(tr);
  }).catch((e) => {
    console.error(e);
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="9" style="color:#c00">bench failed: ${(e as Error).message}</td>`;
    tbody.appendChild(tr);
  });
}
