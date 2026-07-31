// ─────────────────────────────────────────────────────────────
// Browser demo entry point.
//
// Runs a handful of textbook circuits on the GPU, cross-checks
// probabilities against the CPU reference, and renders the
// results as a table in the page.
// ─────────────────────────────────────────────────────────────

import { initGPU, QuantumCircuit } from "./quantum.js";
import { CpuCircuit } from "./cpu-reference.js";
import { bell, deutschJozsaConstant, ghz, qft, randomCircuit } from "./circuits.js";
import type { GateApi } from "./circuits.js";

interface DemoSpec {
  name: string;
  nQubits: number;
  build: (c: GateApi) => void;
}

const DEMOS: DemoSpec[] = [
  { name: "Bell (2q)",             nQubits: 2,  build: bell },
  { name: "GHZ-5 (5q)",            nQubits: 5,  build: ghz },
  { name: "GHZ-10 (10q)",          nQubits: 10, build: ghz },
  { name: "QFT-6 (6q)",            nQubits: 6,  build: qft },
  { name: "Deutsch–Jozsa-4 const", nQubits: 4,  build: deutschJozsaConstant },
  { name: "Random 12q × 8 layers", nQubits: 12, build: (c) => randomCircuit(c, 8, 42) },
];

const root = document.getElementById("demo-slot") ?? document.getElementById("app")!;

function log(html: string): void {
  const div = document.createElement("div");
  div.innerHTML = html;
  root.appendChild(div);
}

function fmt(x: number, digits = 6): string {
  if (x === 0) return "0";
  if (Math.abs(x) < 1e-9) return x.toExponential(2);
  return x.toFixed(digits);
}

async function main(): Promise<void> {
  log(`<h1>webgpu-q — quantum circuit simulator</h1>`);
  // Class is load-bearing: demo.html hides THIS blurb (it has its own header).
  // It used to hide `p:first-of-type`, which also swallowed the "WebGPU
  // unavailable" notice below, leaving non-WebGPU browsers with a blank page.
  log(`<p class="intro-blurb">Statevector simulation on WebGPU, cross-checked against a
        Float64 CPU reference. Each row below is a circuit; "Δ" is the
        max |p_gpu − p_cpu| over all basis states.</p>`);

  let device: GPUDevice;
  try {
    device = await initGPU();
  } catch (err) {
    log(`<p style="color:#c00"><b>WebGPU unavailable:</b> ${(err as Error).message}</p>`);
    return;
  }

  // Adapter info (new WebGPU spec): promise on GPUDevice.adapterInfo in some UAs.
  try {
    const info = (device as unknown as { adapterInfo?: GPUAdapterInfo }).adapterInfo;
    if (info) {
      log(`<p style="color:#666">adapter = <code>${info.vendor || "?"}/${info.architecture || "?"}</code></p>`);
    }
  } catch {
    // ignore
  }

  // Build the table header.
  const table = document.createElement("table");
  table.innerHTML = `
    <thead><tr>
      <th align="left">circuit</th>
      <th align="right">qubits</th>
      <th align="right">GPU ms</th>
      <th align="right">CPU ms</th>
      <th align="right">speedup</th>
      <th align="right">‖ψ‖² GPU</th>
      <th align="right">max Δp</th>
      <th>ok?</th>
    </tr></thead>
    <tbody></tbody>`;
  root.appendChild(table);
  const tbody = table.querySelector("tbody")!;

  for (const spec of DEMOS) {
    const row = document.createElement("tr");
    tbody.appendChild(row);
    row.innerHTML = `<td>${spec.name}</td><td align="right">${spec.nQubits}</td>
      <td colspan="6">running…</td>`;

    const { gpuMs, cpuMs, normGpu, maxDelta, ok } = await runOne(device, spec);
    const speedup = cpuMs / gpuMs;
    row.innerHTML = `
      <td>${spec.name}</td>
      <td align="right">${spec.nQubits}</td>
      <td align="right">${gpuMs.toFixed(2)}</td>
      <td align="right">${cpuMs.toFixed(2)}</td>
      <td align="right">${speedup.toFixed(2)}×</td>
      <td align="right">${fmt(normGpu)}</td>
      <td align="right">${fmt(maxDelta)}</td>
      <td>${ok ? "✅" : "❌"}</td>`;
    // Yield between demos so the UI stays responsive.
    await new Promise((r) => setTimeout(r, 0));
  }

  log(`<p style="color:#666">Done. max|Δp| &lt; 1e-5 is the pass bar
       (f32 GPU vs f64 CPU, so ~6 decimal digits is expected).</p>`);
}

interface Result {
  gpuMs: number;
  cpuMs: number;
  normGpu: number;
  maxDelta: number;
  ok: boolean;
}

async function runOne(device: GPUDevice, spec: DemoSpec): Promise<Result> {
  // GPU
  const qc = new QuantumCircuit({ device, nQubits: spec.nQubits });
  const tGpu0 = performance.now();
  spec.build(qc);
  const pGpu = await qc.probabilities();
  const tGpu1 = performance.now();

  // CPU reference
  const cpu = new CpuCircuit(spec.nQubits);
  const tCpu0 = performance.now();
  spec.build(cpu);
  const pCpu = cpu.probabilities();
  const tCpu1 = performance.now();

  let normGpu = 0;
  let maxDelta = 0;
  const n = 1 << spec.nQubits;
  for (let i = 0; i < n; i++) {
    const g = pGpu[i] ?? 0;
    const c = pCpu[i] ?? 0;
    normGpu += g;
    const d = Math.abs(g - c);
    if (d > maxDelta) maxDelta = d;
  }

  qc.dispose();

  return {
    gpuMs: tGpu1 - tGpu0,
    cpuMs: tCpu1 - tCpu0,
    normGpu,
    maxDelta,
    ok: maxDelta < 1e-5 && Math.abs(normGpu - 1) < 1e-4,
  };
}

main().catch((err) => {
  console.error(err);
  log(`<pre style="color:#c00">${(err as Error).stack || err}</pre>`);
});
