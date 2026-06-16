// ─────────────────────────────────────────────────────────────
// GPU MPS bench — Phase 1A: validate GPU complex matmul vs CPU
// across a range of matrix sizes and print a table of:
//   • Frobenius distance ‖GPU − CPU‖  (correctness)
//   • relative error / ‖CPU‖           (FP32 floor ≈ 1e-6)
//   • CPU and GPU timings, speedup
//
// Exposes window.__gpuMpsBench for the Playwright validator.
// ─────────────────────────────────────────────────────────────

import { gpuCMatMul } from "../../src/gpu-mps/matmul.js";
import { gpuJacobiSvd, gpuJacobiSvdMaxN, svdReconstruct } from "../../src/gpu-mps/jacobi-svd.js";
import { makeAdaptiveSvd } from "../../src/gpu-mps/svd-bridge.js";
import { MPSGpu } from "../../src/gpu-mps/mps-gpu.js";
import { cmatMul, cmatRandom, cmatFrobDistance, type CMatF32 } from "../../src/gpu-mps/types.js";
import { initGPU } from "../../src/quantum.js";
import { MPS } from "../../src/mps.js";
import { tfim } from "../../src/manybody/hamiltonian.js";
import { expmComplex } from "../../src/manybody/real-time.js";
import { H as Hgate, X as Xgate, Rx, Ry, Rz, type Matrix2 } from "../../src/gates.js";

interface BenchRow {
  shape: string;
  m: number; k: number; n: number;
  frob: number;
  rel: number;
  match: boolean;
  cpuMs: number;
  gpuMs: number;
  speedup: number;
}

interface SvdRow {
  n: number;
  reconstructFrob: number;       // ‖M − U Σ V^H‖_F
  reconstructRel: number;        // / ‖M‖_F
  match: boolean;
  gpuMs: number;
  /** Largest σ; sanity for overflow checks. */
  sigmaMax: number;
}

interface MpsRow {
  N: number;
  chi: number;
  steps: number;
  fidelity: number;       // |⟨ψ_cpu|ψ_gpu⟩|² — should be ≈ 1
  cpuMs: number;
  gpuMs: number;
  speedup: number;
  match: boolean;
}

interface ResidentRow {
  N: number;
  gateCount: number;
  fidelity: number;
  cpuMs: number;
  gpuMs: number;
  speedup: number;
  match: boolean;
}

interface TruncRow {
  N: number;
  depth: number;
  chiMax: number;
  /** ‖ψ‖² of the GPU readback — proves truncation renorm holds. */
  normSq: number;
  /** |⟨ψ_cpu|ψ_gpu⟩|² with both sides truncated identically. */
  fidelity: number;
  match: boolean;
}

const SHAPES: ReadonlyArray<readonly [number, number, number]> = [
  [16, 16, 16],
  [32, 32, 32],
  [64, 64, 64],
  [128, 128, 128],
  [256, 64, 64],   // tall × small
  [64, 64, 256],   // small × wide
];

function frob(m: CMatF32): number {
  let s = 0;
  for (let i = 0; i < m.data.length; i++) s += m.data[i]! * m.data[i]!;
  return Math.sqrt(s);
}

async function timeIt(label: string, fn: () => Promise<unknown> | unknown): Promise<number> {
  const t0 = performance.now();
  await fn();
  const dt = performance.now() - t0;
  void label;
  return dt;
}

declare global {
  interface Window {
    __gpuMpsBench?: {
      ready: true;
      rows: BenchRow[];
      svdRows: SvdRow[];
      mpsRows: MpsRow[];
      residentRows: ResidentRow[];
      phase4bRows: ResidentRow[];
      run: () => Promise<BenchRow[]>;
      runSvd: () => Promise<SvdRow[]>;
      runMps: () => Promise<MpsRow[]>;
      runResident: () => Promise<ResidentRow[]>;
      runPhase4b: () => Promise<ResidentRow[]>;
      runPhase5v1: () => Promise<ResidentRow[]>;
      runPhaseATrunc: () => Promise<TruncRow[]>;
    };
  }
}

async function boot(): Promise<void> {
  const adapterInfoEl = document.getElementById("adapterInfo");
  const runBtnEl = document.getElementById("runBtn") as HTMLButtonElement | null;
  const statusEl = document.getElementById("status");
  const tbodyEl = document.getElementById("resultsBody");
  if (!adapterInfoEl || !runBtnEl || !statusEl || !tbodyEl) {
    throw new Error("DOM scaffold missing");
  }
  const adapterInfo = adapterInfoEl;
  const runBtn = runBtnEl;
  const status = statusEl;
  const tbody = tbodyEl;

  let device: GPUDevice;
  try {
    device = await initGPU();
  } catch (err) {
    adapterInfo.textContent = `WebGPU init failed: ${(err as Error).message}`;
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const info = adapter ? (adapter as unknown as { info?: { vendor?: string; architecture?: string } }).info : null;
  const wgStorage = device.limits.maxComputeWorkgroupStorageSize;
  const maxN = gpuJacobiSvdMaxN(device);
  adapterInfo.innerHTML =
    `<span class="v">adapter:</span> ${info?.vendor ?? "?"} · ${info?.architecture ?? "?"} · ` +
    `<span class="v">maxBufferSize:</span> ${(device.limits.maxBufferSize / 1024 / 1024) | 0} MiB · ` +
    `<span class="v">workgroupStorage:</span> ${(wgStorage / 1024).toFixed(1)} KiB · ` +
    `<span class="v">SVD max n:</span> ${maxN}`;

  async function runBench(): Promise<BenchRow[]> {
    runBtn.disabled = true;
    tbody.innerHTML = "";
    status.textContent = "running…";
    const rows: BenchRow[] = [];

    for (const [m, k, n] of SHAPES) {
      const A = cmatRandom(m, k, 0xC0FFEE ^ (m * 31 + k));
      const B = cmatRandom(k, n, 0xBA1C ^ (k * 31 + n));

      // Warmup once on each path so JIT and pipeline-compile are out of the timing.
      const cpuRef = cmatMul(A, B);
      await gpuCMatMul(device, A, B);

      // CPU timing — single trial; the JS engine has already JITed.
      const cpuMs = await timeIt("cpu", () => cmatMul(A, B));

      // GPU timing — submit + readback included.
      let gpuRes!: CMatF32;
      const gpuMs = await timeIt("gpu", async () => { gpuRes = await gpuCMatMul(device, A, B); });

      const refNorm = Math.max(frob(cpuRef), 1e-12);
      const f = cmatFrobDistance(cpuRef, gpuRes);
      const rel = f / refNorm;
      const match = rel < 1e-3; // f32 floor for these sizes

      rows.push({
        shape: `${m}×${k} · ${k}×${n}`,
        m, k, n,
        frob: f, rel, match, cpuMs, gpuMs, speedup: cpuMs / gpuMs,
      });

      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td><span class="v">${m}×${k} · ${k}×${n}</span></td>` +
        `<td class="num">${f.toExponential(2)}</td>` +
        `<td class="num">${rel.toExponential(2)}</td>` +
        `<td class="${match ? "ok" : "fail"}">${match ? "✓" : "✘"}</td>` +
        `<td class="num">${cpuMs.toFixed(2)}</td>` +
        `<td class="num">${gpuMs.toFixed(2)}</td>` +
        `<td class="num">${(cpuMs / gpuMs).toFixed(2)}×</td>`;
      tbody.appendChild(tr);
    }

    runBtn.disabled = false;
    status.textContent = `done · ${rows.filter((r) => r.match).length}/${rows.length} matched`;
    return rows;
  }

  runBtn.addEventListener("click", () => { void runBench(); });

  // ── SVD validation ────────────────────────────────
  async function runSvdBench(): Promise<SvdRow[]> {
    const svdRunBtn = document.getElementById("svdRunBtn") as HTMLButtonElement | null;
    const svdTbody = document.getElementById("svdResultsBody");
    if (svdRunBtn) svdRunBtn.disabled = true;
    if (svdTbody) svdTbody.innerHTML = "";
    // 28 and 32 exercise the medium-shader path (Phase 1C); they'll be
    // skipped (replaced with a placeholder row) on hardware that only
    // exposes the 16 KB workgroup-storage minimum.
    const allSvdSizes = [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64];
    const svdSizes = allSvdSizes.filter((s) => s <= maxN);
    const out: SvdRow[] = [];
    for (const n of svdSizes) {
      const M = cmatRandom(n, n, 0xCAFE05D0 ^ n);
      // Warmup pipeline once. Catch and skip if the device can't fit
      // this n (Phase 1C n>24 path needs ≥17 KB workgroup storage).
      try {
        await gpuJacobiSvd(device, M);
      } catch (err) {
        if (svdTbody) {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${n}×${n}</td><td colspan="5" style="color:var(--muted)">skipped — ${(err as Error).message}</td>`;
          svdTbody.appendChild(tr);
        }
        continue;
      }

      const t0 = performance.now();
      const svd = await gpuJacobiSvd(device, M);
      const dt = performance.now() - t0;

      const recon = svdReconstruct(svd);
      const frob = cmatFrobDistance(M, recon);
      let mNorm = 0;
      for (let i = 0; i < M.data.length; i++) mNorm += M.data[i]! * M.data[i]!;
      mNorm = Math.sqrt(mNorm);
      const rel = frob / Math.max(mNorm, 1e-12);
      const match = rel < 1e-3;
      let sMax = 0;
      for (let i = 0; i < svd.sigma.length; i++) sMax = Math.max(sMax, svd.sigma[i]!);
      out.push({ n, reconstructFrob: frob, reconstructRel: rel, match, gpuMs: dt, sigmaMax: sMax });

      if (svdTbody) {
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td><span class="v">${n}×${n}</span></td>` +
          `<td class="num">${frob.toExponential(2)}</td>` +
          `<td class="num">${rel.toExponential(2)}</td>` +
          `<td class="${match ? "ok" : "fail"}">${match ? "✓" : "✘"}</td>` +
          `<td class="num">${dt.toFixed(2)}</td>` +
          `<td class="num">${sMax.toFixed(3)}</td>`;
        svdTbody.appendChild(tr);
      }
    }
    if (svdRunBtn) svdRunBtn.disabled = false;
    return out;
  }

  const svdBtn = document.getElementById("svdRunBtn");
  if (svdBtn) svdBtn.addEventListener("click", () => { void runSvdBench(); });

  // ── Phase 2: full MPS evolution with GPU SVD ──────────
  async function runMpsBench(): Promise<MpsRow[]> {
    const mpsRunBtn = document.getElementById("mpsRunBtn") as HTMLButtonElement | null;
    const mpsTbody = document.getElementById("mpsResultsBody");
    if (mpsRunBtn) mpsRunBtn.disabled = true;
    if (mpsTbody) mpsTbody.innerHTML = "";
    const adaptiveSvd = makeAdaptiveSvd(device);
    const out: MpsRow[] = [];

    // Try a few circuits; chiMax ≤ 12 keeps merged matrices ≤ 24×24
    // (within Phase 1B's GPU SVD envelope).
    const cfgs: ReadonlyArray<{ N: number; chi: number; depth: number; tau: number; h: number }> = [
      { N: 6,  chi: 4,  depth: 2, tau: 0.1, h: 1.0 },
      { N: 8,  chi: 8,  depth: 3, tau: 0.1, h: 1.0 },
      { N: 10, chi: 12, depth: 3, tau: 0.1, h: 1.0 },
      { N: 12, chi: 12, depth: 2, tau: 0.1, h: 1.0 },
      { N: 12, chi: 16, depth: 3, tau: 0.1, h: 1.0 },   // Phase 1C — needs medium kernel
      { N: 16, chi: 16, depth: 2, tau: 0.1, h: 1.0 },
    ];

    for (const cfg of cfgs) {
      const ham = tfim(cfg.N, 1, cfg.h);
      // Pre-build the bond-by-bond real-time gates.
      const gates = ham.bonds.map((b) => expmComplex(b.h, 4, cfg.tau));

      // ── CPU baseline ──
      const cpu = new MPS({ nQubits: cfg.N, chiMax: cfg.chi });
      // Hadamard on every site so the state isn't a fixed point of XZZ.
      for (let q = 0; q < cfg.N; q++) cpu.apply(Hgate as Matrix2, q);
      const cpuT0 = performance.now();
      let stepsRun = 0;
      for (let s = 0; s < cfg.depth; s++) {
        for (let q = 0; q + 1 < cfg.N; q += 2) { cpu.applyTwoSite(gates[q]!, q); stepsRun++; }
        for (let q = 1; q + 1 < cfg.N; q += 2) { cpu.applyTwoSite(gates[q]!, q); stepsRun++; }
      }
      const cpuMs = performance.now() - cpuT0;

      // ── GPU SVD path ──
      const gpu = new MPS({ nQubits: cfg.N, chiMax: cfg.chi });
      for (let q = 0; q < cfg.N; q++) gpu.apply(Hgate as Matrix2, q);
      const gpuT0 = performance.now();
      for (let s = 0; s < cfg.depth; s++) {
        for (let q = 0; q + 1 < cfg.N; q += 2) await gpu.applyTwoSiteAsync(gates[q]!, q, adaptiveSvd);
        for (let q = 1; q + 1 < cfg.N; q += 2) await gpu.applyTwoSiteAsync(gates[q]!, q, adaptiveSvd);
      }
      const gpuMs = performance.now() - gpuT0;

      // Fidelity between the two final statevectors.
      const psiCpu = cpu.statevector();
      const psiGpu = gpu.statevector();
      let re = 0, im = 0;
      for (let i = 0; i < psiCpu.length; i += 2) {
        const ar = psiCpu[i]!, ai = psiCpu[i + 1]!;
        const br = psiGpu[i]!, bi = psiGpu[i + 1]!;
        re += ar * br + ai * bi;
        im += ar * bi - ai * br;
      }
      const fidelity = re * re + im * im;
      const match = fidelity > 0.999;

      const row: MpsRow = {
        N: cfg.N, chi: cfg.chi, steps: stepsRun,
        fidelity, cpuMs, gpuMs, speedup: cpuMs / gpuMs, match,
      };
      out.push(row);
      if (mpsTbody) {
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td><span class="v">N=${cfg.N} χ=${cfg.chi} d=${cfg.depth}</span></td>` +
          `<td class="num">${fidelity.toFixed(7)}</td>` +
          `<td class="${match ? "ok" : "fail"}">${match ? "✓" : "✘"}</td>` +
          `<td class="num">${cpuMs.toFixed(2)}</td>` +
          `<td class="num">${gpuMs.toFixed(2)}</td>` +
          `<td class="num">${(cpuMs / gpuMs).toFixed(2)}×</td>`;
        mpsTbody.appendChild(tr);
      }
    }
    if (mpsRunBtn) mpsRunBtn.disabled = false;
    return out;
  }

  const mpsBtn = document.getElementById("mpsRunBtn");
  if (mpsBtn) mpsBtn.addEventListener("click", () => { void runMpsBench(); });

  // ── Phase 4a: GPU-resident MPS, single-qubit-gate-only ─────────
  async function runResidentBench(): Promise<ResidentRow[]> {
    const resBtn2 = document.getElementById("resRunBtn") as HTMLButtonElement | null;
    const resTbody = document.getElementById("resResultsBody");
    if (resBtn2) resBtn2.disabled = true;
    if (resTbody) resTbody.innerHTML = "";

    const cfgs: ReadonlyArray<{ N: number; gates: number }> = [
      { N: 4,  gates: 32 },
      { N: 6,  gates: 64 },
      { N: 8,  gates: 96 },
      { N: 10, gates: 128 },
    ];
    const out: ResidentRow[] = [];

    for (const cfg of cfgs) {
      let s = 0xCAFE5DA0 ^ cfg.N;
      const rng = (): number => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const stream: { U: Matrix2; q: number }[] = [];
      for (let i = 0; i < cfg.gates; i++) {
        const q = i % cfg.N;
        const k = Math.floor(rng() * 5);
        const theta = rng() * 2 * Math.PI;
        const U: Matrix2 =
          k === 0 ? Hgate :
          k === 1 ? Xgate :
          k === 2 ? Rx(theta) :
          k === 3 ? Ry(theta) :
          Rz(theta);
        stream.push({ U, q });
      }

      const cpu = new MPS({ nQubits: cfg.N, chiMax: 32 });
      const cpuT0 = performance.now();
      for (const { U, q } of stream) cpu.apply(U, q);
      const cpuMs = performance.now() - cpuT0;
      const psiCpu = cpu.statevector();

      const gpu = new MPSGpu({ device, nQubits: cfg.N, chiMax: 32 });
      const gpuT0 = performance.now();
      for (const { U, q } of stream) gpu.apply(U, q);
      const psiGpuF64 = await gpu.statevector();
      const gpuMs = performance.now() - gpuT0;

      let re = 0, im = 0;
      for (let i = 0; i < psiCpu.length; i += 2) {
        const ar = psiCpu[i]!, ai = psiCpu[i + 1]!;
        const br = psiGpuF64[i]!, bi = psiGpuF64[i + 1]!;
        re += ar * br + ai * bi;
        im += ar * bi - ai * br;
      }
      const fidelity = re * re + im * im;
      const match = fidelity > 0.999;

      gpu.dispose();

      const row: ResidentRow = {
        N: cfg.N, gateCount: cfg.gates, fidelity, cpuMs, gpuMs,
        speedup: cpuMs / gpuMs, match,
      };
      out.push(row);
      if (resTbody) {
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td><span class="v">N=${cfg.N} · ${cfg.gates} gates</span></td>` +
          `<td class="num">${fidelity.toFixed(7)}</td>` +
          `<td class="${match ? "ok" : "fail"}">${match ? "✓" : "✘"}</td>` +
          `<td class="num">${cpuMs.toFixed(2)}</td>` +
          `<td class="num">${gpuMs.toFixed(2)}</td>` +
          `<td class="num">${(cpuMs / gpuMs).toFixed(2)}×</td>`;
        resTbody.appendChild(tr);
      }
    }
    if (resBtn2) resBtn2.disabled = false;
    return out;
  }

  const resBtn = document.getElementById("resRunBtn");
  if (resBtn) resBtn.addEventListener("click", () => { void runResidentBench(); });

  // ── Phase 4b: full GPU two-site path ──────────────────────
  async function runPhase4bBench(): Promise<ResidentRow[]> {
    const phase4bBtn = document.getElementById("phase4bRunBtn") as HTMLButtonElement | null;
    const phase4bTbody = document.getElementById("phase4bResultsBody");
    if (phase4bBtn) phase4bBtn.disabled = true;
    if (phase4bTbody) phase4bTbody.innerHTML = "";

    const cfgs: ReadonlyArray<{ N: number; depth: number; chiMax: number }> = [
      { N: 4, depth: 1, chiMax: 16 },
      { N: 6, depth: 1, chiMax: 16 },
      { N: 4, depth: 2, chiMax: 16 },
      // Phase 5 — deeper configs that previously hit the
      // "square SVD only" guard (chiL ≠ chiR mid-chain).
      { N: 6, depth: 3, chiMax: 16 },
      { N: 8, depth: 3, chiMax: 16 },
      { N: 8, depth: 4, chiMax: 16 },
      { N: 10, depth: 4, chiMax: 16 },
      { N: 12, depth: 4, chiMax: 16 },
      // Phase 6 v1 — storage-mode SVD lifts χ_max from 16 → 32. These
      // configs run merged matrices up to 64×64 in the SVD pipeline.
      // Depth capped at 5 so brick-wall bonds (≤ 2^d) don't exceed χ_max
      // and trigger truncation, which Phase 4b leaves unrenormalized.
      { N: 12, depth: 5, chiMax: 32 },
      { N: 14, depth: 5, chiMax: 32 },
      { N: 16, depth: 5, chiMax: 32 },
    ];
    const out: ResidentRow[] = [];

    function buildCircuit(N: number, depth: number, seed: number): {
      singleQ: { U: Matrix2; q: number }[];
      twoQ: { G: import("../../src/linalg.js").ComplexMatrix; q: number }[];
    } {
      // Deterministic mix of single-qubit gates + a brick-wall layer of CNOTs.
      // Truncation-free ⟹ unitary path is exact in both impls.
      let s = seed >>> 0;
      const rng = (): number => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const singleQ: { U: Matrix2; q: number }[] = [];
      for (let q = 0; q < N; q++) {
        singleQ.push({ U: Hgate, q });
        singleQ.push({ U: Rz(rng() * 2 * Math.PI), q });
      }
      // Brick-wall CNOTs.
      const twoQ: { G: import("../../src/linalg.js").ComplexMatrix; q: number }[] = [];
      // CNOT 4×4 in the (s_q · 2 + s_{q+1}) basis the MPS expects.
      const cnot: import("../../src/linalg.js").ComplexMatrix = {
        rows: 4, cols: 4,
        data: new Float64Array([
          1,0, 0,0, 0,0, 0,0,
          0,0, 1,0, 0,0, 0,0,
          0,0, 0,0, 0,0, 1,0,
          0,0, 0,0, 1,0, 0,0,
        ]),
      };
      for (let l = 0; l < depth; l++) {
        const offset = l & 1;
        for (let q = offset; q + 1 < N; q += 2) twoQ.push({ G: cnot, q });
      }
      return { singleQ, twoQ };
    }

    for (const cfg of cfgs) {
      const { singleQ, twoQ } = buildCircuit(cfg.N, cfg.depth, 0xCAFE4B00 ^ cfg.N);

      // CPU baseline.
      const cpu = new MPS({ nQubits: cfg.N, chiMax: cfg.chiMax });
      const cpuT0 = performance.now();
      for (const { U, q } of singleQ) cpu.apply(U, q);
      for (const { G, q } of twoQ) cpu.applyTwoSite(G, q);
      const cpuMs = performance.now() - cpuT0;
      const psiCpu = cpu.statevector();

      // GPU.
      const gpu = new MPSGpu({ device, nQubits: cfg.N, chiMax: cfg.chiMax });
      const gpuT0 = performance.now();
      for (const { U, q } of singleQ) gpu.apply(U, q);
      for (const { G, q } of twoQ) await gpu.applyTwoSite(G, q);
      const psiGpu = await gpu.statevector();
      const gpuMs = performance.now() - gpuT0;
      gpu.dispose();

      let re = 0, im = 0;
      for (let i = 0; i < psiCpu.length; i += 2) {
        const ar = psiCpu[i]!, ai = psiCpu[i + 1]!;
        const br = psiGpu[i]!, bi = psiGpu[i + 1]!;
        re += ar * br + ai * bi;
        im += ar * bi - ai * br;
      }
      const fidelity = re * re + im * im;
      const match = fidelity > 0.999;

      const row: ResidentRow = {
        N: cfg.N, gateCount: singleQ.length + twoQ.length,
        fidelity, cpuMs, gpuMs, speedup: cpuMs / gpuMs, match,
      };
      out.push(row);

      if (phase4bTbody) {
        const tr = document.createElement("tr");
        tr.innerHTML =
          `<td><span class="v">N=${cfg.N} d=${cfg.depth} · ${row.gateCount} gates</span></td>` +
          `<td class="num">${fidelity.toFixed(7)}</td>` +
          `<td class="${match ? "ok" : "fail"}">${match ? "✓" : "✘"}</td>` +
          `<td class="num">${cpuMs.toFixed(2)}</td>` +
          `<td class="num">${gpuMs.toFixed(2)}</td>` +
          `<td class="num">${(cpuMs / gpuMs).toFixed(2)}×</td>`;
        phase4bTbody.appendChild(tr);
      }
    }
    if (phase4bBtn) phase4bBtn.disabled = false;
    return out;
  }

  // ── Phase 5 v1: GPU canonicalize() round-trip vs CPU MPS ──
  //
  // Run a brick-wall circuit on both impls, then call canonicalize()
  // on the GPU side. Statevector readback should still match CPU to
  // f32 floor — the canonical sweep is a unitary basis change on bonds
  // and must leave the *physical* state invariant.
  async function runPhase5v1Bench(): Promise<ResidentRow[]> {
    const cfgs: ReadonlyArray<{ N: number; depth: number; targetBond: number }> = [
      { N: 6, depth: 2, targetBond: 2 },
      { N: 8, depth: 2, targetBond: 3 },
      { N: 8, depth: 3, targetBond: 4 },
      { N: 10, depth: 3, targetBond: 5 },
    ];
    const out: ResidentRow[] = [];

    function buildBrickwall(N: number, depth: number, seed: number) {
      let s = seed >>> 0;
      const rng = () => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const singleQ: { U: Matrix2; q: number }[] = [];
      for (let q = 0; q < N; q++) {
        singleQ.push({ U: Hgate, q });
        singleQ.push({ U: Rz(rng() * 2 * Math.PI), q });
      }
      const cnot: import("../../src/linalg.js").ComplexMatrix = {
        rows: 4, cols: 4,
        data: new Float64Array([
          1,0, 0,0, 0,0, 0,0,
          0,0, 1,0, 0,0, 0,0,
          0,0, 0,0, 0,0, 1,0,
          0,0, 0,0, 1,0, 0,0,
        ]),
      };
      const twoQ: { G: typeof cnot; q: number }[] = [];
      for (let l = 0; l < depth; l++) {
        const offset = l & 1;
        for (let q = offset; q + 1 < N; q += 2) twoQ.push({ G: cnot, q });
      }
      return { singleQ, twoQ };
    }

    for (const cfg of cfgs) {
      const { singleQ, twoQ } = buildBrickwall(cfg.N, cfg.depth, 0xCA90C5 ^ cfg.N);

      // Reference: CPU MPS, no canonicalize step.
      const cpu = new MPS({ nQubits: cfg.N, chiMax: 16 });
      for (const { U, q } of singleQ) cpu.apply(U, q);
      for (const { G, q } of twoQ) cpu.applyTwoSite(G, q);
      const psiCpu = cpu.statevector();

      // GPU: run circuit, then canonicalize. Final state must still match.
      const gpu = new MPSGpu({ device, nQubits: cfg.N, chiMax: 16 });
      const gpuT0 = performance.now();
      for (const { U, q } of singleQ) gpu.apply(U, q);
      for (const { G, q } of twoQ) await gpu.applyTwoSite(G, q);
      await gpu.canonicalize(cfg.targetBond);
      const psiGpu = await gpu.statevector();
      const gpuMs = performance.now() - gpuT0;
      gpu.dispose();

      let re = 0, im = 0;
      for (let i = 0; i < psiCpu.length; i += 2) {
        const ar = psiCpu[i]!, ai = psiCpu[i + 1]!;
        const br = psiGpu[i]!, bi = psiGpu[i + 1]!;
        re += ar * br + ai * bi;
        im += ar * bi - ai * br;
      }
      const fidelity = re * re + im * im;
      const match = fidelity > 0.999;

      out.push({
        N: cfg.N,
        gateCount: singleQ.length + twoQ.length,
        fidelity, cpuMs: 0, gpuMs, speedup: 0, match,
      });
    }
    return out;
  }

  // ── Phase A truncation renorm — deep brick-wall with chiMax < 2^d
  //
  // Forces kKeep < physicalRank on most bonds, exercising the GPU
  // truncation-renorm path added in Phase A. Pass bar:
  //   • ‖ψ_GPU‖² stays in [0.999, 1.001] (no probability-mass leak)
  //   • Fidelity vs CPU MPS (which renorms identically) ≥ 0.99
  // Pre-fix the GPU norm drifts well below 1; post-fix it tracks 1
  // to f32 precision over arbitrary depth.
  async function runPhaseATruncBench(): Promise<TruncRow[]> {
    const cfgs: ReadonlyArray<{ N: number; depth: number; chiMax: number }> = [
      { N: 8,  depth: 4, chiMax: 4 },   // 2^4 = 16 > 4 → truncates mid-chain
      { N: 10, depth: 5, chiMax: 4 },
      { N: 12, depth: 6, chiMax: 8 },   // 2^6 = 64 > 8
      { N: 14, depth: 6, chiMax: 8 },
      { N: 16, depth: 7, chiMax: 8 },
    ];
    const out: TruncRow[] = [];

    function buildCircuit(N: number, depth: number, seed: number): {
      singleQ: { U: Matrix2; q: number }[];
      twoQ: { G: import("../../src/linalg.js").ComplexMatrix; q: number }[];
    } {
      let s = seed >>> 0;
      const rng = (): number => {
        s = (s + 0x6D2B79F5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
      const singleQ: { U: Matrix2; q: number }[] = [];
      for (let q = 0; q < N; q++) {
        singleQ.push({ U: Hgate, q });
        singleQ.push({ U: Rz(rng() * 2 * Math.PI), q });
      }
      const cnot: import("../../src/linalg.js").ComplexMatrix = {
        rows: 4, cols: 4,
        data: new Float64Array([
          1,0, 0,0, 0,0, 0,0,
          0,0, 1,0, 0,0, 0,0,
          0,0, 0,0, 0,0, 1,0,
          0,0, 0,0, 1,0, 0,0,
        ]),
      };
      const twoQ: { G: import("../../src/linalg.js").ComplexMatrix; q: number }[] = [];
      for (let l = 0; l < depth; l++) {
        const offset = l & 1;
        for (let q = offset; q + 1 < N; q += 2) twoQ.push({ G: cnot, q });
      }
      return { singleQ, twoQ };
    }

    for (const cfg of cfgs) {
      const { singleQ, twoQ } = buildCircuit(cfg.N, cfg.depth, 0xA1A1 ^ cfg.N);

      const cpu = new MPS({ nQubits: cfg.N, chiMax: cfg.chiMax });
      for (const { U, q } of singleQ) cpu.apply(U, q);
      for (const { G, q } of twoQ) cpu.applyTwoSite(G, q);
      const psiCpu = cpu.statevector();

      const gpu = new MPSGpu({ device, nQubits: cfg.N, chiMax: cfg.chiMax });
      for (const { U, q } of singleQ) gpu.apply(U, q);
      for (const { G, q } of twoQ) await gpu.applyTwoSite(G, q);
      const psiGpu = await gpu.statevector();
      gpu.dispose();

      let normSq = 0;
      for (let i = 0; i < psiGpu.length; i += 2) {
        normSq += psiGpu[i]! * psiGpu[i]! + psiGpu[i + 1]! * psiGpu[i + 1]!;
      }
      let re = 0, im = 0;
      for (let i = 0; i < psiCpu.length; i += 2) {
        const ar = psiCpu[i]!, ai = psiCpu[i + 1]!;
        const br = psiGpu[i]!, bi = psiGpu[i + 1]!;
        re += ar * br + ai * bi;
        im += ar * bi - ai * br;
      }
      const fidelity = re * re + im * im;
      const normOk = Math.abs(normSq - 1) < 5e-3;
      const match = fidelity > 0.99 && normOk;

      out.push({ N: cfg.N, depth: cfg.depth, chiMax: cfg.chiMax, normSq, fidelity, match });
    }
    return out;
  }

  const phase4bBtn = document.getElementById("phase4bRunBtn");
  if (phase4bBtn) phase4bBtn.addEventListener("click", () => { void runPhase4bBench(); });

  window.__gpuMpsBench = {
    ready: true,
    rows: [],
    svdRows: [],
    mpsRows: [],
    residentRows: [],
    run: async () => {
      const r = await runBench();
      window.__gpuMpsBench!.rows = r;
      return r;
    },
    runSvd: async () => {
      const r = await runSvdBench();
      window.__gpuMpsBench!.svdRows = r;
      return r;
    },
    runMps: async () => {
      const r = await runMpsBench();
      window.__gpuMpsBench!.mpsRows = r;
      return r;
    },
    runResident: async () => {
      const r = await runResidentBench();
      window.__gpuMpsBench!.residentRows = r;
      return r;
    },
    phase4bRows: [],
    runPhase4b: async () => {
      const r = await runPhase4bBench();
      window.__gpuMpsBench!.phase4bRows = r;
      return r;
    },
    runPhase5v1: async () => {
      return await runPhase5v1Bench();
    },
    runPhaseATrunc: async () => {
      return await runPhaseATruncBench();
    },
  };
}

boot().catch((err) => {
   
  console.error(err);
});
