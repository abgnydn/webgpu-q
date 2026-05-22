// /swarm.html entry point — wires the swarm runtime into a tiny demo
// that counts primes in [0, N) across a tile partition. Pure CPU,
// embarrassingly parallel; the kernel is intentionally trivial so the
// distribution mechanics show through.
//
// Two transports are available: BroadcastChannel (same-machine tabs) and
// WebRTC (cross-machine via PeerJS). User picks via the radio buttons.

import { BroadcastChannelTransport } from "../parallel/swarm/broadcast-transport.js";
import { WebRTCTransport } from "../parallel/swarm/webrtc-transport.js";
import { attachSwarmRuntime, swarmMap } from "../parallel/swarm/swarm-map.js";
import type { SwarmTransport } from "../parallel/swarm/transport.js";
import {
  CHEM_ENERGY_KIND, runChemEnergyTile, bondScanTiles,
  type ChemEnergyTile, type ChemEnergyResult,
} from "./chemistry-kernel.js";

interface PrimeTile { readonly lo: number; readonly hi: number; }

function countPrimes(lo: number, hi: number): number {
  let count = 0;
  for (let n = Math.max(lo, 2); n < hi; n++) {
    let prime = true;
    const r = Math.sqrt(n) | 0;
    for (let d = 2; d <= r; d++) {
      if (n % d === 0) { prime = false; break; }
    }
    if (prime) count++;
  }
  return count;
}

const peersEl   = document.getElementById("peers") as HTMLElement;
const logEl     = document.getElementById("log") as HTMLPreElement;
const runBtn    = document.getElementById("runBtn") as HTMLButtonElement;
const Nin       = document.getElementById("N") as HTMLInputElement;
const tilesIn   = document.getElementById("tiles") as HTMLInputElement;
const progEl    = document.getElementById("progress") as HTMLElement;
const progTxt   = document.getElementById("progressText") as HTMLElement;
const progFill  = document.getElementById("progressFill") as HTMLElement;
const rtcCtrls  = document.getElementById("webrtc-controls") as HTMLElement;
const selfIdEl  = document.getElementById("self-id") as HTMLElement;
const copyIdBtn = document.getElementById("copy-id") as HTMLButtonElement;
const copyLinkBtn = document.getElementById("copy-link") as HTMLButtonElement;
const remoteIdInput = document.getElementById("remote-id") as HTMLInputElement;
const connectBtn = document.getElementById("connect-btn") as HTMLButtonElement;

function log(msg: string): void {
  const ts = new Date().toLocaleTimeString();
  logEl.textContent += `[${ts}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

interface ActiveSwarm {
  transport: SwarmTransport;
  registry: ReturnType<typeof attachSwarmRuntime>;
  kind: "bc" | "webrtc";
}
let active: ActiveSwarm | null = null;

function teardown(): void {
  if (!active) return;
  try { active.transport.close(); } catch { /* ignore */ }
  active = null;
}

function activateBC(): void {
  teardown();
  const transport = new BroadcastChannelTransport();
  const registry = attachSwarmRuntime(transport);
  registry.registerKernel<PrimeTile, number>("primes-tile", (tile) =>
    countPrimes(tile.lo, tile.hi));
  registry.registerKernel<ChemEnergyTile, ChemEnergyResult>(CHEM_ENERGY_KIND, (tile) =>
    runChemEnergyTile(tile));
  active = { transport, registry, kind: "bc" };
  rtcCtrls.style.display = "none";
  log(`BroadcastChannel transport active. Peer id: ${transport.self}`);
  log(`Open this URL in more tabs to grow the swarm.`);
}

async function activateWebRTC(): Promise<void> {
  teardown();
  const transport = new WebRTCTransport({
    onReady: (id) => {
      selfIdEl.textContent = id;
      log(`WebRTC peer ready. Your ID: ${id}`);
      // If the URL has ?join=<otherId>, auto-connect.
      const params = new URLSearchParams(window.location.search);
      const joinTarget = params.get("join");
      if (joinTarget && joinTarget !== id) {
        log(`?join=${joinTarget} — auto-connecting…`);
        transport.connectTo(joinTarget);
      }
    },
    onPeerConnected: (id) => log(`Peer connected: ${id}`),
    onPeerDisconnected: (id) => log(`Peer disconnected: ${id}`),
    onError: (err) => log(`[error] ${err.message}`),
  });
  rtcCtrls.style.display = "block";
  selfIdEl.textContent = "— connecting —";
  try {
    await transport.openAsync();
  } catch (e) {
    log(`Failed to open WebRTC: ${e instanceof Error ? e.message : String(e)}`);
    return;
  }
  const registry = attachSwarmRuntime(transport);
  registry.registerKernel<PrimeTile, number>("primes-tile", (tile) =>
    countPrimes(tile.lo, tile.hi));
  registry.registerKernel<ChemEnergyTile, ChemEnergyResult>(CHEM_ENERGY_KIND, (tile) =>
    runChemEnergyTile(tile));
  active = { transport, registry, kind: "webrtc" };
}

// Transport radio buttons.
for (const radio of document.querySelectorAll<HTMLInputElement>("input[name=transport]")) {
  radio.addEventListener("change", () => {
    if (!radio.checked) return;
    if (radio.value === "bc") activateBC();
    else void activateWebRTC();
  });
}
// Default to BC.
activateBC();

// WebRTC manual connect.
connectBtn.addEventListener("click", () => {
  if (!active || active.kind !== "webrtc") return;
  const otherId = remoteIdInput.value.trim();
  if (!otherId) return;
  log(`Connecting to ${otherId}…`);
  (active.transport as WebRTCTransport).connectTo(otherId);
  remoteIdInput.value = "";
});
remoteIdInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") connectBtn.click();
});

copyIdBtn.addEventListener("click", async () => {
  const id = selfIdEl.textContent ?? "";
  if (!id || id.startsWith("—")) return;
  try { await navigator.clipboard.writeText(id); log(`Copied peer ID: ${id}`); }
  catch { /* user can click the badge */ }
});
copyLinkBtn.addEventListener("click", async () => {
  const id = selfIdEl.textContent ?? "";
  if (!id || id.startsWith("—")) return;
  const url = `${window.location.origin}${window.location.pathname}?join=${encodeURIComponent(id)}&transport=webrtc`;
  try { await navigator.clipboard.writeText(url); log(`Copied join link: ${url}`); }
  catch { /* user can pick up from address bar */ }
});

// Auto-switch to WebRTC if URL says so.
{
  const params = new URLSearchParams(window.location.search);
  if (params.get("transport") === "webrtc") {
    const r = document.querySelector<HTMLInputElement>("input[name=transport][value=webrtc]");
    if (r) { r.checked = true; void activateWebRTC(); }
  }
}

// Peer-roster refresh.
function refreshPeers(): void {
  if (!active) return;
  const others = active.registry.knownPeers();
  const parts: string[] = [
    `<span class="peer self">self · ${active.transport.self || "—"}</span>`,
  ];
  for (const p of others) parts.push(`<span class="peer">${p}</span>`);
  peersEl.innerHTML = parts.join("");
}
window.setInterval(refreshPeers, 800);
refreshPeers();

// Demo 2 — H₂ bond-length scan.
const runScanBtn = document.getElementById("runScanBtn") as HTMLButtonElement;
const rMinIn  = document.getElementById("rMin") as HTMLInputElement;
const rMaxIn  = document.getElementById("rMax") as HTMLInputElement;
const nPtsIn  = document.getElementById("nPoints") as HTMLInputElement;
const scanProgEl   = document.getElementById("scanProgress") as HTMLElement;
const scanProgTxt  = document.getElementById("scanProgressText") as HTMLElement;
const scanProgFill = document.getElementById("scanProgressFill") as HTMLElement;
const scanResultEl = document.getElementById("scanResult") as HTMLElement;

runScanBtn.addEventListener("click", async () => {
  if (!active) { log("No transport active."); return; }
  const rMin = Math.max(0.1, Number(rMinIn.value));
  const rMax = Math.max(rMin + 0.05, Number(rMaxIn.value));
  const N    = Math.max(3, Math.min(100, Math.floor(Number(nPtsIn.value))));
  runScanBtn.disabled = true;
  scanProgEl.style.display = "block";
  scanProgTxt.textContent = `0 / ${N} points done`;
  scanProgFill.style.width = "0%";
  scanResultEl.textContent = "";

  const tiles = bondScanTiles({
    atomA: "H", atomB: "H", basis: "sto-3g",
    rMin, rMax, nPoints: N,
  });

  log(`Distributing H₂ scan (r ∈ [${rMin}, ${rMax}] Å, ${N} points) over ${active.kind.toUpperCase()}…`);
  const t0 = performance.now();
  try {
    const results = await swarmMap<ChemEnergyTile, ChemEnergyResult>(
      active.registry, CHEM_ENERGY_KIND, tiles, {
        // SCF takes a couple ms per point; 200ms claim window plenty.
        claimTimeoutMs: 200,
        onProgress: (done, total) => {
          scanProgTxt.textContent = `${done} / ${total} points done`;
          scanProgFill.style.width = `${(100 * done / total).toFixed(1)}%`;
        },
      });
    const t1 = performance.now();
    let minI = 0;
    for (let i = 1; i < results.length; i++) {
      if (results[i]!.energy < results[minI]!.energy) minI = i;
    }
    const rEq = rMin + minI * (rMax - rMin) / (N - 1);
    log(`✓ Scan complete in ${(t1 - t0).toFixed(0)} ms. Equilibrium r ≈ ${rEq.toFixed(3)} Å, E_min = ${results[minI]!.energy.toFixed(6)} Ha`);
    // Render an ASCII curve.
    const eMin = Math.min(...results.map(r => r.energy));
    const eMax = Math.max(...results.map(r => r.energy));
    const W = 50;
    const lines = results.map((r, i) => {
      const rA = rMin + i * (rMax - rMin) / (N - 1);
      const col = Math.round((r.energy - eMin) / (eMax - eMin + 1e-12) * W);
      const bar = "·".repeat(col) + "▼" + " ".repeat(Math.max(0, W - col));
      return `r=${rA.toFixed(3)} ${bar} ${r.energy.toFixed(6)} Ha`;
    });
    scanResultEl.textContent = lines.join("\n");
  } catch (e) {
    log(`✗ Scan failed: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    runScanBtn.disabled = false;
  }
});

// Run button: dispatch a job.
runBtn.addEventListener("click", async () => {
  if (!active) { log("No transport active."); return; }
  const N = Math.max(1000, Math.floor(Number(Nin.value)));
  const T = Math.max(1, Math.min(256, Math.floor(Number(tilesIn.value))));
  runBtn.disabled = true;
  progEl.style.display = "block";
  progTxt.textContent = `0 / ${T} tiles done`;
  progFill.style.width = "0%";

  const tiles: PrimeTile[] = [];
  const base = Math.floor(N / T);
  let cursor = 0;
  for (let i = 0; i < T; i++) {
    const lo = cursor;
    const hi = i === T - 1 ? N : lo + base;
    tiles.push({ lo, hi });
    cursor = hi;
  }

  log(`Distributing ${T} tiles covering [0, ${N.toLocaleString()}) over ${active.kind.toUpperCase()}…`);
  const t0 = performance.now();
  try {
    const results = await swarmMap<PrimeTile, number>(active.registry, "primes-tile", tiles, {
      onProgress: (done, total) => {
        progTxt.textContent = `${done} / ${total} tiles done`;
        progFill.style.width = `${(100 * done / total).toFixed(1)}%`;
      },
    });
    const total = results.reduce((s, x) => s + x, 0);
    const t1 = performance.now();
    log(`✓ Total primes in [0, ${N.toLocaleString()}) = ${total.toLocaleString()} (${(t1 - t0).toFixed(0)} ms wall, ${T} tiles, ${active.kind.toUpperCase()})`);
  } catch (e) {
    log(`✗ Job failed: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    runBtn.disabled = false;
  }
});
