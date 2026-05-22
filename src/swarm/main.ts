// /swarm.html entry point — wires the swarm runtime into a tiny demo
// that counts primes in [0, N) across a tile partition. Pure CPU,
// embarrassingly parallel; the kernel is intentionally trivial so the
// distribution mechanics show through.

import { BroadcastChannelTransport } from "../parallel/swarm/broadcast-transport.js";
import { attachSwarmRuntime, swarmMap } from "../parallel/swarm/swarm-map.js";

interface PrimeTile { readonly lo: number; readonly hi: number; }

function countPrimes(lo: number, hi: number): number {
  // Trial division (good enough for demo; not for benchmarks).
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

// ── Wire up the page. ──
const peersEl = document.getElementById("peers") as HTMLElement;
const logEl   = document.getElementById("log") as HTMLPreElement;
const runBtn  = document.getElementById("runBtn") as HTMLButtonElement;
const Nin     = document.getElementById("N") as HTMLInputElement;
const tilesIn = document.getElementById("tiles") as HTMLInputElement;
const progEl  = document.getElementById("progress") as HTMLElement;
const progTxt = document.getElementById("progressText") as HTMLElement;
const progFill = document.getElementById("progressFill") as HTMLElement;

function log(msg: string): void {
  const ts = new Date().toLocaleTimeString();
  logEl.textContent += `[${ts}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

const transport = new BroadcastChannelTransport();
const swarm = attachSwarmRuntime(transport);

swarm.registerKernel<PrimeTile, number>("primes-tile", (tile) => {
  return countPrimes(tile.lo, tile.hi);
});

log(`This tab's peer id: ${transport.self}`);
log(`Kernel "primes-tile" registered. Open more tabs at this URL to grow the swarm.`);

// Periodically refresh the peers panel.
function refreshPeers(): void {
  const others = swarm.knownPeers();
  const parts: string[] = [`<span class="peer self">self · ${transport.self}</span>`];
  for (const p of others) parts.push(`<span class="peer">${p}</span>`);
  peersEl.innerHTML = parts.join("");
}
window.setInterval(refreshPeers, 800);
refreshPeers();

// ── Run button: dispatch a job. ──
runBtn.addEventListener("click", async () => {
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

  log(`Distributing ${T} tiles covering [0, ${N.toLocaleString()}) across the swarm…`);
  const t0 = performance.now();
  try {
    const results = await swarmMap<PrimeTile, number>(swarm, "primes-tile", tiles, {
      onProgress: (done, total) => {
        progTxt.textContent = `${done} / ${total} tiles done`;
        progFill.style.width = `${(100 * done / total).toFixed(1)}%`;
      },
    });
    const total = results.reduce((s, x) => s + x, 0);
    const t1 = performance.now();
    log(`✓ Total primes in [0, ${N.toLocaleString()}) = ${total.toLocaleString()} (${(t1 - t0).toFixed(0)} ms wall, ${T} tiles)`);
  } catch (e) {
    log(`✗ Job failed: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    runBtn.disabled = false;
  }
});
