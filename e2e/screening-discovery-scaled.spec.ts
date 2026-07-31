import { test, expect } from "@playwright/test";

// SCALED DISCOVERY — exhaustive combinatorial screen across the SWARM.
//
// Scale bar ② from 10 cherry-picked isomers to the whole chemically-sensible
// subspace: EVERY placement of up to 3 nitrogens on a 6-atom conjugated chain
// (all mono/di/tri-aza isomers = 42), distributed across two BroadcastChannel tabs
// through the greedy-pull swarm. No human bias in candidate selection — we screen
// the space and let the champions emerge. (Capping at ≤3 N keeps the isomers real
// — exotic all-N chains don't converge — and the sweep finishes in-browser; the
// per-molecule HF is the wall, not the swarm, so a 6-atom chain is the sweet spot.)
// All isomers are isoelectronic, so gap differences are purely structural. Then we
// ask: do the top chromophores share the azo motif (a double-bonded N=N) — the
// chromophore behind most commercial dyes? HF/STO-3G ⇒ a triage landscape, not
// proven hits. Single machine, two tabs. Non-converged candidates are dropped (an
// honest screen reports what computed, not what it wished).
//
// MEASURED (2026-06-15): 42/42 converged in ~194 s, work split 23/19 across the
// two tabs (the greedy-pull scheduler balancing real work). Population signal: the
// azo (N=N) motif is 36% of the library but ~80% of the top 15 → ~2.2× enriched —
// the screen tracks the real dye chemistry. BUT the two smallest-gap hits
// (2,3,5- and 2,4,5-triaza, ~2.4 eV) are NON-azo and sit ~3× below the azo pack —
// anomalously small for minimal-basis HF, the classic fingerprint of an RHF
// instability (near-diradical character), i.e. likely an ARTIFACT to re-examine
// with UHF/multireference, not a trusted lead. That is the discipline a screen
// teaches: it surfaces both leads and traps; you scrutinize the top, never trust
// the single best number blindly.

type Atom = { symbol: string; pos: [number, number, number] };
const norm = (v: number[]): number[] => { const n = Math.hypot(v[0]!, v[1]!, v[2]!); return [v[0]! / n, v[1]! / n, v[2]! / n]; };
const K = 6;                          // 6-atom chain — fast enough to finish the full ≤3-N sweep
const MAX_N = 3;                      // ≤3 nitrogens → chemically sensible isomers (C(6,0..3)=42)

function azaChain(nAt: ReadonlySet<number>, k = K): Atom[] {
  const dDouble = 1.34, dSingle = 1.46, dXH = 1.05, aHalf = (28 * Math.PI) / 180;
  const P: [number, number, number][] = [[0, 0, 0]];
  for (let i = 0; i < k - 1; i++) {
    const theta = i % 2 === 0 ? aHalf : -aHalf;
    const d = i % 2 === 0 ? dDouble : dSingle;
    const p = P[i]!;
    P.push([p[0] + d * Math.cos(theta), p[1] + d * Math.sin(theta), 0]);
  }
  const atoms: Atom[] = P.map((p, i) => ({ symbol: nAt.has(i) ? "N" : "C", pos: p }));
  for (let i = 0; i < k; i++) {
    const ci = P[i]!, isN = nAt.has(i), terminal = i === 0 || i === k - 1;
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
const labelOf = (nAt: number[]): string => (nAt.length ? nAt.map((i) => i + 1).join(",") + "-aza" : "all-C");
const hasAzo = (s: Set<number>): boolean => { for (let i = 0; i < K - 1; i += 2) if (s.has(i) && s.has(i + 1)) return true; return false; };

const LIB = Array.from({ length: 1 << K }, (_, m) => m)
  .filter((m) => popcount(m) <= MAX_N)
  .map((m) => {
    const nAt = new Set<number>();
    for (let b = 0; b < K; b++) if (m & (1 << b)) nAt.add(b);
    return { label: labelOf([...nAt]), atoms: azaChain(nAt), azo: hasAzo(nAt), nN: nAt.size };
  });

test.describe("scaled discovery — exhaustive aza-chain screen across the swarm", () => {
  test(`rank all ${LIB.length} ≤${MAX_N}-aza ${K}-atom chains; do the top chromophores share the azo motif?`, async ({ browser }) => {
    test.setTimeout(600 * 1000);
    const ctx = await browser.newContext();
    const a = await ctx.newPage(); const b = await ctx.newPage();
    a.on("pageerror", (e) => console.error(`[a] ${e.message}`)); b.on("pageerror", (e) => console.error(`[b] ${e.message}`));
    a.on("console", (m) => { if (m.text().startsWith("[scaled]")) console.log(m.text()); });
    await a.goto("/swarm.html", { waitUntil: "domcontentloaded" });
    await b.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    await b.evaluate(async () => {
      const [{ attachSwarmRuntime }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("scaled-screen"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => runChemEnergyTile(tile as never));
    });
    await a.waitForTimeout(2000);

    const res = await a.evaluate(async (lib) => {
      const [{ attachSwarmRuntime, swarmMap }, { BroadcastChannelTransport }, { runChemEnergyTile, CHEM_ENERGY_KIND }] = await Promise.all([
        import("/src/parallel/swarm/swarm-map.ts" as string),
        import("/src/parallel/swarm/broadcast-transport.ts" as string),
        import("/src/swarm/chemistry-kernel.ts" as string),
      ]);
      const log = (s: string): void => { console.log(`[scaled] ${s}`); };
      let mine = 0;
      const reg = attachSwarmRuntime(new BroadcastChannelTransport("scaled-screen"));
      reg.registerKernel(CHEM_ENERGY_KIND, async (tile: unknown) => { mine++; return runChemEnergyTile(tile as never); });

      const tiles = lib.map((m) => ({ label: m.label, atoms: m.atoms as never, basis: "sto-3g" as const, method: "hf" as const }));
      const t0 = performance.now();
      const out = await swarmMap(reg, CHEM_ENERGY_KIND, tiles, { claimTimeoutMs: 300 }) as { label: string; homoLumoGapEv: number; converged: boolean; nElectrons: number }[];
      const wall = performance.now() - t0;

      const all = out.map((r, i) => ({ label: r.label, gap: r.homoLumoGapEv, conv: r.converged, nE: r.nElectrons, azo: lib[i]!.azo }));
      const rows = all.filter((r) => r.conv && Number.isFinite(r.gap) && r.gap > 0).sort((x, y) => x.gap - y.gap);
      const TOP = 15;
      const azoAll = rows.filter((r) => r.azo).length / rows.length;
      const azoTop = rows.slice(0, TOP).filter((r) => r.azo).length / TOP;
      log(`screened ${out.length} isomers in ${(wall / 1000).toFixed(0)} s (${rows.length} converged; this tab ran ${mine}, other ${out.length - mine})`);
      log(`gap range ${rows[0]!.gap.toFixed(2)} … ${rows[rows.length - 1]!.gap.toFixed(2)} eV`);
      log(`TOP ${TOP} chromophores (smallest gap):`);
      rows.slice(0, TOP).forEach((r, i) => log(`  ${String(i + 1).padStart(2)}. ${r.label.padEnd(16)} ${r.gap.toFixed(2)} eV  ${r.azo ? "azo(N=N)" : ""}`));
      log(`azo motif: ${(100 * azoAll).toFixed(0)}% of converged isomers, but ${(100 * azoTop).toFixed(0)}% of the top ${TOP} → enrichment ${(azoTop / Math.max(azoAll, 1e-9)).toFixed(2)}×`);
      return { total: out.length, conv: rows.length, mine, allIso: rows.every((r) => r.nE === rows[0]!.nE),
        minGap: rows[0]!.gap, maxGap: rows[rows.length - 1]!.gap, azoAll, azoTop, wallS: wall / 1000,
        topAzoCount: rows.slice(0, TOP).filter((r) => r.azo).length };
    }, LIB);
console.log(`\n[scaled-discovery] ${res.conv}/${res.total} converged in ${res.wallS.toFixed(0)}s · gap ${res.minGap.toFixed(1)}–${res.maxGap.toFixed(1)} eV · azo top-15 ${res.topAzoCount}/15 · enrichment ${(res.azoTop / Math.max(res.azoAll, 1e-9)).toFixed(2)}x\n`);

    expect(res.total).toBe(LIB.length);
    expect(res.conv).toBeGreaterThan(LIB.length * 0.8);             // most candidates compute
    expect(res.allIso).toBe(true);                                  // isoelectronic = fair compare
    expect(res.mine).toBeGreaterThan(0); expect(res.mine).toBeLessThan(res.total); // work split across tabs
    expect(res.maxGap - res.minGap).toBeGreaterThan(1.0);           // real structure–property spread
    expect(res.azoTop).toBeGreaterThan(res.azoAll);                 // the screen ENRICHES for the real motif

    await ctx.close();
  });
});
