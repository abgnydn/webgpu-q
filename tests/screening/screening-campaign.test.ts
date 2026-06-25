// Screening campaign — run-and-record. The e2e specs (screening-validation-polyene,
// screening-discovery-azahexatriene, screening-discovery-scaled) DEFINE this campaign
// but run in the browser and never commit a result artifact. This runs the identical
// chemistry in Node (the chem-energy kernel is engine-independent — WASM here, WASM in
// the tab) and writes the first committed artifact under experiments/results/.
//
// Two bars, the project's screening discipline:
//   ① VALIDATION — does HF/STO-3G reproduce a KNOWN trend? (polyene gap shrinks with
//      conjugation length: the particle-in-a-longer-box effect). If yes, trust the filter.
//   ② DISCOVERY — a BLIND exhaustive screen: every ≤3-N placement on a 6-atom conjugated
//      chain (42 isoelectronic isomers). Rank by gap; ask whether the top chromophores
//      enrich for the azo (N=N) dye motif. The rank IS the result; we report leads + traps.
//
// Gated behind SCREENING_CAMPAIGN=1 so the default `npm test` stays fast (it SKIPS
// here); run with:
//   SCREENING_CAMPAIGN=1 npx vitest run tests/screening/screening-campaign.test.ts
// Runtime ≈ 17 min single-threaded in Node — exact-ERI HF on these n≈36–62 molecules
// is genuinely heavy (decapentaene n=62 alone is ~107 s). That cost is precisely the
// motivation for the browser-tab swarm, which runs the same 42-isomer screen in ~194 s
// across two GPU-assisted tabs. Hence the long per-test timeout below.

import { describe, test, expect } from "vitest";
import { mkdirSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { runChemEnergyTile } from "../../src/swarm/chemistry-kernel.js";
import type { Atom } from "../../src/chemistry/atoms.js";

const norm = (v: readonly number[]): [number, number, number] => {
  const n = Math.hypot(v[0]!, v[1]!, v[2]!);
  return [v[0]! / n, v[1]! / n, v[2]! / n];
};

/** All-trans linear polyene CₖHₖ₊₂ (k even → closed shell), planar zigzag. */
function polyene(k: number): Atom[] {
  const dDouble = 1.34, dSingle = 1.46, dCH = 1.09, aHalf = (28 * Math.PI) / 180;
  const C: [number, number, number][] = [[0, 0, 0]];
  for (let i = 0; i < k - 1; i++) {
    const theta = i % 2 === 0 ? aHalf : -aHalf;
    const d = i % 2 === 0 ? dDouble : dSingle;
    const p = C[i]!;
    C.push([p[0] + d * Math.cos(theta), p[1] + d * Math.sin(theta), 0]);
  }
  const atoms: Atom[] = C.map((p): Atom => ({ symbol: "C", pos: p }));
  for (let i = 0; i < k; i++) {
    const ci = C[i]!;
    if (i === 0 || i === k - 1) {
      const nb = i === 0 ? C[1]! : C[k - 2]!;
      const u = norm([nb[0] - ci[0], nb[1] - ci[1], 0]);
      for (const s of [1, -1]) {
        const phi = (s * 120 * Math.PI) / 180;
        const hx = u[0] * Math.cos(phi) - u[1] * Math.sin(phi);
        const hy = u[0] * Math.sin(phi) + u[1] * Math.cos(phi);
        atoms.push({ symbol: "H", pos: [ci[0] + dCH * hx, ci[1] + dCH * hy, 0] });
      }
    } else {
      const a = C[i - 1]!, b = C[i + 1]!;
      const ua = norm([a[0] - ci[0], a[1] - ci[1], 0]);
      const ub = norm([b[0] - ci[0], b[1] - ci[1], 0]);
      const bis = norm([ua[0] + ub[0], ua[1] + ub[1], 0]);
      atoms.push({ symbol: "H", pos: [ci[0] - dCH * bis[0], ci[1] - dCH * bis[1], 0] });
    }
  }
  return atoms;
}

const K = 6, MAX_N = 3;
/** 6-atom all-trans conjugated chain; positions in `nAt` (0-based) become N
 *  (terminal N keeps 1 H, interior N keeps 0). C↔N this way is electron-neutral,
 *  so every isomer is the same closed-shell 44-e⁻ species — a controlled study. */
function azaChain(nAt: ReadonlySet<number>, k = K): Atom[] {
  const dDouble = 1.34, dSingle = 1.46, dXH = 1.05, aHalf = (28 * Math.PI) / 180;
  const P: [number, number, number][] = [[0, 0, 0]];
  for (let i = 0; i < k - 1; i++) {
    const theta = i % 2 === 0 ? aHalf : -aHalf;
    const d = i % 2 === 0 ? dDouble : dSingle;
    const p = P[i]!;
    P.push([p[0] + d * Math.cos(theta), p[1] + d * Math.sin(theta), 0]);
  }
  const atoms: Atom[] = P.map((p, i): Atom => ({ symbol: nAt.has(i) ? "N" : "C", pos: p }));
  for (let i = 0; i < k; i++) {
    const ci = P[i]!, isN = nAt.has(i), terminal = i === 0 || i === k - 1;
    const nH = terminal ? (isN ? 1 : 2) : (isN ? 0 : 1);
    if (nH === 0) continue;
    if (terminal) {
      const nb = i === 0 ? P[1]! : P[k - 2]!;
      const u = norm([nb[0] - ci[0], nb[1] - ci[1], 0]);
      for (const s of nH === 2 ? [1, -1] : [1]) {
        const phi = (s * 120 * Math.PI) / 180;
        atoms.push({ symbol: "H", pos: [ci[0] + dXH * (u[0] * Math.cos(phi) - u[1] * Math.sin(phi)), ci[1] + dXH * (u[0] * Math.sin(phi) + u[1] * Math.cos(phi)), 0] });
      }
    } else {
      const a = P[i - 1]!, b = P[i + 1]!;
      const ua = norm([a[0] - ci[0], a[1] - ci[1], 0]), ub = norm([b[0] - ci[0], b[1] - ci[1], 0]);
      const bis = norm([ua[0] + ub[0], ua[1] + ub[1], 0]);
      atoms.push({ symbol: "H", pos: [ci[0] - dXH * bis[0], ci[1] - dXH * bis[1], 0] });
    }
  }
  return atoms;
}

const popcount = (m: number): number => { let c = 0, x = m; while (x) { c += x & 1; x >>= 1; } return c; };
const labelOf = (nAt: number[]): string => (nAt.length ? nAt.map((i) => i + 1).join(",") + "-aza" : "all-C");
const hasAzo = (s: ReadonlySet<number>): boolean => { for (let i = 0; i < K - 1; i += 2) if (s.has(i) && s.has(i + 1)) return true; return false; };

const SERIES = [
  { name: "ethylene", k: 2 }, { name: "butadiene", k: 4 }, { name: "hexatriene", k: 6 },
  { name: "octatetraene", k: 8 }, { name: "decapentaene", k: 10 },
].map((m) => ({ name: m.name, nDouble: m.k / 2, atoms: polyene(m.k) }));

const LIB = Array.from({ length: 1 << K }, (_, m) => m)
  .filter((m) => popcount(m) <= MAX_N)
  .map((m) => {
    const nAt = new Set<number>();
    for (let b = 0; b < K; b++) if (m & (1 << b)) nAt.add(b);
    return { label: labelOf([...nAt]), atoms: azaChain(nAt), azo: hasAzo(nAt), nN: nAt.size };
  });

const RUN = process.env.SCREENING_CAMPAIGN === "1";

describe("screening campaign — validation + exhaustive aza discovery (Node)", () => {
  (RUN ? test : test.skip)("run the campaign and write the result artifact", async () => {
    // ── BAR ① validation: a KNOWN trend (gap shrinks with conjugation length) ──
    const validation: { name: string; nDouble: number; gapEv: number; converged: boolean; ms: number }[] = [];
    for (const m of SERIES) {
      const r = await runChemEnergyTile({ label: m.name, atoms: m.atoms, basis: "sto-3g", method: "hf" });
      validation.push({ name: m.name, nDouble: m.nDouble, gapEv: r.homoLumoGapEv, converged: r.converged, ms: Math.round(r.durationMs) });
    }
    let monotonic = true;
    for (let i = 1; i < validation.length; i++) if (!(validation[i]!.gapEv < validation[i - 1]!.gapEv)) monotonic = false;
    const validationDrop = validation[0]!.gapEv - validation[validation.length - 1]!.gapEv;
    console.log("\n[campaign] ── VALIDATION: HOMO–LUMO gap vs conjugation length ──");
    for (const r of validation) console.log(`  ${r.name.padEnd(14)} C=C ${String(r.nDouble).padStart(2)}   ${r.gapEv.toFixed(2)} eV`);
    console.log(`  monotonic decrease: ${monotonic ? "YES" : "NO"}  (drop ${validationDrop.toFixed(2)} eV)`);

    // ── BAR ② discovery: blind exhaustive aza screen ──
    const discovery: { label: string; gapEv: number; converged: boolean; nE: number; azo: boolean; nN: number; ms: number }[] = [];
    for (const c of LIB) {
      const r = await runChemEnergyTile({ label: c.label, atoms: c.atoms, basis: "sto-3g", method: "hf" });
      discovery.push({ label: c.label, gapEv: r.homoLumoGapEv, converged: r.converged, nE: r.nElectrons, azo: c.azo, nN: c.nN, ms: Math.round(r.durationMs) });
    }
    const ranked = discovery.filter((r) => r.converged && Number.isFinite(r.gapEv) && r.gapEv > 0).sort((a, b) => a.gapEv - b.gapEv);
    const TOP = 15;
    const azoAll = ranked.filter((r) => r.azo).length / ranked.length;
    const azoTop = ranked.slice(0, TOP).filter((r) => r.azo).length / TOP;
    const enrichment = azoTop / Math.max(azoAll, 1e-9);
    console.log("\n[campaign] ── DISCOVERY: exhaustive ≤3-N aza-hexatriene screen ──");
    console.log(`  ${ranked.length}/${discovery.length} converged · gap ${ranked[0]!.gapEv.toFixed(2)}…${ranked[ranked.length - 1]!.gapEv.toFixed(2)} eV`);
    ranked.slice(0, TOP).forEach((r, i) => console.log(`   ${String(i + 1).padStart(2)}. ${r.label.padEnd(14)} ${r.gapEv.toFixed(2)} eV  ${r.azo ? "azo(N=N)" : ""}`));
    console.log(`  azo motif: ${(100 * azoAll).toFixed(0)}% of library → ${(100 * azoTop).toFixed(0)}% of top ${TOP} = ${enrichment.toFixed(2)}× enriched`);

    // ── artifact ──
    let gitSha = "unknown";
    try { gitSha = execSync("git rev-parse HEAD").toString().trim(); } catch { /* not a git checkout */ }
    const allIso = ranked.every((r) => r.nE === ranked[0]!.nE);
    const spread = ranked[ranked.length - 1]!.gapEv - ranked[0]!.gapEv;
    const pass = monotonic && validationDrop > 1.0 && ranked.length >= discovery.length * 0.8 && allIso && spread > 1.0 && azoTop > azoAll;
    const artifact = {
      meta: {
        protocol: "screening-campaign",
        hypothesis: "HOMO–LUMO gap (HF/STO-3G) ranks conjugated chromophores: it reproduces the known polyene trend (validation) and, on a blind exhaustive aza-chain screen, enriches the top for the azo (N=N) dye motif (discovery).",
        passBar: "validation: gap monotonically decreasing, drop > 1 eV. discovery: ≥80% converged, isoelectronic, spread > 1 eV, azo-enriched top-15.",
        basis: "sto-3g", method: "rhf",
        seed: "deterministic — fixed library + deterministic SCF, no RNG",
        warmup: 0, trials: 1,
      },
      env: {
        runner: "node-vitest", node: process.version, platform: process.platform, gitSha,
        timestamp: new Date().toISOString(),
        note: "Chemistry is engine-independent: the chem-energy kernel produces the identical ranking on a browser tab (the swarm path) and in Node. The multi-tab wall-clock (cost-aware LPT scheduling, 1.98×) is measured separately; this artifact records the SCIENCE (the ranking).",
      },
      rows: { validation, discoveryRanked: ranked, discoveryRaw: discovery },
      summary: {
        validationMonotonic: monotonic, validationDropEv: validationDrop,
        screened: discovery.length, converged: ranked.length,
        gapMinEv: ranked[0]!.gapEv, gapMaxEv: ranked[ranked.length - 1]!.gapEv, winner: ranked[0]!.label,
        azoFractionAll: azoAll, azoFractionTop15: azoTop, azoEnrichment: enrichment,
      },
      status: pass ? "pass" : "fail",
      diagnosis: pass
        ? `Validated: polyene gap decreases monotonically (drop ${validationDrop.toFixed(2)} eV). Discovery: ${ranked.length}/${discovery.length} converged; winner ${ranked[0]!.label} (${ranked[0]!.gapEv.toFixed(2)} eV); azo motif ${enrichment.toFixed(2)}× enriched in the top ${TOP}. Caveat: the smallest-gap NON-azo hits are likely RHF-instability artifacts (near-diradical) to re-examine with UHF — a screen surfaces both leads and traps, so scrutinize the top, never trust the single best number blindly.`
        : "a pass-bar cell failed — see summary.",
    };
    const dir = `experiments/results/${new Date().toISOString().slice(0, 10)}/screening`;
    mkdirSync(dir, { recursive: true });
    writeFileSync(`${dir}/screening-campaign.json`, JSON.stringify(artifact, null, 2));
    console.log(`\n[campaign] artifact → ${dir}/screening-campaign.json  status=${artifact.status}\n`);

    // ── the bars ──
    expect(validation.every((r) => r.converged)).toBe(true);
    expect(monotonic).toBe(true);                                    // BAR ①
    expect(validationDrop).toBeGreaterThan(1.0);
    expect(ranked.length).toBeGreaterThan(discovery.length * 0.8);   // BAR ② most converge
    expect(allIso).toBe(true);                                       // isoelectronic = fair compare
    expect(spread).toBeGreaterThan(1.0);
    expect(azoTop).toBeGreaterThan(azoAll);                          // the screen enriches the real motif
  }, 1_800_000);
});
