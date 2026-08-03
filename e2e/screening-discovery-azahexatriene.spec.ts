import { test, expect } from "@playwright/test";

// SCREENING DISCOVERY (bar ②) — a BLIND rank where the answer isn't pre-known.
// Runs through the REAL chem-energy kernel in the browser (WASM-fast), single tab.
//
// Campaign: "color-tuning by heteroatom placement." A fixed 6-atom conjugated
// chain (hexatriene skeleton); screen every way of swapping carbons for nitrogens.
// Every candidate is ISOELECTRONIC (44 e⁻), so the gap difference is purely WHERE
// the N sits — the real chemistry that tunes chromophore color (retinal Schiff
// bases in vision). Goal: surface the smallest-gap arrangement (most red-shifted /
// best visible absorber). The ranking is not obvious by eye — it must be computed.
// HF/STO-3G ⇒ a TRIAGE shortlist, not a proven hit. Discovery bars are SANITY only
// (the rank IS the result); we assert convergence + that placement matters, and
// report the winner.

type Atom = { symbol: string; pos: [number, number, number] };
const norm = (v: number[]): number[] => { const n = Math.hypot(v[0]!, v[1]!, v[2]!); return [v[0]! / n, v[1]! / n, v[2]! / n]; };

/** 6-atom all-trans conjugated chain; atoms at 1-based positions in `nPos` become
 *  N (terminal N keeps 1 H, interior N keeps 0). C↔N this way is electron-neutral,
 *  so every isomer is the same closed-shell 44-e⁻ species — a controlled study. */
function azaChain(nPos: number[], k = 6): Atom[] {
  const nAt = new Set(nPos.map((p) => p - 1));
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
        const hx = u[0]! * Math.cos(phi) - u[1]! * Math.sin(phi);
        const hy = u[0]! * Math.sin(phi) + u[1]! * Math.cos(phi);
        atoms.push({ symbol: "H", pos: [ci[0] + dXH * hx, ci[1] + dXH * hy, 0] });
      }
    } else {
      const a = P[i - 1]!, b = P[i + 1]!;
      const ua = norm([a[0] - ci[0], a[1] - ci[1], 0]);
      const ub = norm([b[0] - ci[0], b[1] - ci[1], 0]);
      const bis = norm([ua[0]! + ub[0]!, ua[1]! + ub[1]!, 0]);
      atoms.push({ symbol: "H", pos: [ci[0] - dXH * bis[0]!, ci[1] - dXH * bis[1]!, 0] });
    }
  }
  return atoms;
}

const LIBRARY = [
  { name: "hexatriene (all-C)", n: [] as number[] }, { name: "1-aza", n: [1] },
  { name: "2-aza", n: [2] }, { name: "3-aza", n: [3] }, { name: "1,6-diaza", n: [1, 6] },
  { name: "1,4-diaza", n: [1, 4] }, { name: "2,5-diaza", n: [2, 5] }, { name: "1,3-diaza", n: [1, 3] },
  { name: "3,4-diaza", n: [3, 4] }, { name: "2,4-diaza", n: [2, 4] },
].map((c) => ({ name: c.name, nN: c.n.length, atoms: azaChain(c.n) }));

test.describe("screening discovery — aza-hexatriene chromophore (blind rank)", () => {
  test("rank an isoelectronic library by gap; surface the best chromophore", async ({ page }) => {
    test.setTimeout(180 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[aza]")) console.log(m.text()); });
    await page.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    const result = await page.evaluate(async (library) => {
      const { runChemEnergyTile } = await import("/src/swarm/chemistry-kernel.ts" as string);
      const log = (s: string): void => { console.log(`[aza] ${s}`); };
      const rows: { name: string; gapEv: number; nN: number; conv: boolean; nE: number }[] = [];
      for (const c of library) {
        const r = await runChemEnergyTile({ label: c.name, atoms: c.atoms as never, basis: "sto-3g", method: "hf" });
        rows.push({ name: c.name, gapEv: r.homoLumoGapEv, nN: c.nN, conv: r.converged, nE: r.nElectrons });
      }
      rows.sort((a, b) => a.gapEv - b.gapEv);
      const baseline = rows.find((r) => r.nN === 0)!.gapEv;
      log("rank  candidate            gap (eV)   vs all-C");
      rows.forEach((r, i) => {
        const rel = r.gapEv - baseline >= 0 ? `+${(r.gapEv - baseline).toFixed(2)}` : (r.gapEv - baseline).toFixed(2);
        log(`  ${String(i + 1).padStart(2)}. ${r.name.padEnd(18)} ${r.gapEv.toFixed(2)}   ${rel}${i === 0 ? "  ◀ best" : ""}`);
      });
      log(`winner: ${rows[0]!.name} (${rows[0]!.gapEv.toFixed(2)} eV); spread ${(rows[rows.length - 1]!.gapEv - rows[0]!.gapEv).toFixed(2)} eV`);
      return rows;
    }, LIBRARY);
console.log(`\n[screening-discovery] winner ${result[0]!.name} ${result[0]!.gapEv.toFixed(2)} eV\n`);

    // Discovery bars = sanity only (the rank IS the result, not a known answer):
    expect(result.every((r) => r.conv)).toBe(true);
    expect(result.every((r) => r.nE === 44)).toBe(true);            // isoelectronic — fair compare
    expect(result.every((r) => r.gapEv > 0 && r.gapEv < 30)).toBe(true);
    expect(result[result.length - 1]!.gapEv - result[0]!.gapEv).toBeGreaterThan(0.5); // placement matters
  });
});
