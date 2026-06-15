import { test, expect } from "@playwright/test";

// SCREENING VALIDATION (bar ①) — does the screening descriptor reproduce a KNOWN
// chemical trend? Runs through the REAL chem-energy kernel (the same path the swarm
// screening uses), in the browser (WASM-fast), single tab — no multiple devices.
//
// Trend: an all-trans polyene's HOMO–LUMO gap SHRINKS monotonically as the chain
// grows (ethylene→decapentaene) — the textbook "particle-in-a-longer-box" effect,
// why long conjugated dyes are colored. HF/STO-3G overestimates absolute gaps; the
// TREND (what screening ranks on) is robust. If the engine reproduces it, the
// screening filter is trustworthy.

type Atom = { symbol: string; pos: [number, number, number] };
const norm = (v: number[]): number[] => { const n = Math.hypot(v[0]!, v[1]!, v[2]!); return [v[0]! / n, v[1]! / n, v[2]! / n]; };

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
  const atoms: Atom[] = C.map((p) => ({ symbol: "C", pos: p }));
  for (let i = 0; i < k; i++) {
    const ci = C[i]!;
    if (i === 0 || i === k - 1) {
      const nb = i === 0 ? C[1]! : C[k - 2]!;
      const u = norm([nb[0] - ci[0], nb[1] - ci[1], 0]);
      for (const s of [1, -1]) {
        const phi = (s * 120 * Math.PI) / 180;
        const hx = u[0]! * Math.cos(phi) - u[1]! * Math.sin(phi);
        const hy = u[0]! * Math.sin(phi) + u[1]! * Math.cos(phi);
        atoms.push({ symbol: "H", pos: [ci[0] + dCH * hx, ci[1] + dCH * hy, 0] });
      }
    } else {
      const a = C[i - 1]!, b = C[i + 1]!;
      const ua = norm([a[0] - ci[0], a[1] - ci[1], 0]);
      const ub = norm([b[0] - ci[0], b[1] - ci[1], 0]);
      const bis = norm([ua[0]! + ub[0]!, ua[1]! + ub[1]!, 0]);
      atoms.push({ symbol: "H", pos: [ci[0] - dCH * bis[0]!, ci[1] - dCH * bis[1]!, 0] });
    }
  }
  return atoms;
}

const SERIES = [
  { name: "ethylene", k: 2 }, { name: "butadiene", k: 4 }, { name: "hexatriene", k: 6 },
  { name: "octatetraene", k: 8 }, { name: "decapentaene", k: 10 },
].map((m) => ({ ...m, nDouble: m.k / 2, atoms: polyene(m.k) }));

test.describe("screening validation — HOMO–LUMO gap vs conjugation length", () => {
  test("the gap shrinks monotonically as the conjugated chain grows", async ({ page }) => {
    test.setTimeout(180 * 1000);
    page.on("pageerror", (e) => console.error(`[pageerror] ${e.message}`));
    page.on("console", (m) => { if (m.text().startsWith("[poly]")) console.log(m.text()); });
    await page.goto("/swarm.html", { waitUntil: "domcontentloaded" });

    const rows = await page.evaluate(async (series) => {
      const { runChemEnergyTile } = await import("/src/swarm/chemistry-kernel.ts" as string);
      const log = (s: string): void => { console.log(`[poly] ${s}`); };
      const out: { name: string; nDouble: number; gapEv: number; converged: boolean }[] = [];
      for (const m of series) {
        const r = await runChemEnergyTile({ label: m.name, atoms: m.atoms as never, basis: "sto-3g", method: "hf" });
        out.push({ name: m.name, nDouble: m.nDouble, gapEv: r.homoLumoGapEv, converged: r.converged });
      }
      log("molecule        C=C   HOMO–LUMO gap");
      for (const r of out) log(`  ${r.name.padEnd(14)} ${String(r.nDouble).padStart(2)}    ${r.gapEv.toFixed(2)} eV`);
      log(`drop ethylene→decapentaene: ${(out[0]!.gapEv - out[out.length - 1]!.gapEv).toFixed(2)} eV`);
      return out;
    }, SERIES);

    console.log(`\n[screening-validation] ${rows.map((r) => `${r.name}:${r.gapEv.toFixed(1)}`).join("  ")}\n`);

    expect(rows.every((r) => r.converged)).toBe(true);
    expect(rows.every((r) => r.gapEv > 0)).toBe(true);
    // BAR ①: the engine must reproduce the known monotonic decrease.
    for (let i = 1; i < rows.length; i++) expect(rows[i]!.gapEv).toBeLessThan(rows[i - 1]!.gapEv);
    expect(rows[0]!.gapEv - rows[rows.length - 1]!.gapEv).toBeGreaterThan(1.0);
  });
});
