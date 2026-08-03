import { test, expect } from "@playwright/test";

// Debug Phase 1 aux-DF: compute (s|s) for a single normalized 1s
// Gaussian at origin and compare against the closed-form analytical
// value. This isolates the 2-index kernel from everything else.
//
// For a normalized 1s with exponent α:
//   φ(r) = (2α/π)^(3/4) · exp(-α r²)
// The two-electron repulsion integral:
//   (s|s) = ∫∫ φ(1) (1/r12) φ(2) d³r1 d³r2
//         = N² · 2π^(5/2) / (α² · √(2α))    with N = (2α/π)^(3/4)
//
// For α=1: N² = (2/π)^(3/2), Boys integral = 2π^(5/2)/√2, product ≈ 8.886.

test.describe("aux-DF Phase 1 — kernel debug", () => {
  test("(s|s) for normalized 1s at origin matches closed form", async ({ page }) => {
    test.setTimeout(60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const wasmMod = await import(
        /* @vite-ignore */
        "../../wasm-eri/pkg/wasm_eri.js" as string,
      ) as {
        default(): Promise<unknown>;
        eri_2idx_build: (
          nAux: number,
          nPrimsAux: Uint32Array, primOffAux: Uint32Array,
          alphaAux: Float64Array, cAux: Float64Array,
          centerAux: Float64Array, angularAux: Int32Array,
        ) => Float64Array;
      };
      await wasmMod.default();

      const cases: Array<{ name: string; α: number; expected: number }> = [];

      for (const α of [1.0, 0.5, 2.0, 10.0]) {
        // Closed form: (s|s) = (2α/π)^(3/2) · 2π^(5/2) / (α² · √(2α))
        //                    = 2 · α^(3/2) · π · √2 / α^(5/2)
        //                    = 2 √2 π / √α                  ... wait let me redo
        // N² = (2α/π)^(3/2)
        // Boys integral = 2π^(5/2) / (α · α · √(2α)) = 2π^(5/2) / (α² √(2α))
        // Product = (2α/π)^(3/2) · 2π^(5/2) / (α² √(2α))
        //         = (2α)^(3/2) / π^(3/2) · 2π^(5/2) / (α² √(2α))
        //         = 2 · (2α)^(3/2) · π / (α² √(2α))
        //         = 2π · (2α) / α²                          [ (2α)^(3/2) / √(2α) = 2α ]
        //         = 4π / α
        cases.push({ name: `α=${α}`, α, expected: 4 * Math.PI / α });
      }

      const results: Array<{
        name: string; α: number; computed: number; expected: number; rel: number;
      }> = [];

      for (const c of cases) {
        const nPrims = new Uint32Array([1]);
        const primOff = new Uint32Array([0]);
        const alphaArr = new Float64Array([c.α]);
        const cArr = new Float64Array([1.0]);  // c = 1, norm applied by kernel
        const center = new Float64Array([0, 0, 0]);
        const angular = new Int32Array([0, 0, 0]);

        const M = wasmMod.eri_2idx_build(1, nPrims, primOff, alphaArr, cArr, center, angular);
        const computed = M[0]!;
        const rel = Math.abs(computed - c.expected) / Math.abs(c.expected);
        results.push({ name: c.name, α: c.α, computed, expected: c.expected, rel });
      }
      return { results };
    });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`(s|s) closed-form vs kernel — single normalized 1s @ origin`);
    console.log(`══════════════════════════════════════════════════════════`);
    for (const x of r.results) { console.log(`${x.name.padEnd(8)}  computed=${x.computed.toFixed(8)}  expected=${x.expected.toFixed(8)}  rel=${x.rel.toExponential(2)}`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    for (const x of r.results) {
      expect(x.rel).toBeLessThan(1e-10);
    }
  });

  test("(s_a s_a | s_aux) for normalized 1s at origin — 3-index kernel", async ({ page }) => {
    test.setTimeout(60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const wasmMod = await import(
        /* @vite-ignore */
        "../../wasm-eri/pkg/wasm_eri.js" as string,
      ) as {
        default(): Promise<unknown>;
        eri_3idx_build: (
          nOrb: number, nAux: number,
          nPrimsOrb: Uint32Array, primOffOrb: Uint32Array,
          alphaOrb: Float64Array, cOrb: Float64Array,
          centerOrb: Float64Array, angularOrb: Int32Array,
          nPrimsAux: Uint32Array, primOffAux: Uint32Array,
          alphaAux: Float64Array, cAux: Float64Array,
          centerAux: Float64Array, angularAux: Int32Array,
        ) => Float64Array;
      };
      await wasmMod.default();

      // 1 orbital shell + 1 aux shell, both a single primitive 1s at origin with same α.
      // Closed form for (s_a s_b | s_c) with a=b at origin, both α, c at origin α:
      //   N³(α) · π^(5/2) / (α^(5/2) · √3)
      // where N(α) = (2α/π)^(3/4).

      const cases: Array<{ α: number; expected: number }> = [];
      for (const α of [1.0, 0.5, 2.0, 5.0]) {
        const N = Math.pow(2 * α / Math.PI, 0.75);
        const expected = (N * N * N) * Math.pow(Math.PI, 2.5)
                       / (Math.pow(α, 2.5) * Math.sqrt(3));
        cases.push({ α, expected });
      }

      const results: Array<{ α: number; computed: number; expected: number; rel: number }> = [];

      for (const c of cases) {
        // Single shell with one primitive. Used as both orbital and aux.
        const nPrims = new Uint32Array([1]);
        const primOff = new Uint32Array([0]);
        const alphaArr = new Float64Array([c.α]);
        const cArr = new Float64Array([1.0]);
        const center = new Float64Array([0, 0, 0]);
        const angular = new Int32Array([0, 0, 0]);

        const V = wasmMod.eri_3idx_build(
          1, 1,
          nPrims, primOff, alphaArr, cArr, center, angular,
          nPrims, primOff, alphaArr, cArr, center, angular,
        );
        // V[μν, P] with μ=ν=0, P=0 → V[(0·1+0)·1 + 0] = V[0]
        const computed = V[0]!;
        const rel = Math.abs(computed - c.expected) / Math.abs(c.expected);
        results.push({ α: c.α, computed, expected: c.expected, rel });
      }
      return { results };
    });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`(s_a s_a | s_c) closed-form vs 3-idx kernel`);
    console.log(`══════════════════════════════════════════════════════════`);
    for (const x of r.results) { console.log(`α=${x.α}    computed=${x.computed.toFixed(8)}  expected=${x.expected.toFixed(8)}  rel=${x.rel.toExponential(2)}`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    for (const x of r.results) {
      expect(x.rel).toBeLessThan(1e-10);
    }
  });

  test("Higher-L symmetry sanity: (P|Q) for same-center primitives must respect parity", async ({ page }) => {
    test.setTimeout(60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const wasmMod = await import(
        /* @vite-ignore */
        "../../wasm-eri/pkg/wasm_eri.js" as string,
      ) as {
        default(): Promise<unknown>;
        eri_2idx_build: (
          nAux: number,
          nPrimsAux: Uint32Array, primOffAux: Uint32Array,
          alphaAux: Float64Array, cAux: Float64Array,
          centerAux: Float64Array, angularAux: Int32Array,
        ) => Float64Array;
      };
      await wasmMod.default();

      // Multiple primitives at origin with α=1, varying angular momenta.
      // By parity at origin: (P_a | P_b) = 0 if (a_x + b_x, a_y + b_y, a_z + b_z) has any odd component.
      //   - (px | py): a=(1,0,0), b=(0,1,0): sums (1,1,0) — both odd → integrand odd in BOTH x and y → 0
      //   - (px | px): a=(1,0,0), b=(1,0,0): sums (2,0,0) — all even → nonzero
      //   - (dxy | dxy): (1,1,0)+(1,1,0)=(2,2,0) — even → nonzero
      //   - (dxx | dyy): (2,0,0)+(0,2,0)=(2,2,0) — even → nonzero
      //   - (dxy | dxz): (1,1,0)+(1,0,1)=(2,1,1) — y,z odd → 0
      //   - (g_xxxx | g_xxxx): (4,0,0)+(4,0,0)=(8,0,0) — even → nonzero
      //   - (g_xxxx | g_xxxy): (4,0,0)+(3,1,0)=(7,1,0) — odd → 0

      const cases: Array<{ a: [number, number, number]; b: [number, number, number]; mustBeZero: boolean; name: string }> = [
        { a: [1, 0, 0], b: [1, 0, 0], mustBeZero: false, name: "(px|px) L=1" },
        { a: [1, 0, 0], b: [0, 1, 0], mustBeZero: true,  name: "(px|py) L=1" },
        { a: [1, 1, 0], b: [1, 1, 0], mustBeZero: false, name: "(dxy|dxy) L=2" },
        { a: [2, 0, 0], b: [0, 2, 0], mustBeZero: false, name: "(dxx|dyy) L=2" },
        { a: [1, 1, 0], b: [1, 0, 1], mustBeZero: true,  name: "(dxy|dxz) L=2" },
        { a: [3, 0, 0], b: [3, 0, 0], mustBeZero: false, name: "(fxxx|fxxx) L=3" },
        { a: [3, 0, 0], b: [2, 1, 0], mustBeZero: true,  name: "(fxxx|fxxy) L=3" },
        { a: [4, 0, 0], b: [4, 0, 0], mustBeZero: false, name: "(gxxxx|gxxxx) L=4" },
        { a: [4, 0, 0], b: [3, 1, 0], mustBeZero: true,  name: "(gxxxx|gxxxy) L=4" },
        { a: [2, 2, 0], b: [2, 2, 0], mustBeZero: false, name: "(gxxyy|gxxyy) L=4" },
      ];

      const results: Array<{ name: string; mustBeZero: boolean; value: number; correct: boolean }> = [];
      for (const c of cases) {
        // Two shells at origin, each one primitive with α=1.
        const nPrims = new Uint32Array([1, 1]);
        const primOff = new Uint32Array([0, 1]);
        const alphaArr = new Float64Array([1.0, 1.0]);
        const cArr = new Float64Array([1.0, 1.0]);
        const center = new Float64Array([0, 0, 0, 0, 0, 0]);
        const angular = new Int32Array([c.a[0], c.a[1], c.a[2], c.b[0], c.b[1], c.b[2]]);

        const M = wasmMod.eri_2idx_build(2, nPrims, primOff, alphaArr, cArr, center, angular);
        const value = M[1]!;  // M[0, 1]
        let correct: boolean;
        if (c.mustBeZero) {
          correct = Math.abs(value) < 1e-10;
        } else {
          correct = Math.abs(value) > 1e-3 && Number.isFinite(value);
        }
        results.push({ name: c.name, mustBeZero: c.mustBeZero, value, correct });
      }
      return { results };
    });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`(P|Q) parity sanity at same center, α=1`);
    console.log(`══════════════════════════════════════════════════════════`);
    for (const x of r.results) {
      const expect = x.mustBeZero ? "must = 0" : "must ≠ 0";
      const flag = x.correct ? "✓" : "✗ FAIL";
      console.log(`${x.name.padEnd(25)} ${expect.padEnd(10)} value=${x.value.toExponential(3)}  ${flag}`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    for (const x of r.results) {
      expect(x.correct).toBe(true);
    }
  });
});
