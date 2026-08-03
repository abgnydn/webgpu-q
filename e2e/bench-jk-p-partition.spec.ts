import { test, expect } from "@playwright/test";

// Algorithmic correctness check for partitioning B by aux index P.
//
// For DF-HF the J/K build is linear in the aux-index sum:
//   γ[P] = Σ_μν B[μν,P] · D[μν]                  (linear in P-row of B)
//   J[μν] = Σ_P B[μν,P] · γ[P]                    (linear in P)
//   X[P,μ,σ] = Σ_λ B[μλ,P] · D[λ,σ]               (each P independent)
//   K[μ,σ] = Σ_P Σ_σ' X[P,μ,σ'] · B[σσ',P]        (linear in P)
//
// → Splitting P into disjoint ranges and summing partial (J, K) gives
// the full (J, K). Verifying that empirically with the existing kernel
// is the algorithmic foundation for distributing B by aux-slice across
// browser tabs. If this is bit-exact, the multi-tab swarm is plumbing.

test.describe("JK_DF: partition by aux index P", () => {
  test("benzene cc-pVDZ — sum of P-slice partials = full JK (bit-exact up to FP reorder)", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDFCholesky, generateAutoAux },
        { preloadWasmJK, buildJK_DF },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
      ]);
      await preloadWasmJK();

      // Benzene cc-pVDZ — small enough for the master to hold full B,
      // but big enough to make the partitioning non-trivial.
      const rCC = 1.395, rCH = 1.087;
      const atoms: Array<{ symbol: string; pos: readonly [number, number, number] }> = [];
      for (let i = 0; i < 6; i++) {
        const θ = i * Math.PI / 3;
        atoms.push({ symbol: "C", pos: [rCC * Math.cos(θ), rCC * Math.sin(θ), 0] });
      }
      for (let i = 0; i < 6; i++) {
        const θ = i * Math.PI / 3;
        const r2 = rCC + rCH;
        atoms.push({ symbol: "H", pos: [r2 * Math.cos(θ), r2 * Math.sin(θ), 0] });
      }
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei, { skipERI: true, skipOAO: true });
      const auxShells = generateAutoAux(shells, 1);
      const df = await buildAuxBasisDFCholesky(shells, auxShells, 1e-8);
      const hf = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-4, maxIter: 10, useDF: df,
      });
      const D = hf.D;
      const n = df.n;
      const nAux = df.nAux;
      const B = df.B;

      // Reference: full JK on the full B.
      const ref = buildJK_DF(df, D);

      // Test cases: split P into k = 2, 3, 4, 8 partitions, sum partials,
      // compare to reference.
      const partitionRanges = (k: number): Array<[number, number]> => {
        const out: Array<[number, number]> = [];
        const step = Math.ceil(nAux / k);
        for (let i = 0; i < k; i++) {
          const s = i * step;
          const e = Math.min(s + step, nAux);
          if (e > s) out.push([s, e]);
        }
        return out;
      };

      const slicePartials = (pStart: number, pEnd: number): { J: Float64Array; K: Float64Array } => {
        const nAuxLocal = pEnd - pStart;
        // Build B_local with shape (n², nAuxLocal): copy B[:, :, pStart..pEnd]
        const Blocal = new Float64Array(n * n * nAuxLocal);
        for (let mu = 0; mu < n; mu++) {
          for (let nu = 0; nu < n; nu++) {
            const baseFull = (mu * n + nu) * nAux;
            const baseLoc = (mu * n + nu) * nAuxLocal;
            for (let p = 0; p < nAuxLocal; p++) {
              Blocal[baseLoc + p] = B[baseFull + (pStart + p)]!;
            }
          }
        }
        // Call existing buildJK_DF with B_local treated as full B of
        // dimension n_aux=nAuxLocal. The math gives partial J, K
        // because the existing kernel does Σ_P over its B's full last
        // axis — which we've made the slice.
        const localDF = { B: Blocal, n, nAux: nAuxLocal, threshold: 0 };
        return buildJK_DF(localDF, D);
      };

      const results: Array<{
        k: number; maxJ: number; maxK: number; relMaxJ: number; relMaxK: number;
      }> = [];
      for (const k of [2, 3, 4, 8]) {
        const ranges = partitionRanges(k);
        const Jsum = new Float64Array(n * n);
        const Ksum = new Float64Array(n * n);
        for (const [pStart, pEnd] of ranges) {
          const { J: Jp, K: Kp } = slicePartials(pStart, pEnd);
          for (let i = 0; i < Jsum.length; i++) {
            Jsum[i]! += Jp[i]!;
            Ksum[i]! += Kp[i]!;
          }
        }
        let maxJ = 0, maxK = 0, refMaxJ = 0, refMaxK = 0;
        for (let i = 0; i < Jsum.length; i++) {
          maxJ = Math.max(maxJ, Math.abs(Jsum[i]! - ref.J[i]!));
          maxK = Math.max(maxK, Math.abs(Ksum[i]! - ref.K[i]!));
          refMaxJ = Math.max(refMaxJ, Math.abs(ref.J[i]!));
          refMaxK = Math.max(refMaxK, Math.abs(ref.K[i]!));
        }
        results.push({
          k, maxJ, maxK,
          relMaxJ: maxJ / Math.max(refMaxJ, 1e-30),
          relMaxK: maxK / Math.max(refMaxK, 1e-30),
        });
      }
      let refMaxJ = 0, refMaxK = 0;
      for (let i = 0; i < ref.J.length; i++) {
        refMaxJ = Math.max(refMaxJ, Math.abs(ref.J[i]!));
        refMaxK = Math.max(refMaxK, Math.abs(ref.K[i]!));
      }
      return { n, nAux, refMaxJ, refMaxK, results };
    });

     console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`P-partition correctness — benzene cc-pVDZ (n=${r.n}, n_aux=${r.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Reference ‖J‖_max = ${r.refMaxJ.toFixed(4)}, ‖K‖_max = ${r.refMaxK.toFixed(4)}`);
    console.log(`partitions | max|ΔJ|       max|ΔK|       relJ        relK`);
    console.log(`-----------+--------------+--------------+------------+------------`);
    for (const row of r.results) { console.log(
        `   ${String(row.k).padStart(7)} | ${row.maxJ.toExponential(2).padStart(12)} | ${row.maxK.toExponential(2).padStart(12)} | ${row.relMaxJ.toExponential(2).padStart(10)} | ${row.relMaxK.toExponential(2).padStart(10)}`,
      );
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
     

    // All partition sums should match the reference to f64 precision
    // (the math is linear-in-P, only FP accumulator reorder noise).
    for (const row of r.results) {
      expect(row.relMaxJ).toBeLessThan(1e-12);
      expect(row.relMaxK).toBeLessThan(1e-12);
    }
  });
});
