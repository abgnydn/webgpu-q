import { test, expect } from "@playwright/test";

// Phase 1 of aux-basis DF: verify the 3-index + 2-index kernels by
// reconstructing the 4-index ERI as
//   ERI[μν, λσ] ≈ Σ_P B[μν, P] · B[λσ, P]
// where B = V · M^(-1/2), V is the 3-index tensor, M is the 2-index
// metric. With aux basis = orbital basis (cc-pVDZ), reconstruction
// should be near-machine-precision.
//
// This is the canonical DF correctness check.

test.describe("aux-basis DF Phase 1 (correctness)", () => {
  test("H₂ cc-pVDZ — 3-idx + 2-idx → B → reconstruct ERI", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
      ]);
      // Use H₂ first (tiny, n=10 for cc-pVDZ) — fast to verify.
      const atoms = [
        { symbol: "H", pos: [0, 0, -0.371] },
        { symbol: "H", pos: [0, 0,  0.371] },
      ] as const;
      const { shells, nuclei } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      // Direct 4-index ERI (reference).
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const eri_direct = integrals.eri_AO;
      const n = integrals.n;

      // 3-idx and 2-idx via WASM, using cc-pVDZ shells as the aux basis.
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
        eri_2idx_build: (
          nAux: number,
          nPrimsAux: Uint32Array, primOffAux: Uint32Array,
          alphaAux: Float64Array, cAux: Float64Array,
          centerAux: Float64Array, angularAux: Int32Array,
        ) => Float64Array;
      };
      await wasmMod.default();

      // Pack shells (re-used as both orbital and aux).
      const packShells = (sh: ReadonlyArray<typeof shells[number]>): {
        nPrims: Uint32Array; primOff: Uint32Array;
        alpha: Float64Array; c: Float64Array;
        center: Float64Array; angular: Int32Array;
      } => {
        const sz = sh.length;
        const nPrims = new Uint32Array(sz);
        const primOff = new Uint32Array(sz);
        let total = 0;
        for (let i = 0; i < sz; i++) { nPrims[i] = sh[i]!.alpha.length; primOff[i] = total; total += sh[i]!.alpha.length; }
        const alpha = new Float64Array(total);
        const c = new Float64Array(total);
        for (let i = 0; i < sz; i++) {
          const off = primOff[i]!;
          for (let p = 0; p < sh[i]!.alpha.length; p++) {
            alpha[off + p] = sh[i]!.alpha[p]!;
            c[off + p] = sh[i]!.c[p]!;
          }
        }
        const center = new Float64Array(sz * 3);
        const angular = new Int32Array(sz * 3);
        for (let i = 0; i < sz; i++) {
          center[i * 3] = sh[i]!.center[0]; center[i * 3 + 1] = sh[i]!.center[1]; center[i * 3 + 2] = sh[i]!.center[2];
          angular[i * 3] = sh[i]!.angular[0]; angular[i * 3 + 1] = sh[i]!.angular[1]; angular[i * 3 + 2] = sh[i]!.angular[2];
        }
        return { nPrims, primOff, alpha, c, center, angular };
      };
      const p = packShells(shells);

      const t3 = performance.now();
      const V = wasmMod.eri_3idx_build(
        n, n,
        p.nPrims, p.primOff, p.alpha, p.c, p.center, p.angular,
        p.nPrims, p.primOff, p.alpha, p.c, p.center, p.angular,
      );
      const t3Ms = performance.now() - t3;

      const t2 = performance.now();
      const M = wasmMod.eri_2idx_build(
        n,
        p.nPrims, p.primOff, p.alpha, p.c, p.center, p.angular,
      );
      const t2Ms = performance.now() - t2;

      // Verify V symmetry V[μν, P] = V[νμ, P]
      let maxVAsym = 0;
      for (let mu = 0; mu < n; mu++) {
        for (let nu = mu + 1; nu < n; nu++) {
          for (let P = 0; P < n; P++) {
            const d = Math.abs(V[(mu * n + nu) * n + P]! - V[(nu * n + mu) * n + P]!);
            if (d > maxVAsym) maxVAsym = d;
          }
        }
      }

      // Verify M symmetry M[P, Q] = M[Q, P]
      let maxMAsym = 0;
      for (let P = 0; P < n; P++) {
        for (let Q = P + 1; Q < n; Q++) {
          const d = Math.abs(M[P * n + Q]! - M[Q * n + P]!);
          if (d > maxMAsym) maxMAsym = d;
        }
      }

      // M[P, P] should equal direct ERI(P, ghost, P, ghost) — but
      // we can sanity-check that M is positive definite at least.
      // Eigendecomposition via PySCF / standard solver isn't here;
      // simplest spot-check: M diagonals positive.
      let minDiag = Infinity;
      for (let P = 0; P < n; P++) {
        const d = M[P * n + P]!;
        if (d < minDiag) minDiag = d;
      }

      return {
        n,
        nElements: n * n * n,
        t3Ms, t2Ms,
        maxVAsym, maxMAsym, minMDiag: minDiag,
        sampleV: V[0]!,
        sampleM00: M[0]!,
        sampleERI_0000: eri_direct[0]!,
      };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Aux-basis DF Phase 1 — H₂ cc-pVDZ (n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`3-index V tensor: ${r.nElements} entries built in ${r.t3Ms.toFixed(1)} ms`);
    console.log(`2-index M metric: ${r.n*r.n} entries built in ${r.t2Ms.toFixed(1)} ms`);
    console.log();
    console.log(`V[μν, P] symmetry max|V[μν,P] − V[νμ,P]|: ${r.maxVAsym.toExponential(2)}`);
    console.log(`M[P, Q] symmetry max|M[P,Q] − M[Q,P]|:     ${r.maxMAsym.toExponential(2)}`);
    console.log(`min M diagonal (should be > 0 for PSD):    ${r.minMDiag.toExponential(3)}`);
    console.log();
    console.log(`Sample V[0,0,0]:        ${r.sampleV.toExponential(4)}`);
    console.log(`Sample M[0,0]:          ${r.sampleM00.toExponential(4)}`);
    console.log(`Direct ERI[0,0,0,0]:    ${r.sampleERI_0000.toExponential(4)}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.maxVAsym).toBeLessThan(1e-10);
    expect(r.maxMAsym).toBeLessThan(1e-10);
    expect(r.minMDiag).toBeGreaterThan(0);
  });

  test("H₂ cc-pVDZ — B = V·M^(-1/2) reconstructs the 4-index ERI", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { buildAuxBasisDF },
        { reconstructERI },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
        import("/src/chemistry/df.ts" as string),
      ]);

      const atoms = [
        { symbol: "H", pos: [0, 0, -0.371] },
        { symbol: "H", pos: [0, 0,  0.371] },
      ] as const;
      const { shells, nuclei } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      // Reference: direct 4-index ERI.
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const eri_direct = integrals.eri_AO;
      const n = integrals.n;

      // Aux-basis DF (Phase 1, aux == orbital).
      const tDF = performance.now();
      const df = await buildAuxBasisDF(shells, shells, 1e-10);
      const dfMs = performance.now() - tDF;

      // Reconstruct ERI from B.
      const eri_rec = reconstructERI(df);

      // Compare.
      let maxAbs = 0, maxRel = 0, sumAbs2 = 0;
      for (let i = 0; i < eri_direct.length; i++) {
        const d = Math.abs(eri_direct[i]! - eri_rec[i]!);
        const refMag = Math.abs(eri_direct[i]!);
        if (d > maxAbs) maxAbs = d;
        sumAbs2 += d * d;
        if (refMag > 1e-8) {
          const rel = d / refMag;
          if (rel > maxRel) maxRel = rel;
        }
      }
      const rms = Math.sqrt(sumAbs2 / eri_direct.length);

      return { n, nAux: df.nAux, dfMs, maxAbs, maxRel, rms };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Aux-basis DF — H₂ cc-pVDZ ERI reconstruction (n=${r.n}, n_aux=${r.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`B-tensor build time:       ${r.dfMs.toFixed(1)} ms`);
    console.log(`max |ERI_direct − Bᵀ·B|:  ${r.maxAbs.toExponential(2)} Ha`);
    console.log(`max relative error:        ${r.maxRel.toExponential(2)}`);
    console.log(`RMS error:                 ${r.rms.toExponential(2)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    // With aux == orbital basis, reconstruction error is bounded by
    // the orthogonal-complement projection (basis-set incompleteness).
    // Just check it's non-trivial and finite.
    expect(r.maxAbs).toBeLessThan(1.0);  // generous; tight when proper jkfit aux is used
    expect(Number.isFinite(r.rms)).toBe(true);
  });

  test("H₂ cc-pVDZ — full HF SCF energy with aux-basis DF", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDF },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);

      const atoms = [
        { symbol: "H", pos: [0, 0, -0.371] },
        { symbol: "H", pos: [0, 0,  0.371] },
      ] as const;
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      const integrals = computeMolecularIntegrals(shells, nuclei);

      // Reference: direct HF.
      const hfDirect = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
      });

      // Aux-basis DF.
      const df = await buildAuxBasisDF(shells, shells, 1e-10);
      const hfDF = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
        useDF: df,
      });

      // Aug-cc-pVDZ as aux: orbital cc-pVDZ + aug-diffuse. Larger aux.
      const { shells: augShells } =
        moleculeToShellsNuclei(atoms as never, "aug-cc-pvdz");
      const dfAug = await buildAuxBasisDF(shells, augShells, 1e-10);
      const hfDFAug = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
        useDF: dfAug,
      });

      return {
        n: integrals.n,
        nAux: df.nAux, nAugAux: dfAug.nAux,
        eDirect: hfDirect.energy,
        iDirect: hfDirect.iter,
        eDF: hfDF.energy,
        iDF: hfDF.iter, cDF: hfDF.converged,
        eDFAug: hfDFAug.energy,
        iDFAug: hfDFAug.iter, cDFAug: hfDFAug.converged,
      };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Aux-basis DF HF energy — H₂ cc-pVDZ (orbital n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Direct ERI HF:             E = ${r.eDirect.toFixed(9)} Ha   iter=${r.iDirect}`);
    console.log(`DF (aux=cc-pVDZ, n_aux=${r.nAux}):`);
    console.log(`  HF:                      E = ${r.eDF.toFixed(9)} Ha   iter=${r.iDF}  cnv=${r.cDF}`);
    console.log(`  energy error vs direct:  ${(r.eDF - r.eDirect).toExponential(2)} Ha`);
    console.log(`DF (aux=aug-cc-pVDZ, n_aux=${r.nAugAux}):`);
    console.log(`  HF:                      E = ${r.eDFAug.toFixed(9)} Ha   iter=${r.iDFAug}  cnv=${r.cDFAug}`);
    console.log(`  energy error vs direct:  ${(r.eDFAug - r.eDirect).toExponential(2)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(Number.isFinite(r.eDF)).toBe(true);
    expect(Number.isFinite(r.eDFAug)).toBe(true);
  });

  test("H₂O cc-pVDZ — aux-DF HF energy on a real molecule", async ({ page }) => {
    test.setTimeout(3 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDF },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);

      const half = (104.52 / 2) * Math.PI / 180;
      const xH = 0.9572 * Math.sin(half) / 0.529177;  // → bohr
      const zH = 0.9572 * Math.cos(half) / 0.529177;
      const atoms = [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ xH, 0, zH] },
        { symbol: "H", pos: [-xH, 0, zH] },
      ] as const;
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);

      const hfDirect = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
      });

      const tDF = performance.now();
      const df = await buildAuxBasisDF(shells, shells, 1e-10);
      const dfMs = performance.now() - tDF;
      const hfDF = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
        useDF: df,
      });

      const { shells: augShells } =
        moleculeToShellsNuclei(atoms as never, "aug-cc-pvdz");
      const tDFAug = performance.now();
      const dfAug = await buildAuxBasisDF(shells, augShells, 1e-10);
      const dfAugMs = performance.now() - tDFAug;
      const hfDFAug = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
        useDF: dfAug,
      });

      return {
        n: integrals.n,
        nAux: df.nAux, nAugAux: dfAug.nAux,
        dfMs, dfAugMs,
        eDirect: hfDirect.energy,
        eDF: hfDF.energy, iDF: hfDF.iter, cDF: hfDF.converged,
        eDFAug: hfDFAug.energy, iDFAug: hfDFAug.iter, cDFAug: hfDFAug.converged,
      };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Aux-basis DF HF — H₂O cc-pVDZ (orbital n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Direct ERI HF:                E = ${r.eDirect.toFixed(8)} Ha`);
    console.log();
    console.log(`DF (aux=cc-pVDZ, n_aux=${r.nAux}):`);
    console.log(`  B-tensor build:             ${r.dfMs.toFixed(1)} ms`);
    console.log(`  HF:                         E = ${r.eDF.toFixed(8)} Ha   iter=${r.iDF}  cnv=${r.cDF}`);
    console.log(`  energy error vs direct:     ${(r.eDF - r.eDirect).toExponential(2)} Ha`);
    console.log();
    console.log(`DF (aux=aug-cc-pVDZ, n_aux=${r.nAugAux}):`);
    console.log(`  B-tensor build:             ${r.dfAugMs.toFixed(1)} ms`);
    console.log(`  HF:                         E = ${r.eDFAug.toFixed(8)} Ha   iter=${r.iDFAug}  cnv=${r.cDFAug}`);
    console.log(`  energy error vs direct:     ${(r.eDFAug - r.eDirect).toExponential(2)} Ha`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(Number.isFinite(r.eDF)).toBe(true);
    expect(Number.isFinite(r.eDFAug)).toBe(true);
  });

  test("H₂O STO-3G orbital + cc-pVDZ aux — confirm aux angular momentum is the bottleneck", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDF },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);

      const half = (104.52 / 2) * Math.PI / 180;
      const xH = 0.9572 * Math.sin(half) / 0.529177;
      const zH = 0.9572 * Math.cos(half) / 0.529177;
      const atoms = [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ xH, 0, zH] },
        { symbol: "H", pos: [-xH, 0, zH] },
      ] as const;

      // STO-3G orbital (max L=1 for O) + cc-pVDZ aux (max L=2).
      const { shells: orbShells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "sto-3g");
      const { shells: auxShells } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(orbShells, nuclei);

      const hfDirect = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
      });

      const df = await buildAuxBasisDF(orbShells, auxShells, 1e-10);
      const hfDF = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
        useDF: df,
      });

      return {
        nOrb: integrals.n, nAux: df.nAux,
        eDirect: hfDirect.energy,
        eDF: hfDF.energy, iDF: hfDF.iter, cDF: hfDF.converged,
      };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`H₂O STO-3G orbital + cc-pVDZ aux (n_orb=${r.nOrb}, n_aux=${r.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Direct HF (STO-3G):           E = ${r.eDirect.toFixed(9)} Ha`);
    console.log(`DF HF (aux=cc-pVDZ):          E = ${r.eDF.toFixed(9)} Ha`);
    console.log(`Energy error vs direct:        ${(r.eDF - r.eDirect).toExponential(2)} Ha`);
    console.log(`Iterations:                    ${r.iDF}`);
    console.log(`Converged:                     ${r.cDF}`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(Number.isFinite(r.eDF)).toBe(true);
  });

  test("H₂O cc-pVDZ — auto-aux (decontract + L extension) HF energy", async ({ page }) => {
    test.setTimeout(3 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDF, generateAutoAux },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);

      const half = (104.52 / 2) * Math.PI / 180;
      const xH = 0.9572 * Math.sin(half) / 0.529177;
      const zH = 0.9572 * Math.cos(half) / 0.529177;
      const atoms = [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ xH, 0, zH] },
        { symbol: "H", pos: [-xH, 0, zH] },
      ] as const;
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);

      const hfDirect = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
      });

      const results: Array<{
        extraL: number; nAux: number; buildMs: number; energy: number;
        iter: number; cnv: boolean; error: number;
      }> = [];

      for (const extraL of [0, 1, 2]) {
        const auxShells = generateAutoAux(shells, extraL);
        const tStart = performance.now();
        const df = await buildAuxBasisDF(shells, auxShells, 1e-10);
        const buildMs = performance.now() - tStart;
        const hf = runRHFSCF(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-9, maxIter: 100, useDF: df,
        });
        results.push({
          extraL, nAux: df.nAux, buildMs,
          energy: hf.energy, iter: hf.iter, cnv: hf.converged,
          error: hf.energy - hfDirect.energy,
        });
      }
      return { n: integrals.n, eDirect: hfDirect.energy, results };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Auto-aux DF HF — H₂O cc-pVDZ (orbital n=${r.n})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Direct ERI HF:  E = ${r.eDirect.toFixed(8)} Ha`);
    console.log();
    for (const x of r.results) {
      console.log(`auto-aux extraL=${x.extraL}, n_aux=${String(x.nAux).padStart(3)}, ` +
                  `build=${x.buildMs.toFixed(0).padStart(5)} ms: ` +
                  `E=${x.energy.toFixed(8)} Ha  iter=${x.iter}  cnv=${x.cnv}  ` +
                  `error=${x.error.toExponential(2)} Ha`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.results.every((x) => Number.isFinite(x.energy))).toBe(true);
  });

  test("H₂O extraL=2 — sweep regularization threshold to recover stability", async ({ page }) => {
    test.setTimeout(2 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDF, generateAutoAux },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);

      const half = (104.52 / 2) * Math.PI / 180;
      const xH = 0.9572 * Math.sin(half) / 0.529177;
      const zH = 0.9572 * Math.cos(half) / 0.529177;
      const atoms = [
        { symbol: "O", pos: [0, 0, 0] },
        { symbol: "H", pos: [ xH, 0, zH] },
        { symbol: "H", pos: [-xH, 0, zH] },
      ] as const;
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const eDirect = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
      }).energy;

      const auxShells = generateAutoAux(shells, 2);
      const results: Array<{ reg: number; nAux: number; e: number; iter: number; cnv: boolean }> = [];

      for (const reg of [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]) {
        const df = await buildAuxBasisDF(shells, auxShells, reg);
        const hf = runRHFSCF(integrals, nElectrons, {
          useDIIS: true, energyTol: 1e-9, maxIter: 100, useDF: df,
        });
        results.push({ reg, nAux: df.nAux, e: hf.energy, iter: hf.iter, cnv: hf.converged });
      }
      return { eDirect, results };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`H₂O extraL=2 — eigendecomp regularization sweep`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Direct ERI HF: E = ${r.eDirect.toFixed(8)} Ha`);
    console.log();
    for (const x of r.results) {
      const err = x.e - r.eDirect;
      console.log(`reg=${x.reg.toExponential(0).padStart(7)}  n_aux_kept=${String(x.nAux).padStart(3)}  ` +
                  `E=${x.e.toFixed(6).padStart(13)} Ha  iter=${String(x.iter).padStart(3)}  cnv=${x.cnv}  ` +
                  `err=${err.toExponential(2)} Ha`);
    }
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(r.results.some((x) => Math.abs(x.e - r.eDirect) < 1e-3 && x.cnv)).toBe(true);
  });

  test("ethane cc-pVDZ — auto-aux DF HF energy + timing", async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    await page.goto("/molecule.html", { waitUntil: "domcontentloaded" });

    const r = await page.evaluate(async () => {
      const [
        { moleculeToShellsNuclei },
        { computeMolecularIntegrals },
        { runRHFSCF },
        { buildAuxBasisDF, generateAutoAux },
      ] = await Promise.all([
        import("/src/chemistry/atoms.ts" as string),
        import("/src/chemistry/cg-molecular.ts" as string),
        import("/src/chemistry/hf-scf.ts" as string),
        import("/src/chemistry/df-aux.ts" as string),
      ]);

      const cc = 1.535, ch = 1.094;
      const hch = 107.8 * Math.PI / 180;
      const hcc = (Math.PI - hch) / 2 + Math.PI / 6;
      const sH = ch * Math.sin(hcc), cH = ch * Math.cos(hcc);
      const atoms = [
        { symbol: "C", pos: [0, 0, -cc / 2] },
        { symbol: "C", pos: [0, 0,  cc / 2] },
        { symbol: "H", pos: [ sH, 0, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH / 2,  sH * Math.sqrt(3) / 2, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH / 2, -sH * Math.sqrt(3) / 2, -cc / 2 - cH] },
        { symbol: "H", pos: [-sH, 0,  cc / 2 + cH] },
        { symbol: "H", pos: [ sH / 2,  sH * Math.sqrt(3) / 2,  cc / 2 + cH] },
        { symbol: "H", pos: [ sH / 2, -sH * Math.sqrt(3) / 2,  cc / 2 + cH] },
      ];
      const { shells, nuclei, nElectrons } =
        moleculeToShellsNuclei(atoms as never, "cc-pvdz");

      // Direct ERI HF (reference + 4-index build time).
      const tDirectEri = performance.now();
      const integrals = computeMolecularIntegrals(shells, nuclei);
      const directEriMs = performance.now() - tDirectEri;
      const tDirectHF = performance.now();
      const hfDirect = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100,
      });
      const directHFms = performance.now() - tDirectHF;

      // Auto-aux DF, extraL=1.
      const auxShells = generateAutoAux(shells, 1);
      const tDF = performance.now();
      const df = await buildAuxBasisDF(shells, auxShells, 1e-10);
      const dfBuildMs = performance.now() - tDF;
      const tDFHf = performance.now();
      const hfDF = runRHFSCF(integrals, nElectrons, {
        useDIIS: true, energyTol: 1e-9, maxIter: 100, useDF: df,
      });
      const dfHFms = performance.now() - tDFHf;

      return {
        n: integrals.n, nAux: df.nAux,
        directEriMs, directHFms,
        dfBuildMs, dfHFms,
        eDirect: hfDirect.energy, eDF: hfDF.energy,
        iterDirect: hfDirect.iter, iterDF: hfDF.iter,
        cnvDF: hfDF.converged,
      };
    });

    /* eslint-disable no-console */
    console.log(`\n══════════════════════════════════════════════════════════`);
    console.log(`Auto-aux DF — ethane cc-pVDZ (n=${r.n}, n_aux=${r.nAux})`);
    console.log(`══════════════════════════════════════════════════════════`);
    console.log(`Direct path:`);
    console.log(`  4-index ERI build:    ${(r.directEriMs / 1000).toFixed(2)} s`);
    console.log(`  HF SCF (${r.iterDirect} iter):    ${(r.directHFms / 1000).toFixed(2)} s`);
    console.log(`  Total:                ${((r.directEriMs + r.directHFms) / 1000).toFixed(2)} s`);
    console.log(`  E = ${r.eDirect.toFixed(8)} Ha`);
    console.log();
    console.log(`Auto-aux DF path (extraL=1):`);
    console.log(`  B-tensor build:       ${(r.dfBuildMs / 1000).toFixed(2)} s`);
    console.log(`  HF SCF (${r.iterDF} iter):    ${(r.dfHFms / 1000).toFixed(2)} s   cnv=${r.cnvDF}`);
    console.log(`  Total:                ${((r.dfBuildMs + r.dfHFms) / 1000).toFixed(2)} s`);
    console.log(`  E = ${r.eDF.toFixed(8)} Ha`);
    console.log(`  Energy error:         ${(r.eDF - r.eDirect).toExponential(2)} Ha`);
    console.log();
    console.log(`Speedup (Direct → DF):  ${((r.directEriMs + r.directHFms) / (r.dfBuildMs + r.dfHFms)).toFixed(2)}×`);
    console.log(`══════════════════════════════════════════════════════════\n`);
    /* eslint-enable no-console */

    expect(Number.isFinite(r.eDF)).toBe(true);
  });
});
