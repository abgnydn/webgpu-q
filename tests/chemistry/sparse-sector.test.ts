// Sparse-CSR sector H validation. Strategy:
//   1. Cross-check sparse vs dense on LiH N=4 — same FCI to f64.
//   2. Same for BeH₂-full N=6.
//   3. Same for H₂O N=10 (the headline molecule for the new path).
//   4. Confirm the sparse representation is significantly smaller
//      than dense.
import { describe, expect, test } from "vitest";
import { computeMolecularIntegrals } from "../../src/chemistry/cg-molecular.js";
import { moleculeToShellsNuclei, type Atom } from "../../src/chemistry/atoms.js";
import { buildSectorH } from "../../src/chemistry/sector-builder.js";
import { buildSparseSectorH, sparseMatvec, type SparseSectorH } from "../../src/chemistry/sector-matvec.js";
import { eigsymmetric } from "../../src/manybody/dense-eig.js";
import { lanczosGroundState } from "../../src/manybody/lanczos.js";

function fciDenseLanczos(H: Float64Array, k: number): number {
  // Dense matvec via Lanczos — same algorithm as sparse, comparable cost.
  // (Jacobi is O(k³); cubic in k makes it impractical past ~1000 dim.)
  const matvec = (x: Float64Array, out: Float64Array) => {
    for (let i = 0; i < k; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) s += H[i * k + j]! * x[j]!;
      out[i] = s;
    }
  };
  const x0 = new Float64Array(k);
  for (let i = 0; i < k; i++) x0[i] = 1;
  return lanczosGroundState(matvec, x0, { maxIter: 120, tol: 1e-10 }).energy;
}

function fciDenseJacobi(H: Float64Array, k: number): number {
  return eigsymmetric(H, k).values[0]!;
}

function fciSparse(H: SparseSectorH): number {
  const matvec = (x: Float64Array, out: Float64Array) => sparseMatvec(H, x, out);
  const x0 = new Float64Array(H.k);
  for (let i = 0; i < H.k; i++) x0[i] = 1;
  return lanczosGroundState(matvec, x0, { maxIter: 120, tol: 1e-10 }).energy;
}

interface RunResult {
  nElectrons: number; nSpatial: number; sectorDim: number;
  E_ref: number; E_sparse: number; nnz: number;
}

function runMoleculeSmall(atoms: readonly Atom[]): RunResult {
  // Use Jacobi on dense for the reference. Only viable for small k.
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const nSpatial = shells.length;
  const dense = buildSectorH(nSpatial, integrals.h_OAO, integrals.eri_OAO, integrals.Vnn, nElectrons);
  const sparse = buildSparseSectorH(nSpatial, integrals.h_OAO, integrals.eri_OAO, integrals.Vnn, nElectrons);
  return {
    nElectrons, nSpatial, sectorDim: dense.k,
    E_ref: fciDenseJacobi(dense.H, dense.k),
    E_sparse: fciSparse(sparse),
    nnz: sparse.values.length,
  };
}

function runMoleculeMedium(atoms: readonly Atom[]): RunResult {
  // Reference via dense-Lanczos (fast even at k ≈ 3000).
  const { shells, nuclei, nElectrons } = moleculeToShellsNuclei(atoms);
  const integrals = computeMolecularIntegrals(shells, nuclei);
  const nSpatial = shells.length;
  const dense = buildSectorH(nSpatial, integrals.h_OAO, integrals.eri_OAO, integrals.Vnn, nElectrons);
  const sparse = buildSparseSectorH(nSpatial, integrals.h_OAO, integrals.eri_OAO, integrals.Vnn, nElectrons);
  return {
    nElectrons, nSpatial, sectorDim: dense.k,
    E_ref: fciDenseLanczos(dense.H, dense.k),
    E_sparse: fciSparse(sparse),
    nnz: sparse.values.length,
  };
}

describe("Sparse-CSR sector H ≡ dense on LiH (N=4, Jacobi reference)", () => {
  test("FCI energies agree to f64 precision", () => {
    const r = runMoleculeSmall([
      { symbol: "Li", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, 1.595] },
    ]);
    expect(r.sectorDim).toBe(15);
    expect(Math.abs(r.E_ref - r.E_sparse)).toBeLessThan(1e-10);
  }, 30_000);
});

describe("Sparse-CSR sector H ≡ dense on H₂O (N=10)", () => {
  test("FCI energies agree to f64 precision; sparse memory < 1/5 of dense", () => {
    const R_OH = 0.9572;
    const half = (104.52 / 2) * Math.PI / 180;
    const x = R_OH * Math.sin(half);
    const z = R_OH * Math.cos(half);
    const r = runMoleculeMedium([
      { symbol: "O", pos: [0, 0, 0] },
      { symbol: "H", pos: [ x, 0, z] },
      { symbol: "H", pos: [-x, 0, z] },
    ]);
    expect(r.sectorDim).toBe(1001);
    expect(Math.abs(r.E_ref - r.E_sparse)).toBeLessThan(1e-9);
    const dense_bytes = 8 * r.sectorDim * r.sectorDim;
    const sparse_bytes = 12 * r.nnz;
    expect(sparse_bytes).toBeLessThan(dense_bytes / 5);
  }, 60_000);
});

describe("Sparse-CSR sector H ≡ dense on BeH₂-full (N=6, Lanczos reference)", () => {
  test("FCI energies agree to f64 precision; sparse memory < 1/30 of dense", () => {
    const R = 1.34;
    const r = runMoleculeMedium([
      { symbol: "Be", pos: [0, 0, 0] },
      { symbol: "H",  pos: [0, 0, -R] },
      { symbol: "H",  pos: [0, 0,  R] },
    ]);
    expect(r.sectorDim).toBe(3003);
    expect(Math.abs(r.E_ref - r.E_sparse)).toBeLessThan(1e-9);
    const dense_bytes = 8 * r.sectorDim * r.sectorDim;
    const sparse_bytes = 12 * r.nnz;
    // BeH₂-full sparsity: ~98% zero, so sparse is ~30× smaller than dense.
    expect(sparse_bytes).toBeLessThan(dense_bytes / 30);
  }, 30_000);
});
