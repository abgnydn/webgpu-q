// MPO cross-validation: every model's MPO contracted to dense
// must equal the per-bond Hamiltonian1D dense build to machine
// precision. If this passes, downstream DMRG / TDVP can trust
// the MPO is the same operator as the bond-list.
import { describe, expect, test } from "vitest";
import { tfim, heisenberg, xxz, buildDense } from "../../src/manybody/hamiltonian.js";
import {
  tfimMPO, heisenbergMPO, xxzMPO, mpoToDense, complexMatrixFrobDistance,
  zeroMPOTensor,
  type MPO,
} from "../../src/manybody/mpo.js";

/** Lift a real 2^N × 2^N row-major matrix into a ComplexMatrix shape. */
function realToComplex(M: Float64Array, dim: number): { rows: number; cols: number; data: Float64Array } {
  const data = new Float64Array(2 * dim * dim);
  for (let r = 0; r < dim; r++) {
    for (let c = 0; c < dim; c++) {
      data[(r * dim + c) * 2] = M[r * dim + c]!;
    }
  }
  return { rows: dim, cols: dim, data };
}

describe("MPO equals bond-list dense Hamiltonian", () => {
  test("TFIM N=4 (J=1, h=1)", () => {
    const N = 4;
    const dense = buildDense(tfim(N, 1, 1));
    const mpoDense = mpoToDense(tfimMPO(N, 1, 1));
    const dist = complexMatrixFrobDistance(realToComplex(dense, 1 << N), mpoDense);
    expect(dist).toBeLessThan(1e-12);
  });

  test("TFIM N=6 (J=1, h=0.5) — paramagnetic", () => {
    const N = 6;
    const dense = buildDense(tfim(N, 1, 0.5));
    const mpoDense = mpoToDense(tfimMPO(N, 1, 0.5));
    const dist = complexMatrixFrobDistance(realToComplex(dense, 1 << N), mpoDense);
    expect(dist).toBeLessThan(1e-12);
  });

  test("TFIM N=8 (J=1, h=2) — strong-field", () => {
    const N = 8;
    const dense = buildDense(tfim(N, 1, 2));
    const mpoDense = mpoToDense(tfimMPO(N, 1, 2));
    const dist = complexMatrixFrobDistance(realToComplex(dense, 1 << N), mpoDense);
    expect(dist).toBeLessThan(1e-12);
  });

  test("Heisenberg N=4 (J=1)", () => {
    const N = 4;
    const dense = buildDense(heisenberg(N, 1));
    const mpoDense = mpoToDense(heisenbergMPO(N, 1));
    const dist = complexMatrixFrobDistance(realToComplex(dense, 1 << N), mpoDense);
    expect(dist).toBeLessThan(1e-12);
  });

  test("Heisenberg N=6 (J=2)", () => {
    const N = 6;
    const dense = buildDense(heisenberg(N, 2));
    const mpoDense = mpoToDense(heisenbergMPO(N, 2));
    const dist = complexMatrixFrobDistance(realToComplex(dense, 1 << N), mpoDense);
    expect(dist).toBeLessThan(1e-12);
  });

  test("XXZ N=4 (J=1, Δ=2) — Ising-like easy-axis", () => {
    const N = 4;
    const dense = buildDense(xxz(N, 1, 2));
    const mpoDense = mpoToDense(xxzMPO(N, 1, 2));
    const dist = complexMatrixFrobDistance(realToComplex(dense, 1 << N), mpoDense);
    expect(dist).toBeLessThan(1e-12);
  });

  test("XXZ N=6 (J=1, Δ=0.5) — easy-plane", () => {
    const N = 6;
    const dense = buildDense(xxz(N, 1, 0.5));
    const mpoDense = mpoToDense(xxzMPO(N, 1, 0.5));
    const dist = complexMatrixFrobDistance(realToComplex(dense, 1 << N), mpoDense);
    expect(dist).toBeLessThan(1e-12);
  });
});

describe("MPO basic structural invariants", () => {
  test("TFIM tensor count and bond shapes", () => {
    const M: MPO = tfimMPO(8, 1, 1);
    expect(M.tensors.length).toBe(8);
    expect(M.tensors[0]!.D_l).toBe(1);
    expect(M.tensors[0]!.D_r).toBe(3);
    expect(M.tensors[7]!.D_l).toBe(3);
    expect(M.tensors[7]!.D_r).toBe(1);
    for (let q = 1; q < 7; q++) {
      expect(M.tensors[q]!.D_l).toBe(3);
      expect(M.tensors[q]!.D_r).toBe(3);
    }
  });

  test("XXZ tensor count and bond shapes (D=5)", () => {
    const M: MPO = xxzMPO(6, 1, 1);
    expect(M.tensors.length).toBe(6);
    expect(M.tensors[0]!.D_l).toBe(1);
    expect(M.tensors[0]!.D_r).toBe(5);
    expect(M.tensors[5]!.D_l).toBe(5);
    expect(M.tensors[5]!.D_r).toBe(1);
  });

  test("MPO dense matrix is Hermitian", () => {
    // TFIM has real-symmetric dense matrix; XXZ at Δ=1 (Heisenberg) has
    // imaginary Y components but Y⊗Y is real, so the full operator is
    // real-symmetric. Test both: M = M^†.
    const cases: { name: string; M: MPO }[] = [
      { name: "TFIM(6, 1, 1)", M: tfimMPO(6, 1, 1) },
      { name: "Heisenberg(6, 1)", M: heisenbergMPO(6, 1) },
      { name: "XXZ(4, 1, 0.5)", M: xxzMPO(4, 1, 0.5) },
    ];
    for (const { name, M } of cases) {
      const D = mpoToDense(M);
      const dim = D.rows;
      let maxAsym = 0;
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          const ij = (i * dim + j) * 2;
          const ji = (j * dim + i) * 2;
          // Hermitian: D[i,j] = conj(D[j,i]) → re(D[i,j]) = re(D[j,i]),
          // im(D[i,j]) = -im(D[j,i]).
          maxAsym = Math.max(maxAsym, Math.abs(D.data[ij]! - D.data[ji]!));
          maxAsym = Math.max(maxAsym, Math.abs(D.data[ij + 1]! + D.data[ji + 1]!));
        }
      }
      expect(maxAsym, name).toBeLessThan(1e-12);
    }
  });

  test("Empty MPO tensor allocation is zero-filled", () => {
    const W = zeroMPOTensor(3, 5);
    expect(W.D_l).toBe(3);
    expect(W.D_r).toBe(5);
    expect(W.data.length).toBe(8 * 3 * 5);
    let nonzero = 0;
    for (let i = 0; i < W.data.length; i++) if (W.data[i] !== 0) nonzero++;
    expect(nonzero).toBe(0);
  });
});
