// Two-site DMRG with matrix-free Lanczos, validated against
// ITensor reference energies (TFIM only — XXZ/Heisenberg need a
// complex-Hermitian Lanczos which is a follow-on).
import { describe, expect, test } from "vitest";
import { tfimMPO } from "../../src/manybody/mpo.js";
import { dmrgGroundState } from "../../src/manybody/dmrg.js";
import refData from "./itensor-reference.json" with { type: "json" };

interface RefRow {
  model: string;
  N: number;
  energy: number;
  args: Record<string, number>;
  max_bond_dim: number;
}

function findRef(model: string, N: number, args: Record<string, number>): RefRow {
  const row = (refData.results as RefRow[]).find((r) => {
    if (r.model !== model || r.N !== N) return false;
    for (const k of Object.keys(args)) {
      if (Math.abs(r.args[k]! - args[k]!) > 1e-12) return false;
    }
    return true;
  });
  if (!row) throw new Error(`no ref for model=${model} N=${N} args=${JSON.stringify(args)}`);
  return row;
}

describe("DMRG ground-state vs ITensor reference (TFIM)", () => {
  test("TFIM N=8, J=1, h=0.5 — paramagnetic side", () => {
    const ref = findRef("tfim", 8, { "1": 1, "2": 0.5 });
    const M = tfimMPO(8, 1, 0.5);
    const res = dmrgGroundState(M, { chiMax: 16, maxSweeps: 12, tol: 1e-9 });
    expect(Math.abs(res.energy - ref.energy)).toBeLessThan(1e-5);
  }, 30_000);

  test("TFIM N=8, J=1, h=1 — critical point", () => {
    const ref = findRef("tfim", 8, { "1": 1, "2": 1 });
    const M = tfimMPO(8, 1, 1);
    const res = dmrgGroundState(M, { chiMax: 16, maxSweeps: 16, tol: 1e-9 });
    expect(Math.abs(res.energy - ref.energy)).toBeLessThan(1e-5);
  }, 30_000);

  test("TFIM N=8, J=1, h=2 — ordered side", () => {
    const ref = findRef("tfim", 8, { "1": 1, "2": 2 });
    const M = tfimMPO(8, 1, 2);
    const res = dmrgGroundState(M, { chiMax: 16, maxSweeps: 12, tol: 1e-9 });
    expect(Math.abs(res.energy - ref.energy)).toBeLessThan(1e-5);
  }, 30_000);
});

describe("DMRG basic mechanics", () => {
  test("converges within a few sweeps for small TFIM", () => {
    const M = tfimMPO(6, 1, 1);
    const res = dmrgGroundState(M, { chiMax: 8, maxSweeps: 10, tol: 1e-10 });
    // Should converge — history strictly decreasing (or near-tied).
    for (let i = 1; i < res.history.length; i++) {
      expect(res.history[i]!).toBeLessThanOrEqual(res.history[i - 1]! + 1e-9);
    }
    // Final energy reasonable for N=6 critical TFIM (close to ITensor N=8 value scaled).
    expect(res.energy).toBeLessThan(0);
    expect(res.energy).toBeGreaterThan(-20);
  }, 30_000);
});
