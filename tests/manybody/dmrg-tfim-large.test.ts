// Larger-N TFIM cross-checks against ITensor reference. Single-shot —
// designed to cover scaling, not exhaustive sweeps.
import { describe, expect, test } from "vitest";
import { tfimMPO } from "../../src/manybody/mpo.js";
import { dmrgGroundState } from "../../src/manybody/dmrg.js";

describe("DMRG TFIM scaling (vs ITensor)", () => {
  test("TFIM N=16 h=0.5 within 1e-4 of ITensor", () => {
    const ref = -16.146039701575138;
    const M = tfimMPO(16, 1, 0.5);
    const r = dmrgGroundState(M, { chiMax: 16, maxSweeps: 20, tol: 1e-9 });
    expect(Math.abs(r.energy - ref)).toBeLessThan(1e-4);
  }, 60_000);

  test("TFIM N=16 h=1 (critical) within 1e-4 of ITensor", () => {
    const ref = -20.016387900466384;
    const M = tfimMPO(16, 1, 1);
    const r = dmrgGroundState(M, { chiMax: 24, maxSweeps: 24, tol: 1e-9 });
    expect(Math.abs(r.energy - ref)).toBeLessThan(1e-4);
  }, 90_000);

  test("TFIM N=16 h=2 within 1e-4 of ITensor", () => {
    const ref = -33.901852;
    const M = tfimMPO(16, 1, 2);
    const r = dmrgGroundState(M, { chiMax: 16, maxSweeps: 16, tol: 1e-9 });
    expect(Math.abs(r.energy - ref)).toBeLessThan(1e-4);
  }, 60_000);
});
