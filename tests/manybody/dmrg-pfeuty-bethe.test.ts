// Phase B sanity tests: DMRG ground-state energies converge toward
// the analytical thermodynamic limits at modest N.
//
// At small N (≤ 32), the per-site energy is still substantially
// above the limit (~ b/N boundary correction). These tests don't
// claim convergence yet — they verify the basic math: DMRG runs,
// energies are negative, monotonic-ish in N for the critical cases,
// and the deviation shrinks. The full publishable convergence
// claim lives in the experiment artifacts.
import { describe, expect, test } from "vitest";
import { tfimMPO, heisenbergMPOReal } from "../../src/manybody/mpo.js";
import { dmrgGroundState } from "../../src/manybody/dmrg.js";
import {
  tfimPfeutyEnergyPerBond,
  heisenbergBetheEnergyPerBond,
} from "../../src/manybody/analytical.js";

describe("DMRG TFIM converges toward Pfeuty as N grows", () => {
  test("|E/N − e_∞| at h=1 (critical) shrinks from N=8 to N=24", () => {
    const eInf = tfimPfeutyEnergyPerBond(1, 1);
    const dev = (N: number, chi: number): number => {
      const M = tfimMPO(N, 1, 1);
      const r = dmrgGroundState(M, { chiMax: chi, maxSweeps: 16, tol: 1e-9 });
      return Math.abs(r.energy / N - eInf);
    };
    const d8 = dev(8, 16);
    const d16 = dev(16, 24);
    const d24 = dev(24, 32);
    expect(d24).toBeLessThan(d16);
    expect(d16).toBeLessThan(d8);
    // At N=24, OBC-vs-thermodynamic-limit per-site error is < 0.07
    // (boundary contribution ~ 1.0 / N at criticality).
    expect(d24).toBeLessThan(0.07);
  }, 90_000);
});

describe("DMRG Heisenberg AFM converges toward Bethe as N grows", () => {
  test("|E/N − e_∞| at J=1 shrinks from N=8 to N=24", () => {
    const eInf = heisenbergBetheEnergyPerBond(1);
    const dev = (N: number, chi: number): number => {
      const M = heisenbergMPOReal(N, 1);
      const r = dmrgGroundState(M, { chiMax: chi, maxSweeps: 24, tol: 1e-9 });
      return Math.abs(r.energy / N - eInf);
    };
    const d8 = dev(8, 16);
    const d16 = dev(16, 24);
    const d24 = dev(24, 32);
    expect(d24).toBeLessThan(d16);
    expect(d16).toBeLessThan(d8);
    // Heisenberg has gapless boundary mode → slower convergence than TFIM.
    // At N=24 the gap is still ~ 0.05 per site.
    expect(d24).toBeLessThan(0.06);
  }, 120_000);
});
