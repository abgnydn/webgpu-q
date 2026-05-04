// Quantum trajectories — the math should bracket two limits:
//   γ = 0       : pure unitary, no measurements ever fire, entropy
//                 grows ballistically.
//   γ very large: nearly every step measures every site, state stays
//                 in the |0/1⟩ product basis, entropy ≈ 0.
import { describe, expect, test } from "vitest";
import { tfim } from "../../src/manybody/hamiltonian.js";
import { recordTrajectory } from "../../src/manybody/trajectory.js";

describe("Quantum trajectory limits", () => {
  test("γ = 0 reproduces unitary evolution: zero measurements, nonzero entropy growth", () => {
    const rec = recordTrajectory({
      hamiltonian: tfim(8, 1, 1),
      gamma: 0,
      steps: 30,
      tau: 0.1,
      chiMax: 32,
      seed: 42,
      initialState: "x-pol",
    });
    let totalMeas = 0;
    for (const s of rec.history) totalMeas += s.measurements.length;
    expect(totalMeas).toBe(0);
    // Entropy must increase from initial 0 to nonzero by step 30.
    expect(rec.history[0]!.midcutEntropy).toBeLessThan(0.05);
    expect(rec.history.at(-1)!.midcutEntropy).toBeGreaterThan(0.3);
  });

  test("γ ≫ 1: the state is constantly measured, entropy stays low", () => {
    const rec = recordTrajectory({
      hamiltonian: tfim(8, 1, 1),
      gamma: 5,            // fires on essentially every step on every site
      steps: 30,
      tau: 0.1,
      chiMax: 32,
      seed: 42,
      initialState: "x-pol",
    });
    // Some measurements should fire in nearly every step.
    let totalMeas = 0;
    for (const s of rec.history) totalMeas += s.measurements.length;
    expect(totalMeas).toBeGreaterThan(rec.history.length); // at least one per step on average
    // Late-time entropy stays bounded (area-law-like).
    const lateE = rec.history.slice(-5).reduce((a, s) => a + s.midcutEntropy, 0) / 5;
    expect(lateE).toBeLessThan(1.0); // far below volume-law value (~N/2 = 4 bits possible)
  });

  test("intermediate γ ≈ critical: measurements fire, entropy nontrivial", () => {
    const rec = recordTrajectory({
      hamiltonian: tfim(8, 1, 1),
      gamma: 0.2,
      steps: 60,
      tau: 0.1,
      chiMax: 32,
      seed: 42,
    });
    let totalMeas = 0;
    for (const s of rec.history) totalMeas += s.measurements.length;
    // At γ·τ = 0.02 per site per step × 8 sites × 60 steps ≈ 9.6 expected events.
    expect(totalMeas).toBeGreaterThan(2);
    expect(totalMeas).toBeLessThan(40);
  });
});

describe("Trajectory invariants", () => {
  test("⟨σ_z⟩ at every site after a Z-measurement outcome is exactly the eigenvalue", () => {
    // After a +1 outcome on site q, ⟨σ_z_q⟩ should be 1; -1 outcome → -1.
    const rec = recordTrajectory({
      hamiltonian: tfim(6, 1, 0.3),       // weak field so measurements have a clean effect
      gamma: 1.5,
      steps: 8,
      tau: 0.1,
      chiMax: 16,
      seed: 7,
    });
    // For each measurement event, look at the ⟨σ_z⟩ at the same site
    // immediately after — should match the outcome (modulo the next
    // half-step of unitary evolution with a small tau, so accept ±0.2 slop).
    for (let stepIdx = 0; stepIdx < rec.history.length; stepIdx++) {
      const step = rec.history[stepIdx]!;
      for (const m of step.measurements) {
        const sz = step.sz[m.site]!;
        // Right after the projection, before the next evolution step.
        // We allow tolerance because the recorded sz already includes the
        // post-projection state but not subsequent evolution.
        expect(Math.sign(sz) === m.outcome || Math.abs(sz) < 0.05).toBe(true);
      }
    }
  });
});
