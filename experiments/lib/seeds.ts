// ─────────────────────────────────────────────────────────────
// seeds.ts — named deterministic seeds.
//
// RESEARCH.md §Reproducibility bans Math.random() in experiment
// paths. Every random draw uses a named seed from this file so
// any reader can re-run and compare. Adding a new name here is a
// commit-worthy event — you're declaring a stable reference.
//
// The RNG is mulberry32: 32-bit state, ~2^32 period, fast, good
// enough for benchmark randomization (not for crypto).
// ─────────────────────────────────────────────────────────────

export const SEEDS = {
  // Used in E1 to build random brick-wall circuits for fidelity tests.
  E1_FIDELITY: 0xF1DE1171,
  // E2 bandwidth-roofline: single-qubit H sweep (deterministic, no seed needed).
  E2_BANDWIDTH: 0xB4, // reserved for future parameterized sweep
  // E3 scaling-law random circuits, 8 layers, brick-wall.
  E3_SCALING: 0x5CA1E51A,
  // E4 dispatch-overhead empty-param random circuits.
  E4_OVERHEAD: 0xD15A7CE,
  // E5 MPS correctness — random brick-wall + heisenberg trotter seed.
  E5_MPS_CORRECTNESS: 0x5E71ADD9,
  // E6 qubit-count ceiling (low-entanglement at large N).
  E6_MPS_CEILING: 0xCE11111C,
  // E7 χ scaling vs entanglement entropy.
  E7_MPS_CHI_SCALING: 0x65CA11C1,
  // E8/E9/E10 reuse E4_OVERHEAD-derived seeds (XOR'd with a per-experiment salt).
  // E12 — Tier C 3-qubit cascade fusion (random Rx/Ry/Rz singles).
  E12_TIERC_FUSION: 0xC1ECA5CD,
  // E13 — Tier D 4-qubit cascade fusion.
  E13_TIERD_FUSION: 0xD4CA5CD2,
  // E16 — VQE on H₂ ground state (random-init seeds for each trial).
  E16_H2_VQE: 0x16C0DE16,
  // E18 — DMRG TFIM scaling vs Pfeuty thermodynamic limit (Phase B).
  E18_TFIM_PFEUTY: 0x18FE7547,
  // E19 — DMRG Heisenberg AFM scaling vs Bethe ansatz limit (Phase B).
  E19_HEISENBERG_BETHE: 0x19BE7AE5,
  // E20 — VQE on LiH ground state (Phase C, real-molecule chemistry).
  E20_LIH_VQE: 0x201E11BE,
  // E21 — VQE on BeH₂ ground state (Phase C v2, multi-atom).
  E21_BEH2_VQE: 0x21BE211E,
  // General-purpose seed for tests / smoke runs.
  SMOKE: 0xC0FFEE,
} as const;

export type SeedName = keyof typeof SEEDS;

/** Deterministic 32-bit RNG — mulberry32. Returns [0, 1). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Look up a named seed and return a fresh RNG. */
export function rngFor(name: SeedName): () => number {
  return mulberry32(SEEDS[name]);
}
