// ─────────────────────────────────────────────────────────────
// molecules.ts — pre-built common molecules with experimental
// equilibrium geometries. Saves users from hand-typing XYZ
// coordinates for everyday demos and benchmarks.
//
// All coordinates in Å. Geometries taken from CCCBDB / NIST
// chemistry webbook unless noted. Element coverage matches
// `AtomSymbol`: H-Ar, all 18 (periods 1-3 complete).
// ─────────────────────────────────────────────────────────────

import type { Atom } from "./atoms.js";

/** Atomic hydrogen H. Used as the simplest doublet test case (1 α). */
export const atomicH: Atom[] = [
  { symbol: "H", pos: [0, 0, 0] },
];

/** H₂ at experimental equilibrium r_e = 0.7414 Å. */
export const h2: Atom[] = [
  { symbol: "H", pos: [0, 0,  0.3707] },
  { symbol: "H", pos: [0, 0, -0.3707] },
];

/** He atom — simplest closed-shell test case (2 electrons). */
export const he: Atom[] = [
  { symbol: "He", pos: [0, 0, 0] },
];

/** Li atom — simplest open-shell second-row test case (3 electrons, doublet). */
export const li: Atom[] = [
  { symbol: "Li", pos: [0, 0, 0] },
];

/** LiH at experimental r_e = 1.5957 Å. */
export const liH: Atom[] = [
  { symbol: "Li", pos: [0, 0, 0]      },
  { symbol: "H",  pos: [0, 0, 1.5957] },
];

/** H₂O at experimental geometry (r_OH = 0.9572 Å, ∠HOH = 104.52°). */
export const h2o: Atom[] = (() => {
  const half = (104.52 / 2) * Math.PI / 180;
  const xH = 0.9572 * Math.sin(half);
  const zH = 0.9572 * Math.cos(half);
  return [
    { symbol: "O", pos: [0, 0, 0] },
    { symbol: "H", pos: [ xH, 0, zH] },
    { symbol: "H", pos: [-xH, 0, zH] },
  ];
})();

/** NH₃ at experimental geometry (r_NH = 1.0124 Å, ∠HNH = 106.7°). */
export const nh3: Atom[] = (() => {
  // C₃v symmetry: N on z-axis, three H atoms equally spaced.
  const rNH = 1.0124;
  const theta = 106.7 * Math.PI / 180;        // H-N-H bond angle
  // Distance from N to centroid of 3 H atoms (along -z):
  // cos(N-centroid-H) = (1 + 2 cos θ) / 3 sin² ...  derive directly:
  // place N at origin; H atoms at angle β from +z where β > 90°.
  // For pyramidal C₃v: ∠HNH = θ between any two H-N bonds.
  // Geometry: each H is at (r·sin β · cos φ_k, r·sin β · sin φ_k, r·cos β)
  // with φ_k = 0, 2π/3, 4π/3.
  // ∠HNH between two H's: cos(∠HNH) = sin²β · cos(2π/3) + cos²β
  //                                  = -½·sin²β + cos²β = (3·cos²β − 1)/2
  // ⇒ cos²β = (1 + 2·cos θ) / 3
  const cos2b = (1 + 2 * Math.cos(theta)) / 3;
  const cosb = -Math.sqrt(cos2b);              // pyramid down (H atoms below N)
  const sinb = Math.sqrt(1 - cos2b);
  const result: Atom[] = [{ symbol: "N", pos: [0, 0, 0] }];
  for (let k = 0; k < 3; k++) {
    const phi = k * 2 * Math.PI / 3;
    result.push({
      symbol: "H",
      pos: [rNH * sinb * Math.cos(phi), rNH * sinb * Math.sin(phi), rNH * cosb],
    });
  }
  return result;
})();

/** CH₄ at experimental r_CH = 1.0870 Å, tetrahedral. */
export const ch4: Atom[] = (() => {
  const r = 1.0870;
  // Tetrahedral H positions at corners of an inscribed tetrahedron.
  // Standard parametrization: (+1,+1,+1), (-1,-1,+1), (+1,-1,-1), (-1,+1,-1) / √3 · r.
  const s = r / Math.sqrt(3);
  return [
    { symbol: "C", pos: [0, 0, 0] },
    { symbol: "H", pos: [ s,  s,  s] },
    { symbol: "H", pos: [-s, -s,  s] },
    { symbol: "H", pos: [ s, -s, -s] },
    { symbol: "H", pos: [-s,  s, -s] },
  ];
})();

/** Linear BeH₂ at r_BeH = 1.33 Å. */
export const beh2: Atom[] = [
  { symbol: "Be", pos: [0, 0, 0]    },
  { symbol: "H",  pos: [0, 0,  1.33] },
  { symbol: "H",  pos: [0, 0, -1.33] },
];

/** F atom — open-shell test (9 electrons, doublet). */
export const f: Atom[] = [
  { symbol: "F", pos: [0, 0, 0] },
];

/** HF at experimental r_e = 0.917 Å. */
export const hf: Atom[] = [
  { symbol: "F", pos: [0, 0, 0]     },
  { symbol: "H", pos: [0, 0, 0.917] },
];

/** Library export — pick by name. Useful for demos and parameterized
 *  benchmarks ("run on every molecule in the library"). */
export const molecules = {
  atomicH, h2, he, li, liH, h2o, nh3, ch4, beh2, f, hf,
} as const;

export type MoleculeName = keyof typeof molecules;
