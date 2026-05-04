// ─────────────────────────────────────────────────────────────
// density.ts — sample the H₂ electron density ρ(r) on a 3D grid.
//
// HF reference (closed-shell σ_g doubly occupied):
//   ρ_HF(r) = 2 σ_g(r)²
//
// FCI ground state (closed-shell singlet):
//   |ψ⟩ = c_g |σ_g σ_g⟩ + c_u |σ_u σ_u⟩
//   ρ_FCI(r) = 2 [c_g² σ_g(r)² + c_u² σ_u(r)²]
//
// At equilibrium c_g ≈ 0.995, c_u ≈ −0.1 — the densities are
// nearly indistinguishable. At dissociation c_g, c_u → ±1/√2;
// the density splits into two atomic 1s lobes.
// ─────────────────────────────────────────────────────────────

import { phi1s } from "./orbital.js";
import { S_AB } from "../chemistry/integrals.js";

const ANGSTROM_TO_BOHR = 1 / 0.529177210903;

export interface DensityGrid {
  positions: Float32Array;
  densities: Float32Array;
  atomA: [number, number, number];
  atomB: [number, number, number];
  extentBohr: number;
  gridSize: number;
  maxDensity: number;
}

interface H2Geometry {
  A: [number, number, number];
  B: [number, number, number];
  invNg: number;
  invNu: number;
  extent: number;
}

function geometryAt(R_angstrom: number, extentBohr?: number): H2Geometry {
  const R_Bohr = R_angstrom * ANGSTROM_TO_BOHR;
  const extent = extentBohr ?? Math.max(2.5, R_Bohr / 2 + 2);
  const A: [number, number, number] = [0, 0, -R_Bohr / 2];
  const B: [number, number, number] = [0, 0,  R_Bohr / 2];
  const Sab = S_AB(A, B);
  const Ng = Math.sqrt(2 * (1 + Sab));
  const Nu = Math.sqrt(2 * (1 - Sab));
  return { A, B, invNg: 1 / Ng, invNu: 1 / Nu, extent };
}

/** Closed-shell HF density at bond length R (Å). */
export function densityGridHF(R_angstrom: number, gridSize = 48, extentBohr?: number): DensityGrid {
  return densityGridFromCoeffs(R_angstrom, 1, 0, gridSize, extentBohr);
}

/** FCI density given (c_g, c_u) — assumes c_g² + c_u² = 1 (closed-shell singlet). */
export function densityGridFromCoeffs(
  R_angstrom: number,
  cG: number,
  cU: number,
  gridSize = 48,
  extentBohr?: number,
): DensityGrid {
  const geo = geometryAt(R_angstrom, extentBohr);
  const { A, B, invNg, invNu, extent } = geo;

  const N = gridSize;
  const N3 = N * N * N;
  const positions = new Float32Array(N3 * 3);
  const densities = new Float32Array(N3);

  let pi = 0;
  let di = 0;
  let maxD = 0;

  const step = (2 * extent) / N;
  const start = -extent + step / 2;
  const cG2 = cG * cG;
  const cU2 = cU * cU;

  for (let ix = 0; ix < N; ix++) {
    const x = start + ix * step;
    for (let iy = 0; iy < N; iy++) {
      const y = start + iy * step;
      for (let iz = 0; iz < N; iz++) {
        const z = start + iz * step;
        positions[pi++] = x;
        positions[pi++] = y;
        positions[pi++] = z;

        const phiA = phi1s(x - A[0], y - A[1], z - A[2]);
        const phiB = phi1s(x - B[0], y - B[1], z - B[2]);
        const sigmaG = (phiA + phiB) * invNg;
        const sigmaU = (phiA - phiB) * invNu;
        const rho = 2 * (cG2 * sigmaG * sigmaG + cU2 * sigmaU * sigmaU);
        densities[di++] = rho;
        if (rho > maxD) maxD = rho;
      }
    }
  }

  return {
    positions,
    densities,
    atomA: A,
    atomB: B,
    extentBohr: extent,
    gridSize: N,
    maxDensity: maxD,
  };
}
